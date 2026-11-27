"""
Translation Module for Offline Video Subtitle Translation

This module is 100% OFFLINE once the models are on disk - no API calls, no
rate limits. Backends, in order of preference:

1. NLLB-200-distilled-600M through CTranslate2 (int8 on CPU, float16/int8 on
   GPU). The CTranslate2 copy is created once from the official Hugging Face
   checkpoint (PyTorch is only needed for that one-time conversion) and is
   4-7x faster than the Transformers path on CPU with the same quality.
2. NLLB-200-distilled-600M through HuggingFace Transformers (GPU when
   available, otherwise CPU float32).
3. opus-mt (Helsinki-NLP) EN->VI and JA->EN->VI pipelines.
4. Argos Translate (set up lazily, only when everything above failed).
"""

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import gc
import os
import queue
import re
import shutil
import tempfile
import threading

from loguru import logger

# Optional backends are imported lazily (inside the functions that need them)
# so that importing this module stays cheap and never fails.
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CT2_DIR = "models/nllb-200-distilled-600M-ct2-int8"

# Whisper language code -> NLLB-200 language code.
NLLB_LANG_CODES: Dict[str, str] = {
    "en": "eng_Latn", "vi": "vie_Latn", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "yue": "yue_Hant", "ko": "kor_Hang", "th": "tha_Thai", "id": "ind_Latn",
    "ms": "zsm_Latn", "tl": "tgl_Latn", "km": "khm_Khmr", "lo": "lao_Laoo",
    "my": "mya_Mymr", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "pt": "por_Latn", "it": "ita_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl",
    "uk": "ukr_Cyrl", "pl": "pol_Latn", "cs": "ces_Latn", "tr": "tur_Latn",
    "ar": "arb_Arab", "fa": "pes_Arab", "he": "heb_Hebr", "hi": "hin_Deva",
    "bn": "ben_Beng", "ta": "tam_Taml", "ur": "urd_Arab", "sv": "swe_Latn",
    "da": "dan_Latn", "no": "nob_Latn", "fi": "fin_Latn", "el": "ell_Grek",
    "hu": "hun_Latn", "ro": "ron_Latn",
}

# Every Vietnamese letter with a diacritic, and the subset that does not occur
# in common Western European words (so "café" or "Pokémon" are not mistaken
# for Vietnamese).
_VI_DIACRITIC_CHARS = frozenset(
    "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
)
_VI_SPECIFIC_CHARS = frozenset(
    "ạảầấậẩẫăằắặẳẵẹẻẽềếệểễịỉĩọỏồốộổỗơờớợởỡụủũưừứựửữỳỵỷỹđ"
)
# Scripts written without spaces between words (Japanese kana, CJK ideographs).
# Korean Hangul uses spaces, so it is handled like space-separated text.
_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
# Sentence boundaries: ".", "!" or "?" followed by whitespace (not "...", not
# "3.5"), or CJK full stops with no space needed.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[^.][.!?])\s+(?=\S)|(?<=[。！？])")
# A piece ending like this is not a sentence end ("Mr. Smith", "J. K. Rowling").
_ABBREVIATION_END_RE = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|vs|etc|No|Vol|Fig|Inc|Ltd|Co|Corp)\.|\b[A-Z]\.)$"
)
# English function words that never stand alone in Vietnamese text; one of
# them means "English sentence with Vietnamese names", not Vietnamese.
_EN_FUNCTION_WORDS = frozenset(
    "the and is are was were you we he she they my your this that with for of "
    "i love welcome".split()
)


@dataclass
class TranslationResult:
    """Represents a translation result with metadata."""

    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    timestamp: datetime
    service_used: str


class TranslationCache:
    """Thread-safe in-memory LRU cache for translation results."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize the translation cache.

        Args:
            max_size: Maximum number of cached translations
        """
        self.cache: "OrderedDict[Tuple[str, str, str], TranslationResult]" = OrderedDict()
        self.max_size = max(1, int(max_size))
        self._lock = threading.Lock()

    def get(
        self, text: str, source_lang: str, target_lang: str
    ) -> Optional[TranslationResult]:
        """
        Get cached translation result.

        Args:
            text: Original (cleaned) text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Cached translation result or None
        """
        key = (source_lang, target_lang, text)
        with self._lock:
            result = self.cache.get(key)
            if result is not None:
                self.cache.move_to_end(key)
            return result

    def set(self, result: TranslationResult) -> None:
        """
        Store translation result in cache.

        Args:
            result: Translation result to cache
        """
        key = (result.source_language, result.target_language, result.original_text)
        with self._lock:
            self.cache[key] = result
            self.cache.move_to_end(key)
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """Remove every cached entry."""
        with self._lock:
            self.cache.clear()


def resolve_compute_candidates(
    device_pref: str, compute_pref: str, gpu_order: List[str]
) -> List[Tuple[str, str]]:
    """Build the (device, compute_type) list to try, best first, CPU last.

    Uses CTranslate2's own capability query, so GPUs without efficient
    float16 (e.g. Pascal / GTX 10xx, compute capability 6.x) get int8 or
    float32 instead of failing with "Requested float16 compute type".

    Args:
        device_pref: "auto", "cuda" or "cpu"
        compute_pref: "auto" or an explicit CTranslate2 compute type
        gpu_order: Preferred GPU compute types when compute_pref is "auto"

    Returns:
        Ordered list of (device, compute_type) candidates
    """
    import ctranslate2

    candidates: List[Tuple[str, str]] = []
    device_pref = (device_pref or "auto").lower()
    compute_pref = (compute_pref or "auto").lower()

    if device_pref in ("auto", "cuda", "gpu"):
        try:
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False
        if has_cuda:
            supported = set(ctranslate2.get_supported_compute_types("cuda"))
            if compute_pref == "auto":
                order = list(gpu_order)
            elif compute_pref == "default" or compute_pref in supported:
                # "default" = the model's own type, converted by CTranslate2 if needed
                order = [compute_pref] + [c for c in gpu_order if c != compute_pref]
            else:
                logger.warning(
                    f"compute_type '{compute_pref}' is not supported by this GPU "
                    f"(supported: {', '.join(sorted(supported))}) - using the best supported type"
                )
                order = list(gpu_order)
            for compute_type in order:
                if compute_type == "default" or compute_type in supported:
                    candidates.append(("cuda", compute_type))
            if not candidates:
                logger.warning("No usable GPU compute type - using CPU")
        elif device_pref in ("cuda", "gpu"):
            logger.warning("CUDA requested but no CUDA device found - using CPU")

    cpu_supported = ctranslate2.get_supported_compute_types("cpu")
    cpu_type = compute_pref if compute_pref in cpu_supported else "int8"
    if cpu_type not in cpu_supported:
        cpu_type = "float32"
    candidates.append(("cpu", cpu_type))
    if cpu_type != "float32":
        candidates.append(("cpu", "float32"))
    return candidates


def _local_snapshot(repo_id: str) -> Optional[str]:
    """Folder of a cached Hugging Face snapshot, resolved without network access.

    Args:
        repo_id: Hugging Face repository id

    Returns:
        Local snapshot path, or None if the repo was never downloaded
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return None


def _has_weights(path: str) -> bool:
    """True if a snapshot folder holds model weights (an interrupted download may not)."""
    return any(
        os.path.isfile(os.path.join(path, name))
        for name in ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
    )


def _hf_from_pretrained(loader: Any, name: str, **kwargs: Any) -> Any:
    """Load from the local Hugging Face cache by path; download only if missing.

    Loading by path avoids Hub round-trips on every start (each one retries with
    backoff and can take ~20 s per file when the network is down) and avoids
    transformers 4.57's tokenizer check that calls the Hub API for repo ids and
    fails offline even when every file is cached.

    Args:
        loader: A class with ``from_pretrained`` (tokenizer/model)
        name: Model id or local path
        **kwargs: Extra arguments for ``from_pretrained``

    Returns:
        The loaded object
    """
    path = name if os.path.isdir(name) else _local_snapshot(name)
    if path:
        try:
            return loader.from_pretrained(path, **kwargs)
        except Exception as e:
            logger.debug(f"Loading {name} from the local cache failed ({e}); trying the Hub")
    return loader.from_pretrained(name, **kwargs)


class _NllbBackendBase:
    """Shared batching logic for NLLB backends (sort by length, cap length)."""

    name = "nllb"

    def __init__(self, tokenizer: Any, beam_size: int, batch_size: int):
        self.tokenizer = tokenizer
        self.beam_size = max(1, int(beam_size))
        self.batch_size = max(1, int(batch_size))

    @staticmethod
    def max_new_tokens(source_len: int) -> int:
        """Generous output cap (FLORES vi/src token ratio never exceeds ~2.1).

        Prevents a degenerate beam from decoding up to 1024 tokens and holding
        the whole batch hostage.

        Args:
            source_len: Number of source tokens

        Returns:
            Maximum number of tokens to generate
        """
        return min(1024, 3 * source_len + 16)

    def translate(self, texts: List[str], src_code: str, tgt_code: str) -> List[str]:
        """Translate texts (any order); returns translations in input order.

        Args:
            texts: Source texts
            src_code: NLLB source language code (e.g. eng_Latn)
            tgt_code: NLLB target language code (e.g. vie_Latn)

        Returns:
            Translated texts, same order as ``texts``
        """
        if not texts:
            return []
        self.tokenizer.src_lang = src_code
        encoded = self.tokenizer(list(texts), truncation=True, max_length=1024)["input_ids"]
        order = sorted(range(len(texts)), key=lambda i: len(encoded[i]))
        outputs = [""] * len(texts)
        start = 0
        while start < len(order):
            idx = order[start : start + self.batch_size]
            try:
                decoded = self._translate_ids([encoded[i] for i in idx], tgt_code)
            except Exception as e:
                # GPU out of memory: retry this chunk with half the batch size
                if "out of memory" in str(e).lower() and self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    logger.warning(f"Translation OOM - batch size reduced to {self.batch_size}")
                    self._free_memory()
                    continue
                raise
            for i, text in zip(idx, decoded):
                outputs[i] = text.strip()
            start += len(idx)
        return outputs

    def _free_memory(self) -> None:
        """Release cached GPU memory after an out-of-memory error."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _translate_ids(self, batch: List[List[int]], tgt_code: str) -> List[str]:
        raise NotImplementedError


class _Ct2NllbBackend(_NllbBackendBase):
    """NLLB through CTranslate2 (fast int8 / float16 inference)."""

    name = "nllb_ct2"

    def __init__(self, translator: Any, tokenizer: Any, beam_size: int, batch_size: int):
        super().__init__(tokenizer, beam_size, batch_size)
        self.translator = translator

    def _translate_ids(self, batch: List[List[int]], tgt_code: str) -> List[str]:
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in batch]
        max_len = max(len(t) for t in tokens)
        results = self.translator.translate_batch(
            tokens,
            target_prefix=[[tgt_code]] * len(tokens),
            beam_size=self.beam_size,
            max_batch_size=self.batch_size,
            max_decoding_length=self.max_new_tokens(max_len),
        )
        decoded = []
        for result in results:
            hyp = result.hypotheses[0]
            if hyp and hyp[0] == tgt_code:
                hyp = hyp[1:]
            decoded.append(
                self.tokenizer.decode(
                    self.tokenizer.convert_tokens_to_ids(hyp), skip_special_tokens=True
                )
            )
        return decoded


class _HfNllbBackend(_NllbBackendBase):
    """NLLB through HuggingFace Transformers (fallback when CTranslate2 is unavailable)."""

    name = "nllb_transformers"

    def __init__(self, model: Any, tokenizer: Any, device: str, beam_size: int, batch_size: int):
        super().__init__(tokenizer, beam_size, batch_size)
        self.model = model
        self.device = device

    def _translate_ids(self, batch: List[List[int]], tgt_code: str) -> List[str]:
        import torch

        padded = self.tokenizer.pad({"input_ids": batch}, return_tensors="pt")
        padded = {k: v.to(self.device) for k, v in padded.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **padded,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
                num_beams=self.beam_size,
                max_new_tokens=self.max_new_tokens(padded["input_ids"].shape[1]),
            )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


class TranslationService:
    """Handles offline translation with NLLB (CTranslate2/Transformers) and fallbacks."""

    def __init__(self, config: dict):
        """
        Initialize the translation service.

        This build is strictly OFFLINE at inference time. Only local models are
        used, guaranteeing privacy and unlimited usage without rate limits.

        Args:
            config: Full application configuration dictionary
        """
        self.config = config.get("translation", {}) or {}
        perf_config = config.get("performance", {}) or {}
        self.target_language = self.config.get("target_language", "vi")

        # Generation settings shared by every NLLB backend
        self.beam_size = int(self.config.get("beam_size", 4))
        self.batch_size = int(self.config.get("batch_size", 32))

        # Retry attempts for local model errors (minimal, usually 1 attempt is enough)
        self.retry_attempts = max(1, int(self.config.get("retry_attempts", 1)))

        # Backends (OFFLINE only)
        self.backend: Optional[_NllbBackendBase] = None
        self.backend_device = "cpu"
        self.local_models: Dict[str, Dict[str, Any]] = {}  # opus-mt fallbacks
        self.translators: Dict[str, str] = {}
        self._argos_state: Optional[bool] = None  # None = not checked yet
        self._warned_lang_codes: set = set()

        # Translation cache
        self.cache = TranslationCache(perf_config.get("max_cache_size", 1000))

        # Models and HF fast tokenizers are not safe to use from several
        # threads at once, so every inference call is serialized.
        self._model_lock = threading.Lock()

        # Processing queue for async (real-time) translations
        self.translation_queue: "queue.Queue" = queue.Queue(maxsize=50)
        self.worker_threads: List[threading.Thread] = []
        self.num_workers = perf_config.get("translation_workers", 1)
        self.is_processing = False

        # Callbacks
        self.on_translation_complete: Optional[Callable[[TranslationResult], None]] = None

        self._initialize_translators()
        logger.info("TranslationService initialized in OFFLINE-ONLY mode")

    # ------------------------------------------------------------------ setup

    def _initialize_translators(self) -> None:
        """Initialize the best available backend; heavier fallbacks only if needed."""
        backend_pref = str(self.config.get("backend", "auto")).lower()
        try:
            if backend_pref in ("auto", "ctranslate2", "ct2"):
                self.backend = self._load_ct2_backend()
            if self.backend is None and backend_pref in ("auto", "transformers", "ctranslate2", "ct2"):
                self.backend = self._load_transformers_backend()
            if self.backend is not None:
                for src in NLLB_LANG_CODES:
                    self.translators[f"nllb_{src}_vi"] = "nllb"
            else:
                self._setup_opus_fallbacks()
        except Exception as e:
            logger.error(f"Error initializing translators: {e}")

        if self.backend is None and not self.local_models:
            logger.warning(
                "No neural translation model available - Argos Translate will be "
                "tried on demand; untranslatable lines keep the original text"
            )

    def _resolve_ct2_dir(self) -> Path:
        model_dir = Path(self.config.get("ct2_model_dir", DEFAULT_CT2_DIR)).expanduser()
        if not model_dir.is_absolute():
            model_dir = PROJECT_DIR / model_dir
        return model_dir

    def _load_ct2_backend(self) -> Optional[_NllbBackendBase]:
        """Load NLLB with CTranslate2, converting the official checkpoint once if needed."""
        try:
            import ctranslate2
            from transformers import AutoTokenizer
        except ImportError as e:
            logger.info(f"CTranslate2 backend unavailable ({e})")
            return None

        model_dir = self._resolve_ct2_dir()
        if not (model_dir / "model.bin").exists():
            if not self.config.get("auto_convert", True):
                logger.info(f"CTranslate2 model not found at {model_dir} (auto_convert disabled)")
                return None
            if not self._convert_nllb_to_ct2(model_dir):
                return None

        try:
            # fix_mistral_regex=False: NLLB is not a Mistral tokenizer; without it
            # transformers 4.57 prints a misleading "incorrect regex" warning
            # for folders whose config.json is CTranslate2's (tokenization is unchanged).
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), fix_mistral_regex=False)
        except Exception as e:
            logger.warning(f"Could not load NLLB tokenizer from {model_dir}: {e}")
            return None

        cpu_threads = int(self.config.get("cpu_threads", 4))
        candidates = resolve_compute_candidates(
            self.config.get("device", "auto"),
            self.config.get("compute_type", "auto"),
            ["int8_float16", "float16", "int8", "float32"],
        )
        cuda_broken = False
        cuda_oom = False
        for device, compute_type in candidates:
            if device == "cuda" and cuda_broken:
                continue
            if device == "cuda" and cuda_oom and compute_type == "float32":
                continue  # needs more VRAM than the type that just ran out of memory
            translator = None
            try:
                translator = ctranslate2.Translator(
                    str(model_dir),
                    device=device,
                    compute_type=compute_type,
                    intra_threads=cpu_threads if device == "cpu" else 0,
                    inter_threads=1,
                )
                backend = _Ct2NllbBackend(translator, tokenizer, self.beam_size, self.batch_size)
                # Warm-up: CUDA libraries (cuBLAS) are only loaded on first use,
                # so a broken GPU setup must fail here, not halfway through a job.
                backend.translate(["Hello."], "eng_Latn", "vie_Latn")
                self.backend_device = device
                logger.info(
                    f"NLLB translation model loaded (CTranslate2, {device}, {compute_type}, "
                    f"beam={self.beam_size})"
                )
                return backend
            except Exception as e:
                logger.warning(f"CTranslate2 NLLB on {device}/{compute_type} failed: {e}")
                # Free the failed model before trying the next (smaller) type
                translator = None
                backend = None
                gc.collect()
                # Missing CUDA libraries/driver problems affect every GPU compute
                # type; only out-of-memory is worth retrying with a smaller type.
                if device == "cuda":
                    if "out of memory" in str(e).lower():
                        cuda_oom = True
                    else:
                        cuda_broken = True
        return None

    def _convert_nllb_to_ct2(self, model_dir: Path) -> bool:
        """One-time conversion of the official NLLB checkpoint to CTranslate2 int8.

        Args:
            model_dir: Destination directory

        Returns:
            True if the converted model is ready
        """
        # Never delete a folder we did not create: if ct2_model_dir points at an
        # existing non-empty folder that is not a CTranslate2 model, stop here.
        if model_dir.exists() and (not model_dir.is_dir() or any(model_dir.iterdir())):
            logger.error(
                f"{model_dir} exists and is not a CTranslate2 model - not overwriting it. "
                "Point translation.ct2_model_dir to a new or empty folder."
            )
            return False

        try:
            import ctranslate2.converters
            import torch  # noqa: F401  (required by the converter)
            from transformers import AutoTokenizer
        except ImportError as e:
            logger.info(f"Skipping CTranslate2 conversion ({e}); using Transformers backend")
            return False

        tmp_dir: Optional[Path] = None
        try:
            logger.info(
                f"Converting {NLLB_MODEL_NAME} to CTranslate2 int8 (one-time, "
                f"~1-3 min, ~620 MB) -> {model_dir}"
            )
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(tempfile.mkdtemp(prefix=model_dir.name + ".tmp-", dir=model_dir.parent))
            # Convert from the local snapshot folder (download it once if needed):
            # loading by path keeps the conversion working offline.
            source = _local_snapshot(NLLB_MODEL_NAME)
            if source is None or not _has_weights(source):
                from huggingface_hub import snapshot_download

                logger.info(f"Downloading {NLLB_MODEL_NAME} (~2.5 GB, first run only)...")
                source = snapshot_download(NLLB_MODEL_NAME)
            converter = ctranslate2.converters.TransformersConverter(source)
            converter.convert(str(tmp_dir), quantization="int8", force=True)
            AutoTokenizer.from_pretrained(source).save_pretrained(str(tmp_dir))
            if model_dir.exists():
                model_dir.rmdir()  # empty (checked above)
            tmp_dir.rename(model_dir)
            logger.info("CTranslate2 conversion finished")
            return True
        except Exception as e:
            logger.warning(f"CTranslate2 conversion failed: {e}")
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return False

    def _load_transformers_backend(self) -> Optional[_NllbBackendBase]:
        """Load NLLB with Transformers, on the GPU when one is usable."""
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            logger.info(f"Transformers backend unavailable ({e})")
            return None

        try:
            logger.info(f"Loading NLLB translation model: {NLLB_MODEL_NAME}")
            logger.info("This may take a moment on first load...")
            tokenizer = _hf_from_pretrained(AutoTokenizer, NLLB_MODEL_NAME)
            model = _hf_from_pretrained(AutoModelForSeq2SeqLM, NLLB_MODEL_NAME)
            model.eval()

            device = "cpu"
            device_pref = str(self.config.get("device", "auto")).lower()
            if device_pref in ("auto", "cuda", "gpu") and torch.cuda.is_available():
                try:
                    major, _ = torch.cuda.get_device_capability(0)
                    # float16 only where it is fast (Volta+); Pascal stays float32.
                    dtype = torch.float16 if major >= 7 else torch.float32
                    free_bytes, _ = torch.cuda.mem_get_info()
                    needed = 1.6e9 if dtype == torch.float16 else 3.0e9
                    if free_bytes >= needed:
                        model = model.to("cuda", dtype=dtype)
                        device = "cuda"
                        _ = self._hf_smoke_test(model, tokenizer, device)
                    else:
                        logger.info(
                            f"Only {free_bytes / 1e9:.1f} GB VRAM free - NLLB stays on CPU"
                        )
                except Exception as e:
                    logger.warning(f"NLLB on GPU failed ({e}) - using CPU")
                    model = model.to("cpu", dtype=torch.float32)
                    device = "cpu"
                    torch.cuda.empty_cache()

            self.backend_device = device
            logger.info(f"NLLB translation model loaded (Transformers, {device})")
            return _HfNllbBackend(model, tokenizer, device, self.beam_size, self.batch_size)
        except Exception as e:
            logger.warning(f"Failed to load NLLB model: {e}")
            return None

    def _hf_smoke_test(self, model: Any, tokenizer: Any, device: str) -> str:
        backend = _HfNllbBackend(model, tokenizer, device, 1, 1)
        return backend.translate(["Hello."], "eng_Latn", "vie_Latn")[0]

    def _setup_opus_fallbacks(self) -> None:
        """Load lighter opus-mt models when NLLB is unavailable."""
        try:
            from transformers import MarianMTModel, MarianTokenizer, pipeline
        except ImportError:
            return

        logger.info("Falling back to lighter opus-mt models...")
        for src_lang, tgt_lang, model_name in [
            ("en", "vi", "Helsinki-NLP/opus-mt-en-vi"),
            ("ja", "en", "Helsinki-NLP/opus-mt-ja-en"),
        ]:
            try:
                logger.info(f"Loading fallback model: {model_name}")
                tokenizer = _hf_from_pretrained(MarianTokenizer, model_name)
                model = _hf_from_pretrained(MarianMTModel, model_name)
                key = f"local_{src_lang}_{tgt_lang}"
                self.local_models[key] = {
                    "pipeline": pipeline("translation", model=model, tokenizer=tokenizer),
                    "max_length": 512,
                }
                self.translators[key] = "local"
                logger.info(f"Fallback model {model_name} loaded")
            except Exception as e:
                logger.warning(f"Failed to load fallback model {model_name}: {e}")

    def _ensure_argos(self) -> bool:
        """Set up Argos Translate on first use (it is the last-resort fallback)."""
        if self._argos_state is not None:
            return self._argos_state
        self._argos_state = False
        try:
            import argostranslate.package
            import argostranslate.translate  # noqa: F401

            installed = argostranslate.package.get_installed_packages()
            if not any(p.from_code == "en" and p.to_code == "vi" for p in installed):
                available = [
                    p
                    for p in argostranslate.package.get_available_packages()
                    if p.from_code == "en" and p.to_code == "vi"
                ]
                if available:
                    logger.info("Installing English to Vietnamese Argos package...")
                    argostranslate.package.install_from_path(available[0].download())
            self._argos_state = True
            self.translators["argos_offline"] = "argos"
            logger.info("Argos Translate (offline) initialized")
        except Exception as e:
            logger.warning(f"Argos Translate unavailable: {e}")
        return self._argos_state

    # ------------------------------------------------------------ async API

    def start_async_processing(self) -> None:
        """Start asynchronous translation processing."""
        if not self.is_processing:
            self.is_processing = True
            # Inference is serialized by _model_lock, so extra workers only
            # help overlap the queue handling; keep the configured count.
            self.worker_threads = []
            for i in range(max(1, int(self.num_workers))):
                t = threading.Thread(
                    target=self._process_translation_queue_worker, daemon=True
                )
                t.name = f"TranslationWorker-{i+1}"
                t.start()
                self.worker_threads.append(t)

            logger.info(
                f"Async translation processing started with {len(self.worker_threads)} workers"
            )

    def stop_async_processing(self) -> None:
        """Stop asynchronous translation processing."""
        self.is_processing = False
        for t in self.worker_threads:
            try:
                if t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                pass

        self.worker_threads = []
        logger.info("Async translation processing stopped")

    def translate_async(
        self,
        text: str,
        source_language: str = "auto",
        callback: Optional[Callable[[TranslationResult], None]] = None,
    ) -> None:
        """
        Queue text for asynchronous translation.

        Args:
            text: Text to translate
            source_language: Source language code
            callback: Optional per-request callback (in addition to the global one)
        """
        if not self.is_processing:
            self.start_async_processing()

        try:
            self.translation_queue.put((text, source_language, callback), timeout=1.0)
        except queue.Full:
            logger.warning("Translation queue full, dropping request")

    def _process_translation_queue_worker(self) -> None:
        """Worker function for processing translations from the async queue."""
        while self.is_processing:
            try:
                text, source_lang, callback = self.translation_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                result = self.translate_text(text, source_lang)
                if not result:
                    continue
                callbacks = [
                    cb for cb in (callback, self.on_translation_complete) if cb is not None
                ]
                if len(callbacks) == 2 and callbacks[0] == callbacks[1]:
                    callbacks = callbacks[:1]  # same handler registered twice: call once
                for cb in callbacks:
                    try:
                        cb(result)
                    except Exception as e:
                        logger.error(f"Error in translation callback: {e}")
            except Exception as e:
                logger.error(f"Error processing translation queue: {e}")

    # ----------------------------------------------------------- sync API

    def translate_batch(
        self, texts: List[str], source_language: str = "auto", use_context: bool = True
    ) -> List[Optional[TranslationResult]]:
        """
        Translate multiple texts in one batched call (sorted by length internally).

        Args:
            texts: List of texts to translate
            source_language: Source language code (Whisper code, e.g. "en", "ja")
            use_context: Reserved for context-aware translation (unused)

        Returns:
            List of translation results; None for empty/untranslatable inputs
        """
        results: List[Optional[TranslationResult]] = [None] * len(texts)
        pending: Dict[str, List[int]] = {}

        for i, text in enumerate(texts):
            cleaned = self._clean_text(text)
            if not cleaned:
                continue  # nothing left to translate (e.g. "[Music]", "♪")

            # Whisper was told (or detected) the source language; only guess
            # when it is unknown, so English lines with "café" are never skipped.
            src = (source_language or "auto").lower()
            if src == self.target_language or (src == "auto" and self._is_target_language(cleaned)):
                results[i] = self._make_result(
                    cleaned, cleaned, self.target_language, 1.0, "no_translation"
                )
                continue

            cached = self.cache.get(cleaned, source_language, self.target_language)
            if cached:
                results[i] = cached
            else:
                pending.setdefault(cleaned, []).append(i)

        if not pending:
            return results

        unique_texts = list(pending)
        translations, service = self._translate_many(unique_texts, source_language)
        for cleaned, translated in zip(unique_texts, translations):
            if not translated or not translated.strip():
                continue
            result = self._make_result(cleaned, translated, source_language, 0.9, service)
            self.cache.set(result)
            for i in pending[cleaned]:
                results[i] = result
        return results

    def translate_text(
        self, text: str, source_language: str = "auto"
    ) -> Optional[TranslationResult]:
        """
        Translate text to the target language (same path/quality as batch mode).

        Args:
            text: Text to translate
            source_language: Source language code

        Returns:
            Translation result or None if failed
        """
        if not text or not text.strip():
            return None
        result = self.translate_batch([text], source_language)[0]
        if result:
            logger.info(
                f"Translated: '{result.original_text[:50]}' -> '{result.translated_text[:50]}'"
            )
        else:
            logger.warning(f"Failed to translate: {text[:50]}...")
        return result

    def _make_result(
        self, original: str, translated: str, source: str, confidence: float, service: str
    ) -> TranslationResult:
        return TranslationResult(
            original_text=original,
            translated_text=translated,
            source_language=source,
            target_language=self.target_language,
            confidence=confidence,
            timestamp=datetime.now(),
            service_used=service,
        )

    def _nllb_code(self, lang: str) -> str:
        code = NLLB_LANG_CODES.get((lang or "").lower())
        if code:
            return code
        if lang not in self._warned_lang_codes:
            self._warned_lang_codes.add(lang)
            logger.warning(f"No NLLB code for source language '{lang}', treating it as English")
        return "eng_Latn"

    def _translate_many(
        self, texts: List[str], source_language: str
    ) -> Tuple[List[Optional[str]], str]:
        """Translate unique cleaned texts with the best available backend.

        Long texts are split into sentence chunks first and re-joined after.

        Args:
            texts: Cleaned texts
            source_language: Whisper language code

        Returns:
            (translations aligned with ``texts`` or None, service name)
        """
        pieces: List[str] = []
        owners: List[int] = []
        for i, text in enumerate(texts):
            for chunk in self._split_text_into_chunks(text):
                pieces.append(chunk)
                owners.append(i)

        translated = self._translate_pieces(pieces, source_language)
        if translated is None:
            return [None] * len(texts), "none"
        out_pieces, service = translated

        joined: List[List[str]] = [[] for _ in texts]
        for owner, piece in zip(owners, out_pieces):
            if piece:
                joined[owner].append(piece)
        return [" ".join(parts) if parts else None for parts in joined], service

    def _translate_pieces(
        self, pieces: List[str], source_language: str
    ) -> Optional[Tuple[List[str], str]]:
        tgt_code = NLLB_LANG_CODES.get(self.target_language, "vie_Latn")

        if self.backend is not None:
            for attempt in range(self.retry_attempts):
                try:
                    with self._model_lock:
                        out = self.backend.translate(
                            pieces, self._nllb_code(source_language), tgt_code
                        )
                    return out, f"{self.backend.name}_{source_language}_vi"
                except Exception as e:
                    logger.warning(
                        f"{self.backend.name} translation attempt {attempt + 1} failed: {e}"
                    )

        # opus-mt fallbacks (EN->VI, or JA->EN->VI)
        en_vi = self.local_models.get("local_en_vi")
        ja_en = self.local_models.get("local_ja_en")
        try:
            if en_vi and source_language == "ja" and ja_en:
                with self._model_lock:
                    english = [r["translation_text"] for r in ja_en["pipeline"](pieces)]
                    vietnamese = [r["translation_text"] for r in en_vi["pipeline"](english)]
                return vietnamese, "opus_ja_en+en_vi"
            if en_vi and source_language in ("en", "auto"):
                with self._model_lock:
                    vietnamese = [r["translation_text"] for r in en_vi["pipeline"](pieces)]
                return vietnamese, "local_en_vi"
        except Exception as e:
            logger.warning(f"opus-mt translation failed: {e}")

        # Last resort: Argos Translate
        if self._ensure_argos():
            import argostranslate.translate

            src = source_language if source_language not in (None, "", "auto") else "en"
            out = []
            for piece in pieces:
                try:
                    with self._model_lock:
                        out.append(
                            argostranslate.translate.translate(piece, src, self.target_language)
                        )
                except Exception as e:
                    logger.warning(f"Argos translation failed: {e}")
                    out.append("")
            return out, "argos_offline"
        return None

    # ---------------------------------------------------------- text utils

    def _split_text_into_chunks(self, text: str, max_tokens: int = 400) -> List[str]:
        """Split text into sentences; each is translated on its own and re-joined.

        NLLB is a sentence-level model: given two or three sentences at once it
        often stops after the first one, silently dropping the rest. "...",
        decimals ("3.5") and abbreviations ("Mr. Smith") are not split. A
        single sentence longer than the token budget is cut on whitespace.

        Args:
            text: Text to split
            max_tokens: Approximate token budget per piece (1 token ~ 3 chars)

        Returns:
            List of text pieces (in order, nothing duplicated)
        """
        if not text:
            return [text]

        sentences: List[str] = []
        for piece in _SENTENCE_SPLIT_RE.split(text):
            piece = piece.strip()
            if not piece:
                continue
            if sentences and _ABBREVIATION_END_RE.search(sentences[-1]):
                sentences[-1] = f"{sentences[-1]} {piece}"
            else:
                sentences.append(piece)

        # A single giant "sentence" is cut on whitespace as a last resort.
        final: List[str] = []
        limit = max_tokens * 3
        for sentence in sentences:
            while len(sentence) > limit:
                cut = sentence.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                final.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                final.append(sentence)
        return final or [text]

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for translation.
        Optimized for subtitle text processing.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text ("" when nothing translatable is left)
        """
        if not text:
            return ""

        # Remove excessive whitespace (faster with split/join)
        text = " ".join(text.split())

        # Remove common transcription artifacts
        text = re.sub(r"\[.*?\]", "", text)  # Remove [background noise] etc.
        text = re.sub(r"\(.*?\)", "", text)  # Remove (unclear) etc.

        # Remove music notes and sound effects
        text = re.sub(r"♪.*?♪", "", text)
        text = text.replace("♪", "")

        # Remove repeated punctuation (preserve meaningful ones)
        text = re.sub(r"\.{3,}", "...", text)  # Keep ellipsis
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)

        # Drop leading noise punctuation and trailing separators, but keep the
        # sentence-final mark - NLLB translates complete sentences better.
        text = " ".join(text.split()).lstrip(" .,;:").rstrip(" ,;:")

        # Nothing but punctuation/symbols left -> nothing to translate
        if not re.search(r"\w", text):
            return ""
        return text.strip()

    def detect_repetition(self, text: str, threshold: float = 0.8) -> Tuple[bool, str]:
        """Detect and remove repetitive patterns (Whisper hallucination)."""
        if not text or len(text) < 20:
            return False, text

        words = text.split()
        if len(words) < 4:
            return False, text

        seen_phrases: Dict[str, int] = {}
        phrase_length = min(5, len(words) // 2)

        for i in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[i : i + phrase_length])
            seen_phrases[phrase] = seen_phrases.get(phrase, 0) + 1

        max_repetitions = max(seen_phrases.values()) if seen_phrases else 1
        total_possible = len(words) - phrase_length + 1

        if max_repetitions > 2 and max_repetitions / total_possible > 0.5:
            repeated_phrase = max(seen_phrases, key=seen_phrases.get)
            logger.warning(
                f"Detected repetition pattern: '{repeated_phrase}' x{max_repetitions}"
            )

            cleaned_words: List[str] = []
            phrase_count = 0
            i = 0
            while i < len(words):
                current_phrase = " ".join(words[i : i + phrase_length])
                if current_phrase == repeated_phrase:
                    if phrase_count < 1:
                        cleaned_words.extend(words[i : i + phrase_length])
                    phrase_count += 1
                    i += phrase_length
                else:
                    cleaned_words.append(words[i])
                    i += 1

            return True, " ".join(cleaned_words)

        return False, text

    def detect_segment_repetition(
        self,
        segments: List[dict],
        similarity_threshold: float = 0.85,
        max_allowed_repetitions: int = 2,
    ) -> List[dict]:
        """Filter consecutive duplicate segments (Whisper hallucination).

        Args:
            segments: Segments with "text"
            similarity_threshold: Jaccard similarity treated as a repeat
            max_allowed_repetitions: Repeats kept before dropping the rest

        Returns:
            Filtered segments
        """
        if not segments or len(segments) < 2:
            return segments

        cleaned_segments = []
        prev_text = ""
        repetition_count = 0

        for segment in segments:
            current_text = segment.get("text", "").strip()

            if not current_text:
                continue

            similarity = self._calculate_text_similarity(prev_text, current_text)

            if similarity >= similarity_threshold:
                repetition_count += 1
                if repetition_count <= max_allowed_repetitions:
                    cleaned_segments.append(segment)
                else:
                    logger.debug(
                        f"Skipping repeated segment #{len(cleaned_segments)+1}: "
                        f"'{current_text[:50]}...' (similarity: {similarity:.2f})"
                    )
            else:
                repetition_count = 0
                cleaned_segments.append(segment)

            prev_text = current_text

        removed_count = len(segments) - len(cleaned_segments)
        if removed_count > 0:
            logger.info(
                f"🔄 Removed {removed_count} repetitive segments "
                f"({len(cleaned_segments)}/{len(segments)} remaining)"
            )

        return cleaned_segments

    @staticmethod
    def _similarity_tokens(text: str) -> set:
        """Words for space-separated scripts, character bigrams for CJK."""
        if _CJK_RE.search(text):
            chars = re.sub(r"[\s\W]+", "", text)
            return {chars[i : i + 2] for i in range(max(1, len(chars) - 1))}
        return set(re.findall(r"\w+", text.lower()))

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts (0.0 - 1.0)."""
        if not text1 or not text2:
            return 0.0

        tokens1 = self._similarity_tokens(text1)
        tokens2 = self._similarity_tokens(text2)
        if not tokens1 or not tokens2:
            return 0.0

        union = len(tokens1 | tokens2)
        return len(tokens1 & tokens2) / union if union else 0.0

    def _is_target_language(self, text: str) -> bool:
        """
        Check if text is likely already in the target language.

        Requires at least one letter that only Vietnamese uses (ơ, ư, đ, ạ, ả, ...),
        at least half of the words carrying Vietnamese diacritics (99.7% of
        FLORES Vietnamese sentences pass) and no English function words, so
        English lines with "café", "Pokémon" or "Hà Nội" are still translated.

        Args:
            text: Text to check

        Returns:
            True if text is likely in target language
        """
        if self.target_language != "vi":
            return False
        lowered = text.lower()
        if _VI_SPECIFIC_CHARS.isdisjoint(lowered):
            return False
        words = re.findall(r"\w+", lowered)
        if not words or not _EN_FUNCTION_WORDS.isdisjoint(words):
            return False
        accented = sum(1 for w in words if not _VI_DIACRITIC_CHARS.isdisjoint(w))
        return accented / len(words) >= 0.5

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get supported languages for translation.

        Returns:
            Dictionary of language codes and names
        """
        return {
            "en": "English",
            "vi": "Vietnamese",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "th": "Thai",
            "id": "Indonesian",
            "ms": "Malay",
        }

    def _contains_vietnamese_chars(self, text: str) -> bool:
        """Check if text contains Vietnamese characters."""
        return not _VI_DIACRITIC_CHARS.isdisjoint(text.lower())

    def set_callback(self, callback: Callable[[TranslationResult], None]) -> None:
        """
        Set callback function for translation completion.

        Args:
            callback: Function to call when translation completes
        """
        self.on_translation_complete = callback

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get translation cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cached_translations": len(self.cache.cache),
            "max_cache_size": self.cache.max_size,
            "cache_usage_percent": int(
                (len(self.cache.cache) / self.cache.max_size) * 100
            ),
        }

    def clear_cache(self) -> None:
        """Clear the translation cache."""
        self.cache.clear()
        logger.info("Translation cache cleared")

    def release_models(self) -> None:
        """Drop model references so RAM/VRAM can be reclaimed (e.g. before TTS)."""
        self.backend = None
        self.local_models.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_async_processing()
        self.clear_cache()
        logger.info("TranslationService cleanup completed")
