"""
Translation Module for Offline Video Subtitle Translation

This module is 100% OFFLINE - no internet connection required.
Only local models (Argos Translate and HuggingFace Transformers) are used.
Guarantees unlimited usage with no rate limits or API costs.
"""

from typing import Optional, Dict, List, Tuple
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
import re
import time

# Try to import Argos Translate (offline translation)
try:
    import argostranslate.package
    import argostranslate.translate

    ARGOS_AVAILABLE = True
except ImportError:
    ARGOS_AVAILABLE = False
    logger.warning("Argos Translate not available - install for offline translation")

# Try to import Transformers for local neural translation (offline if models are cached locally)
try:
    from transformers import (
        MarianMTModel,
        MarianTokenizer,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - install for local neural translation")


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
    """Simple in-memory cache for translation results."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize the translation cache.

        Args:
            max_size: Maximum number of cached translations
        """
        self.cache: Dict[str, TranslationResult] = {}
        self.max_size = max_size
        self.access_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def get(
        self, text: str, source_lang: str, target_lang: str
    ) -> Optional[TranslationResult]:
        """
        Get cached translation result.

        Args:
            text: Original text
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Cached translation result or None
        """
        cache_key = self._generate_key(text, source_lang, target_lang)

        with self._lock:
            if cache_key in self.cache:
                self.access_times[cache_key] = datetime.now()
                return self.cache[cache_key]

        return None

    def set(self, result: TranslationResult) -> None:
        """
        Store translation result in cache.

        Args:
            result: Translation result to cache
        """
        cache_key = self._generate_key(
            result.original_text, result.source_language, result.target_language
        )

        with self._lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                self._cleanup_cache()

            self.cache[cache_key] = result
            self.access_times[cache_key] = datetime.now()

    def _generate_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for the given parameters."""
        return f"{source_lang}:{target_lang}:{hash(text.lower().strip())}"

    def _cleanup_cache(self) -> None:
        """Remove least recently used cache entries."""
        if not self.access_times:
            return

        # Sort by access time and remove oldest 20%
        sorted_keys = sorted(
            self.access_times.keys(), key=lambda k: self.access_times[k]
        )

        remove_count = max(1, len(sorted_keys) // 5)

        for key in sorted_keys[:remove_count]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)


class TranslationService:
    """Handles translation services with multiple FREE and OPEN-SOURCE providers."""

    def __init__(self, config: dict):
        """
        Initialize the translation service.

        This build is strictly OFFLINE. Only local models (Argos Translate
        and/or HuggingFace Transformers) are used. This ensures 100% offline
        operation with no internet dependency, guaranteeing privacy and
        unlimited usage without rate limits.

        Args:
            config: Translation configuration dictionary
        """
        self.config = config["translation"]
        self.target_language = self.config.get("target_language", "vi")

        # Retry attempts for local model errors (minimal, usually 1 attempt is enough)
        self.retry_attempts = self.config.get("retry_attempts", 1)

        # Initialize translators (OFFLINE only)
        self.translators = {}
        self.local_models = {}
        self._initialize_translators()

        # Translation cache
        cache_size = config.get("performance", {}).get("max_cache_size", 100)
        self.cache = TranslationCache(cache_size)

        # Processing queue for async translations
        self.translation_queue = queue.Queue(maxsize=50)
        # Worker threads for parallel processing
        self.processing_thread = None
        self.worker_threads = []
        self.num_workers = config.get("performance", {}).get("translation_workers", 4)
        self.is_processing = False

        # Thread safety for model inference
        self._model_lock = threading.Lock()

        # Callbacks
        self.on_translation_complete = None

        logger.info("TranslationService initialized in OFFLINE-ONLY mode")

    def _initialize_translators(self) -> None:
        """Initialize FREE and OPEN-SOURCE translation services only."""
        try:
            # Argos Translate (completely offline and free)
            if ARGOS_AVAILABLE:
                self._setup_argos_translate()

            # Local neural translation models (free)
            if TRANSFORMERS_AVAILABLE:
                self._setup_local_models()

            logger.info(
                f"Initialized {len(self.translators)} FREE translation services"
            )

        except Exception as e:
            logger.error(f"Error initializing translators: {e}")

    def _setup_argos_translate(self) -> None:
        """Setup Argos Translate for offline translation."""
        try:
            # Install language packages if not already installed
            available_packages = argostranslate.package.get_available_packages()
            installed_packages = argostranslate.package.get_installed_packages()

            # Prefer English->Vietnamese package; other pairs may be installed manually
            en_vi_packages = [
                p
                for p in available_packages
                if p.from_code == "en" and p.to_code == "vi"
            ]

            if en_vi_packages and not any(
                p.from_code == "en" and p.to_code == "vi" for p in installed_packages
            ):
                logger.info("Installing English to Vietnamese translation package...")
                argostranslate.package.install_from_path(en_vi_packages[0].download())

            self.translators["argos_offline"] = "argos"
            logger.info("Argos Translate (offline) initialized")

        except Exception as e:
            logger.warning(f"Failed to setup Argos Translate: {e}")

    def _setup_local_models(self) -> None:
        """Setup local neural translation models using Transformers.

        Now using NLLB-200-distilled-600M for better support of long texts.
        Max token limit: 1024 (double of opus-mt's 512).
        Supports direct JA→VI and EN→VI translation.
        """
        try:
            # Use NLLB-200-distilled-600M for high-quality offline translation
            # This model supports 200+ languages including JA→VI directly
            model_name = "facebook/nllb-200-distilled-600M"

            try:
                logger.info(f"Loading NLLB translation model: {model_name}")
                logger.info("This may take a moment on first load...")

                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

                # Store tokenizer and model for direct access
                self.nllb_tokenizer = tokenizer
                self.nllb_model = model

                # NLLB language codes mapping
                self.nllb_lang_codes = {
                    "en": "eng_Latn",
                    "vi": "vie_Latn",
                    "ja": "jpn_Jpan",
                    "zh": "zho_Hans",
                    "ko": "kor_Hang",
                    "th": "tha_Thai",
                    "id": "ind_Latn",
                }

                # Register NLLB for multiple language pairs
                for src in ["en", "ja", "zh", "ko"]:
                    self.local_models[f"nllb_{src}_vi"] = {
                        "tokenizer": tokenizer,
                        "model": model,
                        "src_lang": src,
                        "tgt_lang": "vi",
                        "max_length": 1024,  # NLLB max token limit
                    }
                    self.translators[f"nllb_{src}_vi"] = "nllb"

                logger.info(f"NLLB model loaded successfully (max_length: 1024 tokens)")
                logger.info(f"Supported translations: EN→VI, JA→VI, ZH→VI, KO→VI")

            except Exception as e:
                logger.warning(f"Failed to load NLLB model: {e}")
                logger.info("Falling back to lighter models...")

                # Fallback to opus-mt models if NLLB fails
                fallback_configs = [
                    ("en", "vi", "Helsinki-NLP/opus-mt-en-vi"),
                    ("ja", "en", "Helsinki-NLP/opus-mt-ja-en"),
                ]

                for src_lang, tgt_lang, fallback_model in fallback_configs:
                    try:
                        logger.info(f"Loading fallback model: {fallback_model}")
                        tokenizer = MarianTokenizer.from_pretrained(fallback_model)
                        model = MarianMTModel.from_pretrained(fallback_model)

                        self.local_models[f"local_{src_lang}_{tgt_lang}"] = {
                            "tokenizer": tokenizer,
                            "model": model,
                            "pipeline": pipeline(
                                "translation", model=model, tokenizer=tokenizer
                            ),
                            "max_length": 512,
                        }

                        self.translators[f"local_{src_lang}_{tgt_lang}"] = "local"
                        logger.info(f"Fallback model {fallback_model} loaded")

                    except Exception as e2:
                        logger.warning(
                            f"Failed to load fallback model {fallback_model}: {e2}"
                        )

        except Exception as e:
            logger.warning(f"Failed to setup local models: {e}")

    def start_async_processing(self) -> None:
        """Start asynchronous translation processing."""
        if not self.is_processing:
            self.is_processing = True
            # Start multiple worker threads to increase throughput
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
        # Join worker threads
        for t in getattr(self, "worker_threads", []):
            try:
                if t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                pass

        self.worker_threads = []
        logger.info("Async translation processing stopped")

    def translate_batch(
        self, texts: List[str], source_language: str = "auto", use_context: bool = True
    ) -> List[Optional[TranslationResult]]:
        """
        Translate multiple texts in batch for better performance.

        Args:
            texts: List of texts to translate
            source_language: Source language code
            use_context: Use surrounding context for better translation accuracy

        Returns:
            List of translation results
        """
        if not texts:
            return []

        results = [None] * len(texts)
        texts_to_translate = {}

        # Check cache and filter
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue

            cleaned_text = self._clean_text(text)

            # Skip if already target language
            if self._is_target_language(cleaned_text):
                results[i] = TranslationResult(
                    original_text=cleaned_text,
                    translated_text=cleaned_text,
                    source_language=self.target_language,
                    target_language=self.target_language,
                    confidence=1.0,
                    timestamp=datetime.now(),
                    service_used="no_translation",
                )
                continue

            # Check cache
            cached_result = self.cache.get(
                cleaned_text, source_language, self.target_language
            )
            if cached_result:
                results[i] = cached_result
            else:
                texts_to_translate[i] = cleaned_text

        # Batch translate uncached texts
        if texts_to_translate:
            batch_texts = list(texts_to_translate.values())
            batch_indices = list(texts_to_translate.keys())

            # Use NLLB for batch translation if available
            translated = self._batch_translate_with_nllb(batch_texts, source_language)

            for idx, trans_text in zip(batch_indices, translated):
                if trans_text:
                    result = TranslationResult(
                        original_text=texts_to_translate[idx],
                        translated_text=trans_text,
                        source_language=source_language,
                        target_language=self.target_language,
                        confidence=0.9,
                        timestamp=datetime.now(),
                        service_used=f"nllb_{source_language}_vi_batch",
                    )
                    self.cache.set(result)
                    results[idx] = result

        return results

    def _batch_translate_with_nllb(
        self, texts: List[str], source_lang: str = "en", batch_size: int = 8
    ) -> List[str]:
        """
        Batch translate using NLLB model with automatic chunking for long texts.
        Optimized for video dài với batch size nhỏ để xử lý ổn định.

        Args:
            texts: List of texts to translate
            source_lang: Source language code (en, ja, zh, ko, etc.)
            batch_size: Number of texts per batch (default 8 for stability)

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        # Check if NLLB is available
        if not hasattr(self, "nllb_model") or not hasattr(self, "nllb_tokenizer"):
            # Fallback to opus-mt if available
            model_info = self.local_models.get("local_en_vi")
            if model_info and "pipeline" in model_info:
                results = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    try:
                        with self._model_lock:
                            batch_results = model_info["pipeline"](
                                batch, batch_size=len(batch)
                            )
                        results.extend([r["translation_text"] for r in batch_results])
                    except Exception as e:
                        logger.warning(f"Opus-MT batch failed: {e}")
                        results.extend(batch)
                return results

            # Final fallback to Argos
            return [self._translate_with_argos(text, source_lang) for text in texts]

        # Use NLLB for batch translation
        results = []
        src_code = self.nllb_lang_codes.get(source_lang, "eng_Latn")
        tgt_code = self.nllb_lang_codes.get("vi", "vie_Latn")

        # Process in small batches for stability with long videos
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                # Set source language
                self.nllb_tokenizer.src_lang = src_code

                # Tokenize batch OUTSIDE lock
                inputs = self.nllb_tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,  # NLLB supports longer texts
                )

                # Get target language token ID
                forced_bos_token_id = self.nllb_tokenizer.convert_tokens_to_ids(
                    tgt_code
                )

                # Lock ONLY during model inference
                with self._model_lock:
                    translated_tokens = self.nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=1024,
                        num_beams=4,  # Balance quality vs speed
                        early_stopping=True,
                    )

                # Decode OUTSIDE lock
                batch_translations = self.nllb_tokenizer.batch_decode(
                    translated_tokens, skip_special_tokens=True
                )

                results.extend(batch_translations)

            except Exception as e:
                logger.warning(f"NLLB batch translation failed: {e}")
                # Fallback: translate individually with chunking
                for text in batch:
                    trans = self._translate_with_nllb(text, source_lang, "vi")
                    results.append(trans if trans else text)

        return results

    def _translate_with_nllb(
        self, text: str, src_lang: str, tgt_lang: str = "vi"
    ) -> Optional[str]:
        """Dịch văn bản sử dụng NLLB model với xử lý tự động chia chunk.

        Args:
            text: Văn bản cần dịch
            src_lang: Mã ngôn ngữ nguồn (en, ja, zh, ko, ...)
            tgt_lang: Mã ngôn ngữ đích (mặc định vi)

        Returns:
            Văn bản đã dịch hoặc None nếu thất bại
        """
        if not hasattr(self, "nllb_model") or not hasattr(self, "nllb_tokenizer"):
            return None

        try:
            # Lấy NLLB language codes
            src_code = self.nllb_lang_codes.get(src_lang, "eng_Latn")
            tgt_code = self.nllb_lang_codes.get(tgt_lang, "vie_Latn")

            # Chia văn bản thành chunks nếu quá dài (an toàn với limit 1024)
            chunks = self._split_text_into_chunks(text, max_tokens=900)

            if len(chunks) > 1:
                logger.info(f"Text split into {len(chunks)} chunks for translation")

            translated_chunks = []

            for i, chunk in enumerate(chunks):
                try:
                    # Set source language
                    self.nllb_tokenizer.src_lang = src_code

                    # Prepare inputs OUTSIDE lock
                    inputs = self.nllb_tokenizer(
                        chunk,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=1024,
                    )

                    # Get target language token ID
                    forced_bos_token_id = self.nllb_tokenizer.convert_tokens_to_ids(
                        tgt_code
                    )

                    # Lock ONLY during model inference (critical section)
                    with self._model_lock:
                        translated_tokens = self.nllb_model.generate(
                            **inputs,
                            forced_bos_token_id=forced_bos_token_id,
                            max_length=1024,
                        )

                    # Decode OUTSIDE lock
                    translated_text = self.nllb_tokenizer.batch_decode(
                        translated_tokens, skip_special_tokens=True
                    )[0]

                    translated_chunks.append(translated_text)

                    if len(chunks) > 1:
                        logger.debug(f"Chunk {i+1}/{len(chunks)} translated")

                except Exception as e:
                    logger.warning(f"Failed to translate chunk {i+1}: {e}")
                    # Fallback: giữ nguyên chunk gốc
                    translated_chunks.append(chunk)

            # Ghép các chunks lại
            final_translation = " ".join(translated_chunks)

            # Clean up overlapping parts nếu có
            final_translation = re.sub(r"\s+", " ", final_translation).strip()

            return final_translation

        except Exception as e:
            logger.error(f"NLLB translation failed: {e}")
            return None

    def _translate_with_argos(self, text: str, source_lang: str) -> str:
        """Translate using Argos without full result object."""
        if not ARGOS_AVAILABLE:
            return text
        try:
            with self._model_lock:
                return argostranslate.translate.translate(
                    text, source_lang, self.target_language
                )
        except Exception as e:
            logger.warning(f"Argos translation failed: {e}")
            return text

    def translate_text(
        self, text: str, source_language: str = "auto"
    ) -> Optional[TranslationResult]:
        """
        Translate text to the target language.

        Args:
            text: Text to translate
            source_language: Source language code (auto-detect if 'auto')

        Returns:
            Translation result or None if failed
        """
        if not text or not text.strip():
            return None

        # Clean and normalize text
        cleaned_text = self._clean_text(text)

        # Check cache first
        cached_result = self.cache.get(
            cleaned_text, source_language, self.target_language
        )
        if cached_result:
            logger.debug(f"Using cached translation: {cleaned_text[:30]}...")
            return cached_result

        # Skip translation if text is already in target language
        if self._is_target_language(cleaned_text):
            result = TranslationResult(
                original_text=cleaned_text,
                translated_text=cleaned_text,
                source_language=self.target_language,
                target_language=self.target_language,
                confidence=1.0,
                timestamp=datetime.now(),
                service_used="no_translation",
            )
            self.cache.set(result)
            return result

        # Japanese → Vietnamese: prefer a two-step offline pipeline JA→EN→VI
        # if both local models are available. This is triggered when the
        # caller explicitly sets source_language="ja" (e.g. via --jp flag).
        if source_language == "ja":
            result = self._translate_ja_to_vi_offline(cleaned_text)
            if result:
                self.cache.set(result)
                logger.info(
                    f"JA→VI offline: '{cleaned_text[:50]}...' -> '{result.translated_text[:50]}...'"
                )
                return result

        # General case: try translation services in order of preference (OFFLINE only)
        # Priority: NLLB (best quality) → opus-mt → Argos (fallback)
        result = None

        # 1. Try NLLB first (highest quality, supports long texts)
        if hasattr(self, "nllb_model"):
            nllb_key = f"nllb_{source_language}_vi"
            if nllb_key in self.translators:
                try:
                    translated_text = self._translate_with_nllb(
                        cleaned_text, source_language, self.target_language
                    )
                    if translated_text:
                        result = TranslationResult(
                            original_text=cleaned_text,
                            translated_text=translated_text,
                            source_language=source_language,
                            target_language=self.target_language,
                            confidence=0.95,  # NLLB high confidence
                            timestamp=datetime.now(),
                            service_used=nllb_key,
                        )
                        logger.debug(f"Translated with NLLB: {source_language}→vi")
                except Exception as e:
                    logger.warning(f"NLLB translation failed, trying fallback: {e}")

        # 2. Fallback to opus-mt or Argos if NLLB fails
        if not result:
            service_priority = [
                "local_en_vi",  # opus-mt fallback
                "argos_offline",  # Argos fallback
            ]

            for service_name in service_priority:
                if service_name in self.translators:
                    result = self._translate_with_service(
                        service_name, cleaned_text, source_language
                    )
                    if result:
                        logger.debug(f"Used fallback service: {service_name}")
                        break

        # Cache successful translation
        if result:
            self.cache.set(result)
            logger.info(
                f"Translated: '{cleaned_text[:50]}...' -> '{result.translated_text[:50]}...'"
            )
        else:
            logger.warning(f"Failed to translate: {cleaned_text[:50]}...")

        return result

    def _translate_ja_to_vi_offline(self, text: str) -> Optional[TranslationResult]:
        """Translate Japanese to Vietnamese using offline models only.

        Priority:
        1) NLLB direct JA→VI (best quality, supports long texts)
        2) Two-step JA→EN→VI with opus-mt models
        3) Argos Translate JA→VI (last resort)
        """
        try:
            # 1. Try NLLB direct JA→VI (most accurate)
            if hasattr(self, "nllb_model"):
                try:
                    logger.debug("Trying NLLB direct JA→VI translation")
                    vi_text = self._translate_with_nllb(text, "ja", "vi")

                    if vi_text:
                        return TranslationResult(
                            original_text=text,
                            translated_text=vi_text,
                            source_language="ja",
                            target_language="vi",
                            confidence=0.95,  # NLLB highest confidence
                            timestamp=datetime.now(),
                            service_used="nllb_ja_vi_direct",
                        )
                except Exception as e:
                    logger.warning(f"NLLB JA→VI failed, trying fallback: {e}")

            # 2. Fallback: Two-step JA→EN→VI with opus-mt models
            ja_en_key = "local_ja_en"
            en_vi_key = "local_en_vi"

            ja_en = self.local_models.get(ja_en_key)
            en_vi = self.local_models.get(en_vi_key)

            if ja_en and en_vi:
                logger.debug("Using fallback JA→EN→VI pipeline")

                ja_en_pipe = ja_en["pipeline"]
                en_vi_pipe = en_vi["pipeline"]

                # Step 1: JA→EN
                with self._model_lock:
                    en_res = ja_en_pipe(text)
                en_text = en_res[0]["translation_text"] if en_res else text

                # Step 2: EN→VI
                with self._model_lock:
                    vi_res = en_vi_pipe(en_text)
                vi_text = vi_res[0]["translation_text"] if vi_res else en_text

                return TranslationResult(
                    original_text=text,
                    translated_text=vi_text,
                    source_language="ja",
                    target_language="vi",
                    confidence=0.85,
                    timestamp=datetime.now(),
                    service_used="opus_ja_en+en_vi",
                )

            # 3. Last resort: Argos Translate JA→VI
            if ARGOS_AVAILABLE:
                try:
                    logger.debug("Last resort: Argos JA→VI")
                    with self._model_lock:
                        vi_text = argostranslate.translate.translate(text, "ja", "vi")
                    if vi_text and vi_text.strip():
                        return TranslationResult(
                            original_text=text,
                            translated_text=vi_text,
                            source_language="ja",
                            target_language="vi",
                            confidence=0.7,
                            timestamp=datetime.now(),
                            service_used="argos_ja_vi",
                        )
                except Exception as e:
                    logger.warning(f"Argos JA→VI translation failed: {e}")

        except Exception as e:
            logger.error(f"Error in offline JA→VI pipeline: {e}")

        return None

    def _translate_with_service(
        self, service_name: str, text: str, source_language: str
    ) -> Optional[TranslationResult]:
        """
        Translate text using a specific service.

        Args:
            service_name: Name of the translation service to use
            text: Text to translate
            source_language: Source language code

        Returns:
            Translation result or None if failed
        """
        if service_name not in self.translators:
            return None

        translator = self.translators[service_name]

        for attempt in range(self.retry_attempts):
            try:
                if translator == "argos":
                    # Using Argos Translate (completely offline)
                    # Lock to ensure thread-safety
                    with self._model_lock:
                        src = source_language or "en"
                        tgt = self.target_language
                        translated_text = argostranslate.translate.translate(
                            text, src, tgt
                        )

                    return TranslationResult(
                        original_text=text,
                        translated_text=translated_text,
                        source_language=src,
                        target_language=self.target_language,
                        confidence=0.8,
                        timestamp=datetime.now(),
                        service_used=service_name,
                    )

                elif service_name.startswith("local_"):
                    # Using local neural models
                    # Minimize lock scope for better parallelism
                    model_info = self.local_models.get(service_name)
                    if model_info:
                        pipeline = model_info["pipeline"]
                        # Lock only during inference
                        with self._model_lock:
                            result = pipeline(text)

                        return TranslationResult(
                            original_text=text,
                            translated_text=result[0]["translation_text"],
                            source_language=source_language,
                            target_language=self.target_language,
                            confidence=0.9,  # High confidence for local models
                            timestamp=datetime.now(),
                            service_used=service_name,
                        )

            except Exception as e:
                logger.warning(
                    f"Translation attempt {attempt + 1} failed with {service_name}: {e}"
                )
                continue

        return None

    def translate_async(
        self,
        text: str,
        source_language: str = "auto",
        callback: Optional[callable] = None,
    ) -> None:
        """
        Queue text for asynchronous translation.

        Args:
            text: Text to translate
            source_language: Source language code
            callback: Optional callback function for result
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

                # Translate text
                result = self.translate_text(text, source_lang)

                # Call callback if provided
                if callback and result:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Error in translation callback: {e}")

                # Call global callback
                if self.on_translation_complete and result:
                    try:
                        self.on_translation_complete(result)
                    except Exception as e:
                        logger.error(f"Error in global translation callback: {e}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing translation queue: {e}")

    def _split_text_into_chunks(
        self, text: str, max_tokens: int = 900, overlap: int = 50
    ) -> List[str]:
        """Tự động chia văn bản dài thành các chunks nhỏ hơn để tránh vượt quá giới hạn token.

        Args:
            text: Văn bản cần chia
            max_tokens: Số token tối đa mỗi chunk (mặc định 900 để an toàn với limit 1024)
            overlap: Số từ overlap giữa các chunk để giữ context

        Returns:
            Danh sách các text chunks
        """
        if not text or len(text) < 100:
            return [text]

        # Ước tính token count (1 token ≈ 4 ký tự cho tiếng Anh, 2-3 cho tiếng Việt)
        estimated_tokens = len(text) // 3

        if estimated_tokens <= max_tokens:
            return [text]

        # Chia theo câu để giữ ngữ nghĩa
        sentences = re.split(r"([.!?]\s+)", text)

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = len(sentence) // 3

            # Nếu thêm câu này vào sẽ vượt quá limit
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())

                # Bắt đầu chunk mới với overlap (lấy vài câu cuối của chunk trước)
                overlap_text = " ".join(current_chunk.split()[-overlap:])
                current_chunk = overlap_text + " " + sentence
                current_tokens = len(current_chunk) // 3
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens

        # Thêm chunk cuối
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        logger.debug(
            f"Split text into {len(chunks)} chunks (estimated {estimated_tokens} tokens)"
        )
        return chunks if chunks else [text]

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for translation.
        Optimized for subtitle text processing.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text
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

        # Remove trailing/leading punctuation that's likely noise
        text = text.strip(" .,;:")

        return text.strip()

    def detect_repetition(self, text: str, threshold: float = 0.8) -> Tuple[bool, str]:
        """Detect and remove repetitive patterns (Whisper hallucination)."""
        if not text or len(text) < 20:
            return False, text

        words = text.split()
        if len(words) < 4:
            return False, text

        seen_phrases = {}
        phrase_length = min(5, len(words) // 2)

        for i in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[i : i + phrase_length])
            if phrase in seen_phrases:
                seen_phrases[phrase] += 1
            else:
                seen_phrases[phrase] = 1

        max_repetitions = max(seen_phrases.values()) if seen_phrases else 1
        total_possible = len(words) - phrase_length + 1

        if max_repetitions > 2 and max_repetitions / total_possible > 0.5:
            repeated_phrase = max(seen_phrases, key=seen_phrases.get)
            logger.warning(
                f"Detected repetition pattern: '{repeated_phrase}' x{max_repetitions}"
            )

            cleaned_words = []
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

            cleaned_text = " ".join(cleaned_words)
            return True, cleaned_text

        return False, text

    def detect_segment_repetition(
        self, segments: List[dict], similarity_threshold: float = 0.85
    ) -> List[dict]:
        """Filter consecutive duplicate segments (Whisper hallucination)."""
        if not segments or len(segments) < 2:
            return segments

        cleaned_segments = []
        prev_text = ""
        repetition_count = 0
        max_allowed_repetitions = 2

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

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts (0.0 - 1.0)."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _is_target_language(self, text: str) -> bool:
        """
        Check if text is likely already in the target language.

        Args:
            text: Text to check

        Returns:
            True if text is likely in target language
        """
        # Simple heuristic for Vietnamese text detection
        if self.target_language == "vi":
            vietnamese_chars = set(
                "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
            )
            text_chars = set(text.lower())

            # If text contains Vietnamese characters, likely Vietnamese
            if text_chars & vietnamese_chars:
                return True

        return False

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get supported languages for translation.

        Returns:
            Dictionary of language codes and names
        """
        # Common language codes supported by free services
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
        vietnamese_chars = set(
            "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        )
        return bool(set(text.lower()) & vietnamese_chars)

    def set_callback(self, callback: callable) -> None:
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
        self.cache.cache.clear()
        self.cache.access_times.clear()
        logger.info("Translation cache cleared")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_async_processing()
        self.clear_cache()
        logger.info("TranslationService cleanup completed")
