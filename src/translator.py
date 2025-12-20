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
    from transformers import MarianMTModel, MarianTokenizer, pipeline

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

        We load separate models for different language directions so that
        Japanese → Vietnamese can use a two-step offline pipeline
        (JA→EN then EN→VI) when available.
        """
        try:
            # Use free Helsinki-NLP models for translation
            model_configs = [
                ("en", "vi", "Helsinki-NLP/opus-mt-en-vi"),  # EN → VI
                ("ja", "en", "Helsinki-NLP/opus-mt-ja-en"),  # JA → EN
            ]

            for src_lang, tgt_lang, model_name in model_configs:
                try:
                    logger.info(f"Loading local translation model: {model_name}")
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)

                    self.local_models[f"local_{src_lang}_{tgt_lang}"] = {
                        "tokenizer": tokenizer,
                        "model": model,
                        "pipeline": pipeline(
                            "translation", model=model, tokenizer=tokenizer
                        ),
                    }

                    self.translators[f"local_{src_lang}_{tgt_lang}"] = "local"
                    logger.info(f"Local model {model_name} loaded successfully")

                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")

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
            if source_language == "ja":
                # JA->VI pipeline
                for i, text in texts_to_translate.items():
                    result = self._translate_ja_to_vi_offline(text)
                    if result:
                        self.cache.set(result)
                        results[i] = result
            else:
                # EN->VI batch
                batch_texts = list(texts_to_translate.values())
                batch_indices = list(texts_to_translate.keys())

                translated = self._batch_translate_en_vi(batch_texts)
                for idx, trans_text in zip(batch_indices, translated):
                    if trans_text:
                        result = TranslationResult(
                            original_text=texts_to_translate[idx],
                            translated_text=trans_text,
                            source_language=source_language,
                            target_language=self.target_language,
                            confidence=0.9,
                            timestamp=datetime.now(),
                            service_used="local_en_vi_batch",
                        )
                        self.cache.set(result)
                        results[idx] = result

        return results

    def _batch_translate_en_vi(
        self, texts: List[str], batch_size: int = 8
    ) -> List[str]:
        """
        Batch translate EN->VI using local model for better throughput.

        Args:
            texts: List of texts to translate
            batch_size: Number of texts per batch

        Returns:
            List of translated texts
        """
        if not texts:
            return []

        model_info = self.local_models.get("local_en_vi")
        if not model_info:
            # Fallback to Argos
            return [self._translate_with_argos(text, "en") for text in texts]

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Lock only during inference
            with self._model_lock:
                try:
                    pipeline = model_info["pipeline"]
                    batch_results = pipeline(batch, batch_size=len(batch))
                    results.extend([r["translation_text"] for r in batch_results])
                except Exception as e:
                    logger.warning(f"Batch translation failed: {e}")
                    # Fallback to individual
                    results.extend(batch)

        return results

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
        service_priority = [
            "local_en_vi",  # Local neural model (fastest, no internet)
            "argos_offline",  # Offline rule-based / statistical
        ]

        result = None
        for service_name in service_priority:
            if service_name in self.translators:
                result = self._translate_with_service(
                    service_name, cleaned_text, source_language
                )
                if result:
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

    # NOTE: Previous versions implemented a special JA->VI pipeline that used
    # online services. We now provide an OFFLINE-ONLY JA→VI pipeline using
    # local Transformers models when available.

    def _translate_ja_to_vi_offline(self, text: str) -> Optional[TranslationResult]:
        """Translate Japanese to Vietnamese using offline models only.

        Pipeline:
        1) JA → EN using local_ja_en model (if available)
        2) EN → VI using local_en_vi model (if available)

        If one of the models is missing, this falls back to Argos Translate
        (JA→VI via its own installed packages) when possible.
        """
        try:
            ja_en_key = "local_ja_en"
            en_vi_key = "local_en_vi"

            ja_en = self.local_models.get(ja_en_key)
            en_vi = self.local_models.get(en_vi_key)

            # Two-step JA→EN→VI with local Transformers models
            if ja_en and en_vi:
                logger.debug("Using local JA→EN then EN→VI pipeline")

                # Reduce lock scope: only lock during actual inference
                ja_en_pipe = ja_en["pipeline"]
                en_vi_pipe = en_vi["pipeline"]

                # Step 1: JA→EN (lock only during inference)
                with self._model_lock:
                    en_res = ja_en_pipe(text)
                en_text = en_res[0]["translation_text"] if en_res else text

                # Step 2: EN→VI (lock only during inference)
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
                    service_used="local_ja_en+local_en_vi",
                )

            # Fallback: Argos Translate JA→VI directly if packages exist
            if ARGOS_AVAILABLE:
                try:
                    logger.debug("Falling back to Argos JA→VI if available")
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

        # Try to detect language with short text analysis
        try:
            if len(text) > 20:  # Only for longer texts
                translator = self.translators.get("google_main")
                if translator:
                    detection = translator.detect(text)
                    return detection.lang == self.target_language
        except:
            pass

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

    def detect_language(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Detect the language of the input text using free methods.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (language_code, confidence) or None
        """
        try:
            # Simple heuristic language detection (free, offline)
            if self._contains_vietnamese_chars(text):
                return ("vi", 0.8)
            if self._contains_english_patterns(text):
                return ("en", 0.7)
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

        # Default fallback: assume English with low confidence
        return ("en", 0.5)

    def _contains_vietnamese_chars(self, text: str) -> bool:
        """Check if text contains Vietnamese characters."""
        vietnamese_chars = set(
            "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        )
        return bool(set(text.lower()) & vietnamese_chars)

    def _contains_english_patterns(self, text: str) -> bool:
        """Check if text contains common English patterns."""
        english_words = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = text.lower().split()
        return any(word in english_words for word in words)

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


def test_translation_service():
    """Test function for the TranslationService class."""
    import yaml

    # Test configuration
    config = {
        "translation": {
            "target_language": "vi",
            "service": "google",
            "google": {"timeout": 10, "retry_attempts": 3},
        },
        "performance": {"max_cache_size": 50},
    }

    # Test texts
    test_texts = [
        "Hello, this is a test of the real-time translation system.",
        "How are you doing today?",
        "This is a longer sentence that should be translated into Vietnamese automatically.",
        "Thank you for using this application.",
        "Goodbye and have a nice day!",
    ]

    # Create translation service
    translator = TranslationService(config)

    def on_translation_done(result):
        print(f"Async translation completed: {result.translated_text}")

    translator.set_callback(on_translation_done)
    translator.start_async_processing()

    try:
        print("Testing synchronous translation:")
        for text in test_texts:
            result = translator.translate_text(text)
            if result:
                print(f"Original: {result.original_text}")
                print(f"Vietnamese: {result.translated_text}")
                print(f"Service: {result.service_used}")
                print("-" * 50)
            else:
                print(f"Failed to translate: {text}")

        print("\nTesting asynchronous translation:")
        for text in test_texts:
            translator.translate_async(text)

        # Wait for async translations
        time.sleep(5)

        # Print cache stats
        stats = translator.get_cache_stats()
        print(f"\nCache Stats: {stats}")

    except KeyboardInterrupt:
        pass
    finally:
        translator.cleanup()


if __name__ == "__main__":
    test_translation_service()
