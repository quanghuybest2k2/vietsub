#!/usr/bin/env python3
"""
Offline Vietnamese Subtitle Generator (100% Offline)

Video processing application that uses speech recognition and translation
to generate Vietnamese subtitles from video files - completely offline.

Author: Vietnamese Subtitle System
Date: 2025
"""
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import argparse
import gc
import signal
import time
import subprocess
import shutil
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml
import re
import unicodedata


from loguru import logger

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.translator import (
    TranslationService,
    TranslationResult,
    resolve_compute_candidates,
)

# Realtime-only dependencies (PyAudio, SciPy, OpenCV, keyboard) are imported
# lazily, so --export-srt / --video / --voiceover neither need them installed
# nor pay their import time.

PROJECT_DIR = Path(__file__).resolve().parent


def get_ffmpeg_exe() -> str:
    """Return an ffmpeg executable: PATH first, then the imageio-ffmpeg bundle.

    Returns:
        Path to an ffmpeg binary

    Raises:
        RuntimeError: If no ffmpeg binary is available
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg not found. Install FFmpeg and add it to PATH "
            "(or `pip install imageio-ffmpeg`)."
        ) from e


def probe_media(path: str) -> Dict[str, Any]:
    """Read basic stream facts with PyAV (no ffprobe needed).

    Args:
        path: Media file path

    Returns:
        {"duration": seconds, "has_audio": bool, "audio_codec": name or None,
         "audio_offset": seconds the first audio sample is played after the
         start of the file (Whisper timestamps must be shifted by it)}
    """
    import av

    with av.open(path, metadata_errors="ignore") as container:
        duration = container.duration / av.time_base if container.duration else 0.0
        if not duration:
            for stream in container.streams:
                if stream.duration and stream.time_base:
                    duration = max(duration, float(stream.duration * stream.time_base))
        info: Dict[str, Any] = {
            "duration": duration,
            "has_audio": len(container.streams.audio) > 0,
            "audio_codec": None,
            "audio_offset": 0.0,
            "video_codec": None,
            "display_size": None,  # (width, height) as shown, rotation applied
        }
        if info["has_audio"]:
            audio = container.streams.audio[0]
            info["audio_codec"] = audio.codec_context.name
            if audio.start_time is not None and audio.time_base:
                container_start = (container.start_time or 0) / av.time_base
                offset = float(audio.start_time * audio.time_base) - container_start
                info["audio_offset"] = offset if offset > 0.02 else 0.0
        if container.streams.video:
            video = container.streams.video[0]
            info["video_codec"] = video.codec_context.name
            width, height = video.width, video.height
            try:
                # Phone videos are often stored landscape with a 90° display
                # rotation; ffmpeg rotates them before drawing subtitles.
                frame = next(container.decode(video=0))
                if abs(int(getattr(frame, "rotation", 0) or 0)) % 180 == 90:
                    width, height = height, width
            except Exception:
                pass
            if width and height:
                info["display_size"] = (width, height)
        return info


def subtitle_style(display_size: Optional[Tuple[int, int]]) -> Optional[str]:
    """libass force_style for frames narrower than 16:9 (portrait/square).

    ffmpeg sizes SRT text relative to the frame HEIGHT, so on a 9:16 phone
    video a normal cue became ~5-7 huge lines covering a third of the frame.

    Args:
        display_size: (width, height) of the displayed video

    Returns:
        force_style value, or None to keep ffmpeg's default (16:9 and wider)
    """
    if not display_size or not display_size[1]:
        return None
    width, height = display_size
    scale = max(0.6, min(1.0, (width / height) / (16 / 9)))
    if scale >= 0.99:
        return None
    return f"Fontsize={round(16 * scale)}"


# Audio codecs that play everywhere inside MP4/MOV; anything else (camera PCM,
# Vorbis, ...) is re-encoded to AAC instead of being copied.
_MP4_SAFE_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}


def audio_copy_args(output_file: str, audio_codec: Optional[str]) -> List[str]:
    """ffmpeg audio arguments: copy when the codec suits the container, else AAC.

    Args:
        output_file: Destination path (its extension decides the container)
        audio_codec: Source audio codec name from probe_media()

    Returns:
        ffmpeg arguments for the audio stream
    """
    if Path(output_file).suffix.lower() in (".mp4", ".m4v", ".mov") and (
        audio_codec not in _MP4_SAFE_AUDIO
    ):
        return ["-c:a", "aac", "-b:a", "192k"]
    return ["-c:a", "copy"]


def _preload_torch_cuda() -> None:
    """Load PyTorch's CUDA libraries before CTranslate2 needs them.

    CTranslate2 dlopen()s cuBLAS lazily; when the only cuBLAS on the machine is
    the one pip-installed with PyTorch, importing torch first makes it visible.
    On Windows this also avoids a cuDNN DLL conflict. Harmless without torch.
    """
    try:
        import torch

        if torch.cuda.is_available():
            _ = torch.zeros(1, device="cuda")
            _ = torch.nn.functional.relu(torch.tensor(0.0, device="cuda"))
    except Exception as e:
        logger.debug(f"PyTorch CUDA preload skipped (non-fatal): {e}")


class VietnameseSubtitleGenerator:
    """Main application class for offline video subtitle generation (100% local processing)."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the subtitle generator application.

        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.whisper_model = None
        self.device = None
        self.compute_type = None
        self.forced_source_lang = None
        self.forced_model_size = None  # User-selected model size
        self.forced_device = None  # --device auto|cuda|cpu
        self.audio_processor = None
        self.subtitle_overlay = None
        self.translation_service = None

        # Application state
        self.is_running = False
        self.is_paused = False

        # TTS voice setting (default: female Vietnamese voice using VieNeu v3 Turbo)
        self.tts_voice = "Ngọc Lan"  # Female, soft/gentle (default v3 voice)
        # Available VieNeu v3 Turbo voices:
        # Female: Ngọc Lan (default), Ngọc Linh, Trúc Ly, Mỹ Duyên
        # Male: Xuân Vĩnh, Thái Sơn, Gia Bảo, Đức Trí, Trọng Hữu, Bình An

        # TTS temperature: controls creativity/randomness in voice generation
        # Range: 0.0-1.5. Higher = more expressive but may have artifacts.
        self.tts_temperature = self.config.get("tts", {}).get("temperature", 1.0)

        # Statistics
        self.stats = {
            "transcriptions": 0,
            "translations": 0,
            "errors": 0,
            "start_time": None,
        }

        # Setup logging
        self._setup_logging()

        logger.info("Vietnamese Subtitle Generator initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file (relative paths also
                resolve against the project directory)

        Returns:
            Configuration dictionary
        """
        path = Path(config_path)
        if not path.exists() and not path.is_absolute():
            path = PROJECT_DIR / config_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            sys.exit(1)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})

        # Configure loguru
        logger.remove()  # Remove default handler

        # Console logging
        if log_config.get("console_output", True):
            logger.add(
                sys.stdout,
                level=log_config.get("level", "INFO"),
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            )

        # File logging
        if log_config.get("file_output", True):
            logger.add(
                "logs/subtitle_generator.log",
                level=log_config.get("level", "INFO"),
                rotation=log_config.get("max_file_size", "10 MB"),
                retention=log_config.get("backup_count", 3),
                compression="zip",
            )

    # ------------------------------------------------------------------ models

    def load_whisper_model(self) -> bool:
        """Load faster-whisper on the best working device, falling back to CPU.

        GPU compute types are picked from what the GPU actually supports
        (float16 on Volta+, int8 on Pascal, ...). CUDA libraries are only
        loaded on first use, so a short warm-up decode validates the GPU and
        any failure falls back to the next candidate instead of aborting.

        Returns:
            True if a model is ready
        """
        whisper_config = self.config.get("whisper", {})
        # Use forced model size if set via --model flag, otherwise use config
        model_size = self.forced_model_size or whisper_config.get("model_size", "base")
        device_pref = self.forced_device or whisper_config.get("device", "auto")
        compute_pref = whisper_config.get("compute_type", "auto")
        cpu_threads = int(whisper_config.get("cpu_threads", 0) or 0)

        try:
            import numpy as np
            from faster_whisper import WhisperModel
            from faster_whisper.utils import download_model
        except ImportError as e:
            logger.error(f"faster-whisper is not installed: {e}")
            return False

        logger.info("Loading faster-whisper model...")
        candidates = resolve_compute_candidates(
            device_pref, compute_pref, ["float16", "int8_float16", "int8", "float32"]
        )
        if any(device == "cuda" for device, _ in candidates):
            _preload_torch_cuda()

        # Resolve from the local cache first: no Hugging Face round-trip (or
        # long timeout when offline) on every start.
        try:
            model_path = model_size
            if not os.path.isdir(model_size):
                try:
                    model_path = download_model(model_size, local_files_only=True)
                except Exception:
                    model_path = None
                # An interrupted first download leaves a snapshot without
                # model.bin: resume the download instead of failing forever.
                if not model_path or not os.path.isfile(os.path.join(model_path, "model.bin")):
                    logger.info(f"Downloading faster-whisper model '{model_size}' (first run only)...")
                    model_path = download_model(model_size)
        except Exception as e:
            logger.error(f"Could not get faster-whisper model '{model_size}': {e}")
            return False

        cuda_broken = False
        cuda_oom = False
        for device, compute_type in candidates:
            if device == "cuda" and cuda_broken:
                continue
            if device == "cuda" and cuda_oom and compute_type == "float32":
                continue  # needs more VRAM than the type that just ran out of memory
            model = None
            try:
                model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=cpu_threads,
                )
                if device == "cuda":
                    segments, _ = model.transcribe(
                        np.zeros(16000, dtype=np.float32),
                        language="en",
                        beam_size=1,
                        vad_filter=False,
                        without_timestamps=True,
                        condition_on_previous_text=False,
                    )
                    list(segments)
                self.whisper_model = model
                self.device = device
                self.compute_type = compute_type
                logger.info(
                    f"faster-whisper model '{model_size}' loaded on {device} ({compute_type})"
                )
                return True
            except Exception as e:
                logger.warning(f"faster-whisper on {device}/{compute_type} failed: {e}")
                if device == "cuda":
                    if "out of memory" in str(e).lower():
                        cuda_oom = True
                    else:
                        cuda_broken = True
                model = None  # free the failed model before loading the next one
                gc.collect()

        logger.error("Could not load faster-whisper on any device")
        return False

    def get_translation_service(self) -> TranslationService:
        """Create the translation service on first use.

        File modes load it only after transcription, when Whisper has been
        released, so both models never compete for RAM/VRAM.

        Returns:
            The translation service
        """
        if self.translation_service is None:
            logger.info("Initializing translation service...")
            self.translation_service = TranslationService(self.config)
        return self.translation_service

    def _release_whisper(self) -> None:
        """Free the Whisper model (RAM/VRAM) once transcription is done."""
        if self.whisper_model is not None:
            self.whisper_model = None
            gc.collect()

    def _release_models(self) -> None:
        """Free Whisper and translation models before loading the TTS engine."""
        self._release_whisper()
        if self.translation_service is not None:
            self.translation_service.release_models()
            self.translation_service = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def initialize_components(self) -> bool:
        """
        Initialize all application components (real-time microphone mode).

        Returns:
            True if all components initialized successfully
        """
        try:
            # Initialize faster-whisper model (speech-to-text)
            if not self.load_whisper_model():
                return False

            # Initialize audio processor
            logger.info("Initializing audio processor...")
            try:
                from src.audio_processor import AudioProcessor

                self.audio_processor = AudioProcessor(self.config)
                self.audio_processor.set_callbacks(on_audio_chunk=self._on_audio_chunk)
            except Exception as e:
                logger.warning(
                    f"AudioProcessor initialization failed: {e}. Continuing without audio processor."
                )
                self.audio_processor = None

            # Initialize translation service
            self.get_translation_service()
            self.translation_service.set_callback(self._on_translation_complete)
            self.translation_service.start_async_processing()

            # Initialize subtitle overlay
            logger.info("Initializing subtitle overlay...")
            from src.subtitle_overlay import SubtitleOverlay

            self.subtitle_overlay = SubtitleOverlay(self.config)

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    # --------------------------------------------------------------- realtime

    def setup_hotkeys(self) -> None:
        """Setup keyboard hotkeys for application control."""
        hotkey_config = self.config.get("hotkeys", {})

        try:
            import keyboard

            # Start/Stop hotkey
            start_stop_key = hotkey_config.get("start_stop", "ctrl+shift+s")
            keyboard.add_hotkey(start_stop_key, self._toggle_recording)

            # Pause/Resume hotkey
            pause_resume_key = hotkey_config.get("pause_resume", "ctrl+shift+p")
            keyboard.add_hotkey(pause_resume_key, self._toggle_pause)

            # Toggle display hotkey
            toggle_display_key = hotkey_config.get("toggle_display", "ctrl+shift+d")
            keyboard.add_hotkey(toggle_display_key, self._toggle_display)

            # Quit hotkey
            quit_key = hotkey_config.get("quit", "ctrl+shift+q")
            keyboard.add_hotkey(quit_key, self._quit_application)

            logger.info("Hotkeys registered successfully")

        except Exception as e:
            logger.warning(f"Failed to setup hotkeys: {e}")

    def start(self) -> None:
        """Start the subtitle generation system."""
        if not self.initialize_components():
            logger.error("Failed to initialize components")
            return

        # Setup hotkeys
        self.setup_hotkeys()

        # Start subtitle overlay
        self.subtitle_overlay.start_overlay_display()

        # Start audio recording
        if self.audio_processor:
            if not self.audio_processor.start_recording():
                logger.error("Failed to start audio recording")
                return
        else:
            logger.warning(
                "Audio processor not available, continuing without audio recording"
            )

        self.is_running = True
        self.stats["start_time"] = time.time()

        logger.info("Vietnamese Subtitle Generator started")
        logger.info("Hotkeys:")
        hotkeys = self.config.get("hotkeys", {})
        logger.info(f"  Start/Stop: {hotkeys.get('start_stop', 'Ctrl+Shift+S')}")
        logger.info(f"  Pause/Resume: {hotkeys.get('pause_resume', 'Ctrl+Shift+P')}")
        logger.info(
            f"  Toggle Display: {hotkeys.get('toggle_display', 'Ctrl+Shift+D')}"
        )
        logger.info(f"  Quit: {hotkeys.get('quit', 'Ctrl+Shift+Q')}")

        # Main loop
        self._main_loop()

    def _main_loop(self) -> None:
        """Main application loop."""
        try:
            while self.is_running:
                # Print statistics periodically
                if self.config.get("debug", {}).get("show_processing_time", False):
                    self._print_stats()

                time.sleep(5)  # Update every 5 seconds

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def _on_audio_chunk(self, audio_file_path: str) -> None:
        """
        Callback for processing audio chunks from the audio processor.

        Args:
            audio_file_path: Path to temporary audio file
        """
        if self.is_paused:
            return

        try:
            whisper_config = self.config.get("whisper", {})
            # Language detection on a single 2-5 s chunk is unreliable: once a
            # chunk was detected with high confidence, keep that language.
            language = (
                self.forced_source_lang
                or getattr(self, "_session_language", None)
                or whisper_config.get("language", "auto")
            )

            # Transcribe audio using Whisper (faster-whisper API: no fp16 flag,
            # language=None means auto-detect)
            start_time = time.time()
            segments, info = self.whisper_model.transcribe(
                audio_file_path,
                language=None if language in (None, "", "auto") else language,
                temperature=whisper_config.get("temperature", 0.0),
                best_of=whisper_config.get("best_of", 1),
                beam_size=whisper_config.get("realtime_beam_size", 1),
                condition_on_previous_text=False,
                vad_filter=True,
            )

            # Extract text from segments
            text = "".join(segment.text for segment in segments).strip()
            transcription_time = time.time() - start_time
            detected_language = info.language

            if (
                text
                and not self.forced_source_lang
                and getattr(self, "_session_language", None) is None
                and language in (None, "", "auto")
                and (info.language_probability or 0.0) >= 0.8
            ):
                self._session_language = detected_language
                logger.info(
                    f"Source language locked to '{detected_language}' "
                    f"(p={info.language_probability:.2f}); use --language to override"
                )

            if text:
                logger.info(f"Transcribed ({detected_language}): {text}")
                self.stats["transcriptions"] += 1

                if self.config.get("debug", {}).get("show_processing_time", False):
                    logger.debug(f"Transcription took {transcription_time:.2f} seconds")

                # Queue for translation; the result reaches the overlay through
                # the callback registered with set_callback() (exactly once).
                source_lang = self.forced_source_lang or detected_language
                self.translation_service.translate_async(text, source_language=source_lang)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            self.stats["errors"] += 1

    def _on_translation_complete(self, result: TranslationResult) -> None:
        """
        Callback for completed translations.

        Args:
            result: Translation result object
        """
        try:
            self.stats["translations"] += 1

            # Display subtitle
            self.subtitle_overlay.add_subtitle(result.translated_text)

            logger.info(f"Subtitle displayed: {result.translated_text}")

            # Log verbose translation info if enabled
            if self.config.get("debug", {}).get("verbose_translation", False):
                logger.debug(f"Translation details:")
                logger.debug(f"  Original: {result.original_text}")
                logger.debug(f"  Translated: {result.translated_text}")
                logger.debug(f"  Source lang: {result.source_language}")
                logger.debug(f"  Service: {result.service_used}")
                logger.debug(f"  Confidence: {result.confidence}")

        except Exception as e:
            logger.error(f"Error displaying subtitle: {e}")
            self.stats["errors"] += 1

    def _toggle_recording(self) -> None:
        """Toggle audio recording on/off."""
        if not self.audio_processor:
            logger.warning("Audio processor not available")
            return

        if self.is_running:
            if self.audio_processor.is_recording:
                self.audio_processor.stop_recording()
                logger.info("Recording stopped")
            else:
                if self.audio_processor.start_recording():
                    logger.info("Recording started")
                else:
                    logger.error("Failed to start recording")

    def _toggle_pause(self) -> None:
        """Toggle pause/resume processing."""
        if not self.audio_processor:
            logger.warning("Audio processor not available")
            return

        self.is_paused = not self.is_paused

        if self.is_paused:
            self.audio_processor.pause_recording()
            logger.info("Processing paused")
        else:
            self.audio_processor.resume_recording()
            logger.info("Processing resumed")

    def _toggle_display(self) -> None:
        """Toggle subtitle display visibility."""
        # This could control overlay window visibility
        # For now, just clear current subtitles
        self.subtitle_overlay.clear_subtitles()
        logger.info("Subtitles cleared")

    def _quit_application(self) -> None:
        """Quit the application."""
        logger.info("Quit requested via hotkey")
        self.is_running = False

    def _print_stats(self) -> None:
        """Print application statistics."""
        if not self.stats["start_time"]:
            return

        runtime = time.time() - self.stats["start_time"]
        hours, remainder = divmod(runtime, 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info("=== Statistics ===")
        logger.info(f"Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        logger.info(f"Transcriptions: {self.stats['transcriptions']}")
        logger.info(f"Translations: {self.stats['translations']}")
        logger.info(f"Errors: {self.stats['errors']}")
        if self.translation_service:
            cache_stats = self.translation_service.get_cache_stats()
            logger.info(f"Cache usage: {cache_stats['cache_usage_percent']}%")
        if self.subtitle_overlay:
            logger.info(
                f"Active subtitles: {self.subtitle_overlay.get_active_subtitle_count()}"
            )

    def stop(self) -> None:
        """Stop the subtitle generation system."""
        logger.info("Stopping Vietnamese Subtitle Generator...")

        self.is_running = False

        # Stop components
        if self.audio_processor:
            self.audio_processor.cleanup()

        if self.translation_service:
            self.translation_service.cleanup()

        if self.subtitle_overlay:
            self.subtitle_overlay.cleanup()

        # Print final statistics
        self._print_stats()

        logger.info("Vietnamese Subtitle Generator stopped")

    # ------------------------------------------------------------ file modes

    def _transcribe_file(
        self, input_file: str, beam_size: int, audio_offset: float = 0.0
    ) -> Dict[str, Any]:
        """Transcribe a media file with the anti-repetition settings from config.

        faster-whisper decodes the container directly (PyAV), so no temporary
        WAV, moviepy or ffmpeg binary is needed for this step.

        Args:
            input_file: Video/audio file
            beam_size: Decoding beam size
            audio_offset: Seconds the audio stream starts after the file start;
                added to every timestamp so cues line up with the video

        Returns:
            {"segments": [{"start", "end", "text"}], "language": code}
        """
        if self.whisper_model is None and not self.load_whisper_model():
            raise RuntimeError("faster-whisper model is not available")

        whisper_config = self.config.get("whisper", {})
        anti_rep_config = self.config.get("anti_repetition", {})

        # Use forced source language for transcription if set
        transcribe_lang = self.forced_source_lang or whisper_config.get("language", "auto")
        options = dict(
            language=None if transcribe_lang in (None, "", "auto") else transcribe_lang,
            beam_size=beam_size,
            best_of=whisper_config.get("best_of", 1),
            # A single temperature disables the fallback, which also makes
            # compression_ratio_threshold inert (see config.yaml).
            temperature=whisper_config.get("temperature", 0.0),
            condition_on_previous_text=anti_rep_config.get("condition_on_previous_text", False),
            compression_ratio_threshold=anti_rep_config.get("compression_ratio_threshold", 2.4),
            log_prob_threshold=-1.0,
            no_speech_threshold=anti_rep_config.get("no_speech_threshold", 0.5),
            repetition_penalty=anti_rep_config.get("repetition_penalty", 1.0),
            vad_filter=True,
            vad_parameters=whisper_config.get("vad_parameters") or None,
        )
        # Optional batched decoding (several VAD chunks per forward pass): large
        # speed-up on strong GPUs; no gain measured on CPU with base, and it
        # has no temperature fallback - so it is opt-in.
        batch_size = int(whisper_config.get("batch_size", 0) or 0)
        runner = self.whisper_model
        if batch_size > 0:
            from faster_whisper import BatchedInferencePipeline

            runner = BatchedInferencePipeline(self.whisper_model)
            options.update(batch_size=batch_size, without_timestamps=False)

        logger.info(
            f"Transcribing on {self.device} ({self.compute_type}), beam_size={beam_size}, "
            f"repetition_penalty={options['repetition_penalty']}, "
            f"condition_on_previous_text={options['condition_on_previous_text']}"
            + (f", batched x{batch_size}" if batch_size > 0 else "")
        )
        if audio_offset:
            logger.info(f"Audio starts {audio_offset:.2f}s after the video: shifting subtitles")

        started = time.time()
        segments, info = runner.transcribe(input_file, **options)

        result: Dict[str, Any] = {"text": "", "segments": [], "language": info.language}
        total = round(float(info.duration or 0.0), 1)
        position = 0.0
        with tqdm(
            total=total, desc="Transcribing", unit="s", mininterval=1.0,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
        ) as pbar:
            for segment in segments:
                result["segments"].append(
                    {
                        "start": segment.start + audio_offset,
                        "end": segment.end + audio_offset,
                        "text": segment.text,
                    }
                )
                result["text"] += segment.text
                advance = min(total, segment.end) - position
                if advance > 0:
                    pbar.update(advance)
                    position += advance
            if total > position:
                pbar.update(total - position)

        elapsed = time.time() - started
        speed = (info.duration / elapsed) if elapsed > 0 else 0.0
        logger.info(
            f"Transcribed {len(result['segments'])} segments ({info.language}) "
            f"in {elapsed:.1f}s ({speed:.1f}x realtime)"
        )
        return result

    def process_video_file(self, input_file: str, output_file: str) -> bool:
        """
        Process a video file and add Vietnamese subtitles.

        Args:
            input_file: Path to input video file
            output_file: Path to output video file

        Returns:
            True if processing successful
        """
        work_dir = tempfile.mkdtemp(prefix="vietsub_")
        try:
            logger.info(f"Processing video file: {input_file}")
            ffmpeg = get_ffmpeg_exe()  # fail fast, before the long transcription
            media = probe_media(input_file)
            if not media["has_audio"]:
                logger.error("Input video has no audio stream - nothing to transcribe")
                return False

            # Transcribe entire audio (greedy by default: beam 5 gave longer,
            # coarser-timed cues on the sample videos and is ~1.5x slower)
            beam_size = self.config.get("whisper", {}).get("beam_size", 1)
            result = self._transcribe_file(
                input_file, beam_size=beam_size, audio_offset=media["audio_offset"]
            )
            self._release_whisper()

            # Create subtitle file (also kept in srt/ for the user)
            subtitle_file = os.path.join(work_dir, "subtitles.srt")
            cue_count = self._create_srt_file(result, subtitle_file)
            try:
                srt_copy = prepare_srt_dir("srt") / f"{Path(input_file).stem}_vietnamese.srt"
                shutil.copyfile(subtitle_file, srt_copy)
                logger.info(f"Subtitle file saved to: {srt_copy}")
            except Exception as e:
                logger.warning(f"Could not keep a copy of the SRT: {e}")

            if cue_count == 0:
                logger.warning("No speech detected - copying the video without subtitles")
                subprocess.run(
                    [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                     "-i", input_file, "-c", "copy", output_file],
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
            else:
                # Add subtitles to video
                logger.info("Adding subtitles to video...")
                self._burn_subtitles(
                    ffmpeg, input_file, subtitle_file, output_file, media["audio_codec"],
                    subtitle_style(media["display_size"]),
                )

            logger.info(f"Video processing completed: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            return False
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _nvenc_available(self, ffmpeg: str) -> bool:
        """Probe NVENC with a tiny encode (driver/ffmpeg versions must match)."""
        device_pref = (self.forced_device or self.config.get("whisper", {}).get("device", "auto")).lower()
        if device_pref == "cpu":
            return False
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() == 0:
                return False
        except Exception:
            return False
        try:
            probe = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                 "-i", "color=black:s=256x256:d=0.1", "-frames:v", "1",
                 "-c:v", "h264_nvenc", "-f", "null", "-"],
                capture_output=True,
                timeout=30,
                stdin=subprocess.DEVNULL,
            )
            return probe.returncode == 0
        except Exception:
            return False

    def _video_encoder_candidates(self, ffmpeg: str) -> List[List[str]]:
        """Encoder argument lists to try in order (GPU NVENC when it works, then libx264).

        Args:
            ffmpeg: ffmpeg executable

        Returns:
            List of ffmpeg video-encoder argument lists
        """
        video_config = self.config.get("video", {}) or {}
        encoder = str(video_config.get("encoder", "auto")).lower()
        crf = str(video_config.get("crf", 23))
        x264 = [
            "-c:v", "libx264", "-preset", str(video_config.get("preset", "veryfast")),
            "-crf", crf, "-pix_fmt", "yuv420p",
        ]
        nvenc = [
            "-c:v", "h264_nvenc", "-preset", str(video_config.get("nvenc_preset", "p5")),
            "-rc", "vbr", "-cq", crf, "-b:v", "0", "-pix_fmt", "yuv420p",
        ]
        if encoder == "libx264":
            return [x264]
        if encoder in ("h264_nvenc", "nvenc"):
            return [nvenc, x264]
        if encoder == "auto":
            return [nvenc, x264] if self._nvenc_available(ffmpeg) else [x264]
        return [["-c:v", encoder, "-pix_fmt", "yuv420p"], x264]

    def _burn_subtitles(
        self,
        ffmpeg: str,
        input_file: str,
        subtitle_file: str,
        output_file: str,
        audio_codec: Optional[str] = "aac",
        force_style: Optional[str] = None,
    ) -> None:
        """Burn an SRT into the video, trying the configured encoders in order.

        ffmpeg runs inside the SRT's folder so the subtitles filter only sees a
        plain file name (Windows drive letters and quotes break its syntax).

        Args:
            ffmpeg: ffmpeg executable
            input_file: Source video
            subtitle_file: SRT path
            output_file: Destination video
            audio_codec: Source audio codec (copied when MP4-compatible)
            force_style: Optional libass style override (see subtitle_style())

        Raises:
            RuntimeError: If every encoder failed
        """
        subtitles_filter = f"subtitles={os.path.basename(subtitle_file)}"
        if force_style:
            subtitles_filter += f":force_style='{force_style}'"
        base = [
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-stats", "-y",
            "-i", os.path.abspath(input_file),
            "-vf", subtitles_filter,
        ]
        tail = audio_copy_args(output_file, audio_codec) + [os.path.abspath(output_file)]
        for encoder_args in self._video_encoder_candidates(ffmpeg):
            logger.info(f"Encoding with {encoder_args[1]}...")
            proc = subprocess.run(
                base + encoder_args + tail,
                cwd=os.path.dirname(os.path.abspath(subtitle_file)),
                stdin=subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                return
            logger.warning(f"ffmpeg with {encoder_args[1]} failed (exit code {proc.returncode})")
        raise RuntimeError("ffmpeg could not burn the subtitles into the video")

    def _create_srt_file(
        self, transcription_result: dict, output_file: str, is_export_mode: bool = False
    ) -> int:
        """
        Create SRT subtitle file from transcription result.
        Optimized with batch translation for better performance.
        Includes anti-repetition filtering for long videos.

        Args:
            transcription_result: Whisper transcription result
            output_file: Path to output SRT file
            is_export_mode: Kept for compatibility (both modes share settings)

        Returns:
            Number of subtitle cues written

        Raises:
            RuntimeError: If no line at all could be translated
        """
        translator = self.get_translation_service()

        # Drop non-speech segments ("[Music]", "♪", "(applause)") up front so
        # they neither become empty cues nor hallucinated translations.
        segments = [
            seg
            for seg in transcription_result.get("segments", [])
            if translator._clean_text(seg.get("text", ""))
        ]

        # Anti-repetition: Filter repeated segments (Whisper hallucination)
        anti_rep_config = self.config.get("anti_repetition", {})
        original_count = len(segments)
        segments = translator.detect_segment_repetition(
            segments,
            similarity_threshold=anti_rep_config.get("segment_similarity_threshold", 0.85),
            max_allowed_repetitions=anti_rep_config.get("max_allowed_repetitions", 2),
        )
        if len(segments) < original_count:
            logger.info(
                f"🔄 Anti-repetition: Filtered {original_count - len(segments)} "
                f"repetitive segments"
            )

        # Whisper often cuts a sentence in two; translating the halves separately
        # gives broken Vietnamese, so short fragments are merged first.
        segments = self._merge_fragments(segments)

        if not segments:
            logger.warning("No speech segments to translate")
            with open(output_file, "w", encoding="utf-8"):
                pass
            return 0

        # Whisper's detected language is the right NLLB source when not forced
        source_lang = self.forced_source_lang or transcription_result.get("language") or "en"
        texts = [seg["text"].strip() for seg in segments]
        translated: List[Optional[str]] = [None] * len(texts)

        # The service sorts by length and batches internally; the outer
        # chunks only drive the progress bar.
        step = max(64, 4 * translator.batch_size)
        logger.info(f"Translating {len(texts)} segments ({source_lang} → vi)...")
        with tqdm(total=len(texts), desc="Translating", unit="seg", mininterval=1.0) as pbar:
            for start in range(0, len(texts), step):
                batch = texts[start : start + step]
                try:
                    results = translator.translate_batch(batch, source_language=source_lang)
                except Exception as e:
                    logger.warning(f"Batch translation failed: {e}")
                    results = [None] * len(batch)
                for j, item in enumerate(results):
                    if item and item.translated_text:
                        translated[start + j] = item.translated_text
                pbar.update(len(batch))

        failed = sum(1 for t in translated if t is None)
        if failed == len(translated):
            raise RuntimeError(
                "Translation failed for every subtitle line (no working translation model)"
            )
        if failed:
            logger.warning(f"⚠️  {failed}/{len(texts)} lines could not be translated; kept original text")
        logger.info(f"✅ Batched translation completed: {len(texts) - failed} segments")

        # Write to file preserving order
        with open(output_file, "w", encoding="utf-8") as f:
            for i, (segment, text) in enumerate(zip(segments, translated), 1):
                start_s = float(segment["start"])
                end_s = max(float(segment["end"]), start_s + 0.1)
                f.write(f"{i}\n")
                f.write(f"{self._format_srt_time(start_s)} --> {self._format_srt_time(end_s)}\n")
                f.write(f"{(text or texts[i - 1]).strip()}\n\n")
        return len(segments)

    def _merge_fragments(self, segments: List[dict]) -> List[dict]:
        """Merge a segment into the previous one when they form one sentence.

        Only when the previous text has no sentence-final punctuation, the pause
        between them is short, and the merged cue still fits the configured
        duration/length (about two subtitle lines).

        Args:
            segments: Segments with start/end/text, in time order

        Returns:
            New list of (possibly merged) segments
        """
        srt_config = self.config.get("srt", {}) or {}
        if not srt_config.get("merge_fragments", True) or len(segments) < 2:
            return segments
        max_gap = float(srt_config.get("max_merge_gap", 0.8))
        max_seconds = float(srt_config.get("max_cue_seconds", 6.0))
        max_chars = int(srt_config.get("max_cue_chars", 84))
        # Japanese/Chinese are written without spaces; Korean uses spaces but,
        # like them, its characters are double-width on screen.
        no_space = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
        wide = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
        terminal = tuple(".!?…。！？")

        merged: List[dict] = []
        for segment in segments:
            text = segment["text"].strip()
            if merged:
                prev = merged[-1]
                joined_cjk = bool(no_space.search(prev["text"][-1:])) and bool(no_space.search(text[:1]))
                limit = max_chars // 2 if wide.search(prev["text"] + text) else max_chars
                candidate = prev["text"] + ("" if joined_cjk else " ") + text
                if (
                    not prev["text"].endswith(terminal)
                    and segment["start"] - prev["end"] <= max_gap
                    and segment["end"] - prev["start"] <= max_seconds
                    and len(candidate) <= limit
                ):
                    prev["text"] = candidate
                    prev["end"] = segment["end"]
                    continue
            merged.append({"start": segment["start"], "end": segment["end"], "text": text})

        if len(merged) < len(segments):
            logger.info(f"Merged sentence fragments: {len(segments)} → {len(merged)} subtitle lines")
        return merged

    def _format_srt_time(self, seconds: float) -> str:
        """
        Format time for SRT subtitle format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (HH:MM:SS,mmm)
        """
        total_ms = int(round(max(0.0, seconds) * 1000))
        hours, rest = divmod(total_ms, 3_600_000)
        minutes, rest = divmod(rest, 60_000)
        secs, millis = divmod(rest, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def export_srt_only(self, input_file: str, output_srt_path: str) -> bool:
        """
        Export subtitle file (.srt) only without creating video.
        Optimized for faster processing with minimal quality loss.

        Args:
            input_file: Path to input video file
            output_srt_path: Path to output .srt file

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Extracting subtitles from: {input_file}")
            logger.info(f"Output subtitle file: {output_srt_path}")
            media = probe_media(input_file)
            if not media["has_audio"]:
                logger.error("Input video has no audio stream - nothing to transcribe")
                return False

            # Transcribe audio (with export optimization: greedy decoding)
            logger.info("Transcribing audio (optimized for speed)...")
            beam_size = self.config.get("whisper", {}).get("export_beam_size", 1)
            result = self._transcribe_file(
                input_file, beam_size=beam_size, audio_offset=media["audio_offset"]
            )
            self._release_whisper()

            # Create subtitle file (with optimized batch size for export)
            logger.info("Creating subtitle file with translations...")
            self._create_srt_file(result, output_srt_path, is_export_mode=True)

            logger.info(f"Subtitle file created successfully: {output_srt_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting subtitle file: {e}")
            return False

    # -------------------------------------------------------------- voiceover

    def create_voiceover_video(
        self, input_file: str, srt_file: str, output_file: str
    ) -> bool:
        """
        Create video with Vietnamese TTS voiceover from SRT file.

        This method:
        1. Parses SRT file to get text and timings
        2. Generates Vietnamese TTS audio for each segment using VieNeu
        3. Lowers the original audio (~25%) and mixes the TTS voice over it
        4. Muxes the mixed audio with the untouched video stream (no re-encode)

        Args:
            input_file: Path to input video file
            srt_file: Path to SRT subtitle file
            output_file: Path to output video file

        Returns:
            True if successful, False otherwise
        """
        try:
            from vieneu import Vieneu
            from vieneu_utils.phonemize_text import (
                phonemize_text_with_emotions,
            )
            import pydub  # noqa: F401  (used by the mixing helpers)
        except ImportError as e:
            logger.error(f"Missing required library: {e}")
            logger.error(
                "Please install: pip install vieneu pydub --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/"
            )
            return False

        try:
            ffmpeg = get_ffmpeg_exe()  # fail fast, before TTS generation

            # Whisper/NLLB are no longer needed: give their RAM/VRAM to the TTS model
            self._release_models()

            logger.info(f"Creating voiceover video from: {input_file}")
            logger.info(f"Using SRT file: {srt_file}")
            logger.info(f"Using VieNeu TTS voice: {self.tts_voice}")
            logger.info(f"TTS temperature: {self.tts_temperature} (0.0-1.5, higher=more expressive)")

            # Parse SRT file
            segments = self._parse_srt_file(srt_file)
            if not segments:
                logger.error("No segments found in SRT file")
                return False

            logger.info(f"Found {len(segments)} subtitle segments")

            # Create temp directory for audio files
            temp_dir = tempfile.mkdtemp(prefix="voiceover_")

            try:
                media = probe_media(input_file)
                video_duration, has_audio = media["duration"], media["has_audio"]
                if video_duration <= 0:
                    video_duration = max(seg["end"] for seg in segments) + 1.0

                # Initialize VieNeu TTS (v3 Turbo)
                logger.info("Initializing VieNeu TTS v3 Turbo engine...")
                tts = Vieneu()  # Auto-detects GPU/CPU, defaults to v3 Turbo

                # Resolve the preset voice once for all segments
                voice_data = tts.get_preset_voice(self.tts_voice)

                # Check if GPU batch engine is available (PyTorch backend)
                can_batch = (
                    hasattr(tts, "backend") and tts.backend == "pytorch"
                    and hasattr(tts, "engine") and tts.engine is not None
                )

                if can_batch:
                    logger.info(
                        "GPU detected — using V3TurboBatchEngine "
                        "for accelerated TTS generation"
                    )
                    try:
                        from vieneu.v3_turbo_serve import V3TurboBatchEngine

                        batch_engine = V3TurboBatchEngine(tts.engine)
                        ref_codes = voice_data.get("codes")
                        voice_token_id = voice_data.get("reserved_id")

                        # Collect valid segments
                        valid_batches = []
                        for i, segment in enumerate(segments):
                            text = segment["text"].strip()
                            if text:
                                valid_batches.append((i, text))

                        total_valid = len(valid_batches)
                        batch_size = int(self.config.get("tts", {}).get("batch_size", 32))
                        generated = 0

                        logger.info(
                            f"Generating TTS for {total_valid} segments "
                            f"in batches of {batch_size}..."
                        )

                        for batch_start in range(0, total_valid, batch_size):
                            batch_end = min(batch_start + batch_size, total_valid)
                            batch_items = valid_batches[batch_start:batch_end]

                            # Pre-phonemize for this batch
                            requests = []
                            for idx, text in batch_items:
                                phonemes = phonemize_text_with_emotions(text)
                                requests.append({
                                    "phonemes": phonemes,
                                    "ref_codes": ref_codes,
                                    "voice_token_id": voice_token_id,
                                })

                            try:
                                wavs = batch_engine.generate_batch(
                                    requests,
                                    temperature=self.tts_temperature,
                                    max_new_frames=300,
                                )
                                for (idx, _), wav in zip(batch_items, wavs):
                                    if len(wav) > 0:
                                        output_path = os.path.join(
                                            temp_dir, f"tts_{idx}.wav"
                                        )
                                        tts.save(wav, output_path)
                                        generated += 1
                            except Exception as e:
                                logger.warning(
                                    f"Batch failed at group "
                                    f"{batch_start // batch_size + 1}: {e}. "
                                    "Falling back to individual..."
                                )
                                for idx, text in batch_items:
                                    try:
                                        output_path = os.path.join(
                                            temp_dir, f"tts_{idx}.wav"
                                        )
                                        audio = tts.infer(
                                            text,
                                            voice=self.tts_voice,
                                            temperature=self.tts_temperature,
                                        )
                                        tts.save(audio, output_path)
                                        generated += 1
                                    except Exception as e2:
                                        logger.warning(
                                            f"Failed TTS for segment {idx}: {e2}"
                                        )

                        logger.info(
                            f"Batch TTS: {generated}/{total_valid} segments"
                        )

                    except Exception as e:
                        logger.warning(
                            f"Batch engine init failed: {e}. "
                            "Falling back to sequential..."
                        )
                        can_batch = False

                if not can_batch:
                    # Sequential TTS (CPU/ONNX or fallback)
                    logger.info(
                        f"Sequential TTS (backend: "
                        f"{getattr(tts, 'backend', 'unknown')})"
                    )
                    for i, segment in enumerate(
                        tqdm(segments, desc="Generating TTS Audio", mininterval=1.0)
                    ):
                        text = segment["text"].strip()
                        if not text:
                            continue
                        output_path = os.path.join(temp_dir, f"tts_{i}.wav")
                        if os.path.exists(output_path):
                            continue  # already produced by the batch engine
                        try:
                            audio = tts.infer(
                                text,
                                voice=self.tts_voice,
                                temperature=self.tts_temperature,
                            )
                            tts.save(audio, output_path)
                        except Exception as e:
                            logger.warning(
                                f"Failed TTS for segment {i}: {e}"
                            )
                            continue

                # Cleanup VieNeu TTS
                tts.close()
                logger.info("VieNeu TTS generation completed")

                expected = sum(1 for seg in segments if seg["text"].strip())
                produced = sum(
                    1 for i in range(len(segments))
                    if os.path.exists(os.path.join(temp_dir, f"tts_{i}.wav"))
                )
                if produced == 0:
                    logger.error("TTS failed for every segment")
                    return False
                if produced < expected:
                    logger.warning(f"⚠️  TTS missing for {expected - produced}/{expected} segments")

                # Mixing phase (tempo fitting in parallel, then one numpy buffer)
                logger.info("Mixing audio segments...")
                # The voice is mixed into the original audio track, which starts
                # audio_offset seconds into the video; without an original
                # track the voice is the audio and starts with the video.
                combined_audio_path = self._mix_voiceover(
                    ffmpeg, segments, temp_dir, video_duration,
                    timeline_offset=media["audio_offset"] if has_audio else 0.0,
                )

                # Create video with mixed audio (original + TTS voiceover)
                logger.info("Creating final video with voiceover...")
                self._mux_voiceover(
                    ffmpeg, input_file, combined_audio_path, output_file, has_audio,
                    media["video_codec"],
                )

                logger.info(f"Voiceover video created successfully: {output_file}")
                return True

            finally:
                # Cleanup temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Error creating voiceover video: {e}")
            return False

    def _fit_tts_segment(
        self, ffmpeg: str, temp_dir: str, index: int, start_ms: int, cue_ms: int, slot_ms: int
    ) -> Optional[Tuple[int, Any]]:
        """Load one TTS clip and fit its length to the subtitle slot.

        Speeds up (max 1.3x) only when the clip would run into the next cue,
        slows down (min 0.85x) when it is much shorter than its own cue, and
        trims (with a short fade) at the next cue instead of at the cue end.
        The clip starts 50 ms after the cue start and keeps 50 ms clear
        before the next one.

        Args:
            ffmpeg: ffmpeg executable
            temp_dir: Folder with tts_{index}.wav
            index: Segment index
            start_ms: Cue start (ms)
            cue_ms: Cue duration (ms)
            slot_ms: Time until the next cue starts (ms)

        Returns:
            (position_ms, AudioSegment, truncated) or None
        """
        from pydub import AudioSegment

        pad_ms = 50
        window_ms = slot_ms - 2 * pad_ms
        tts_path = os.path.join(temp_dir, f"tts_{index}.wav")
        if not os.path.exists(tts_path) or window_ms <= 0:
            return None
        try:
            # Load TTS audio (VieNeu outputs WAV format)
            tts_audio = AudioSegment.from_wav(tts_path)
            current_duration = len(tts_audio)

            # Determine tempo change for natural pacing (on the raw clip length)
            final_speed = 1.0
            if current_duration > window_ms:
                # Audio longer than the gap to the next cue → speed up gently
                final_speed = min(current_duration / window_ms, 1.3)
            elif cue_ms > 0 and current_duration < cue_ms * 0.7:
                # Audio much shorter than its cue → slow down slightly to fill naturally
                final_speed = max(current_duration / cue_ms, 0.85)

            # Apply tempo change if needed
            if abs(final_speed - 1.0) > 0.01:
                processed_path = os.path.join(temp_dir, f"tts_{index}_processed.wav")
                result = subprocess.run(
                    [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", tts_path,
                     "-filter:a", f"atempo={final_speed:.4f}", "-y", processed_path],
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                )
                if result.returncode == 0 and os.path.exists(processed_path):
                    tts_audio = AudioSegment.from_wav(processed_path)
                else:
                    logger.warning(f"atempo ffmpeg failed for segment {index}, using original audio")

            # Never run into the next cue; fade instead of a hard cut (no click)
            truncated = len(tts_audio) > window_ms
            if truncated:
                tts_audio = tts_audio[:window_ms].fade_out(40)
            return start_ms + pad_ms, tts_audio, truncated
        except Exception as e:
            logger.warning(f"Failed to process segment {index}: {e}")
            return None

    def _mix_voiceover(
        self,
        ffmpeg: str,
        segments: List[dict],
        temp_dir: str,
        duration_s: float,
        timeline_offset: float = 0.0,
    ) -> str:
        """Mix all TTS clips into one WAV (linear time, unlike pydub.overlay).

        Args:
            ffmpeg: ffmpeg executable
            segments: Parsed SRT segments (times on the video timeline)
            temp_dir: Folder with tts_{i}.wav files
            duration_s: Video duration in seconds
            timeline_offset: Seconds at which the track this WAV is mixed into
                starts (the original audio's start offset); cue times are
                shifted by it so the voice is not late

        Returns:
            Path of the mixed WAV

        Raises:
            RuntimeError: If no voice clip could be used
        """
        import numpy as np

        video_end_ms = int(duration_s * 1000)
        jobs = []
        starts = [int(seg["start"] * 1000) for seg in segments]
        for i, segment in enumerate(segments):
            if not segment["text"].strip():
                continue
            start_ms = starts[i]
            end_ms = int(segment["end"] * 1000)
            next_start = next(
                (starts[j] for j in range(i + 1, len(segments)) if segments[j]["text"].strip()),
                video_end_ms,
            )
            slot_ms = max(end_ms, next_start) - start_ms
            jobs.append((i, start_ms, end_ms - start_ms, slot_ms))

        workers = max(1, min(8, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            clips = list(
                tqdm(
                    pool.map(lambda job: self._fit_tts_segment(ffmpeg, temp_dir, *job), jobs),
                    total=len(jobs),
                    desc="Processing Audio",
                    mininterval=1.0,
                )
            )
        clips = [clip for clip in clips if clip]
        if not clips:
            raise RuntimeError("No voice line could be placed in the video")
        if len(clips) < len(jobs):
            logger.warning(f"⚠️  {len(jobs) - len(clips)}/{len(jobs)} voice lines were dropped")
        truncated = sum(1 for clip in clips if clip[2])
        if truncated:
            logger.warning(
                f"⚠️  {truncated}/{len(clips)} voice lines were longer than their slot even "
                f"at 1.3x speed and were cut short"
            )

        rate = clips[0][1].frame_rate
        offset_ms = int(round(timeline_offset * 1000))
        mix = np.zeros(int(duration_s * rate) + rate, dtype=np.float32)
        for start_ms, clip, _ in clips:
            clip = clip.set_frame_rate(rate).set_channels(1).set_sample_width(2)
            samples = np.frombuffer(clip.raw_data, dtype=np.int16)
            start = int((start_ms - offset_ms) * rate / 1000)
            if start < 0:
                # Cue begins before the original audio track: drop the part of
                # the clip that would play before it instead of delaying it
                samples = samples[-start:]
                start = 0
            end = min(len(mix), start + len(samples))
            if end > start:
                mix[start:end] += samples[: end - start]

        combined_audio_path = os.path.join(temp_dir, "combined_audio.wav")
        np.clip(mix, -32768, 32767, out=mix)
        block = rate * 60  # write a minute at a time (no second full-length copy)
        with wave.open(combined_audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(rate)
            for i in range(0, len(mix), block):
                wav_file.writeframes(mix[i : i + block].astype(np.int16).tobytes())
        return combined_audio_path

    @staticmethod
    def _amix_supports_normalize(ffmpeg: str) -> bool:
        """True when this ffmpeg's amix filter has the `normalize` option (FFmpeg 4.4+)."""
        try:
            out = subprocess.run(
                [ffmpeg, "-hide_banner", "-h", "filter=amix"],
                capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
            )
            return "normalize" in (out.stdout + out.stderr)
        except Exception:
            return False

    def _mux_voiceover(
        self,
        ffmpeg: str,
        input_file: str,
        voice_path: str,
        output_file: str,
        has_audio: bool,
        video_codec: Optional[str] = "h264",
    ) -> None:
        """Mux the voiceover with the original video stream copied as-is.

        Args:
            ffmpeg: ffmpeg executable
            input_file: Source video
            voice_path: Mixed TTS WAV
            output_file: Destination video
            has_audio: Whether the source has an audio track to keep (lowered to ~25%)
            video_codec: Source video codec; only H.264/HEVC are copied, others
                (MJPEG, ProRes, VP8, ...) are re-encoded so the MP4 plays everywhere

        Raises:
            subprocess.CalledProcessError: If ffmpeg fails with every encoder
        """
        if has_audio:
            # Keep original audio (lower volume) + mix TTS voiceover.
            # amix normally divides every input by the input count, which left
            # the voice ~7.5 dB below the intended level; normalize=0 plus a
            # limiter (level=0: no auto make-up gain) keeps the intended
            # 12 dB voice/original balance without clipping.
            if self._amix_supports_normalize(ffmpeg):
                mix_filter = (
                    "[0:a]volume=0.25[original];[1:a]volume=1.0[voiceover];"
                    "[original][voiceover]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
                    "alimiter=limit=0.89:level=0[aout]"
                )
            else:  # FFmpeg < 4.4
                mix_filter = (
                    "[0:a]volume=0.3[original];[1:a]volume=1.2[voiceover];"
                    "[original][voiceover]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                )
            audio_args = ["-filter_complex", mix_filter, "-map", "0:v:0", "-map", "[aout]"]
        else:
            audio_args = ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]

        base = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-stats", "-y",
                "-i", input_file, "-i", voice_path] + audio_args
        tail = ["-c:a", "aac", "-b:a", "192k"]
        if Path(output_file).suffix.lower() in (".mp4", ".m4v", ".mov"):
            tail += ["-movflags", "+faststart"]
        tail.append(output_file)
        # Nothing is drawn on the frames, so the video stream is copied
        # (seconds instead of a full re-encode, and no quality loss).
        attempts = self._video_encoder_candidates(ffmpeg)
        if video_codec == "h264":
            attempts = [["-c:v", "copy"]] + attempts
        elif video_codec == "hevc":
            # 'hvc1' tag: Apple/QuickTime players refuse HEVC tagged 'hev1'
            attempts = [["-c:v", "copy", "-tag:v", "hvc1"]] + attempts
        for video_args in attempts:
            ffmpeg_cmd = base + video_args + tail
            logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            proc = subprocess.run(ffmpeg_cmd, stdin=subprocess.DEVNULL)
            if proc.returncode == 0:
                return
            logger.warning(f"ffmpeg with video {video_args[1]} failed (exit code {proc.returncode})")
        raise subprocess.CalledProcessError(proc.returncode, ffmpeg_cmd)

    def _parse_srt_file(self, srt_path: str) -> list:
        """
        Parse SRT file and return list of segments with timing and text.

        Args:
            srt_path: Path to SRT file

        Returns:
            List of dicts with 'start', 'end', 'text' keys
        """
        segments = []
        timing_re = re.compile(
            r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
        )

        try:
            # utf-8-sig also accepts files saved with a BOM (Notepad)
            with open(srt_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            # Split by blank lines to get individual subtitle blocks
            blocks = re.split(r"\n\s*\n", content.strip())

            for block in blocks:
                lines = [line.strip() for line in block.strip().split("\n")]
                for k, line in enumerate(lines):
                    timing_match = timing_re.search(line)
                    if timing_match:
                        text = " ".join(l for l in lines[k + 1 :] if l)
                        if text:
                            segments.append(
                                {
                                    "start": self._parse_srt_timestamp(timing_match.group(1)),
                                    "end": self._parse_srt_timestamp(timing_match.group(2)),
                                    "text": text,
                                }
                            )
                        break

        except Exception as e:
            logger.error(f"Error parsing SRT file: {e}")

        return segments

    def _parse_srt_timestamp(self, timestamp: str) -> float:
        """
        Parse SRT timestamp format (HH:MM:SS,mmm) to seconds.

        Args:
            timestamp: Timestamp string in format HH:MM:SS,mmm

        Returns:
            Time in seconds as float
        """
        parts = timestamp.replace(",", ".").split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])

        return hours * 3600 + minutes * 60 + seconds


def prepare_srt_dir(path: str = "srt", clean: bool = False) -> Path:
    """Prepare the SRT directory.

    Existing files are kept by default: wiping the folder would delete the
    output of another run working at the same time.

    Args:
        path: Directory path
        clean: Remove existing contents first

    Returns:
        The directory path
    """
    srt_dir = Path(path)
    if clean and srt_dir.exists():
        try:
            shutil.rmtree(srt_dir)
            logger.info("🗑️  Cleaned up old subtitle files")
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not clean srt directory: {e}")

    srt_dir.mkdir(parents=True, exist_ok=True)
    return srt_dir


_VOICE_CHOICES = [
    "Ngọc Lan", "Ngọc Linh", "Trúc Ly", "Mỹ Duyên",
    "Xuân Vĩnh", "Thái Sơn", "Gia Bảo", "Đức Trí", "Trọng Hữu", "Bình An",
]


_VOICE_CHAR_MAP = str.maketrans({"Đ": "D", "đ": "d"})


def _ascii_voice_key(s: str) -> str:
    """Normalize Vietnamese voice name to ASCII for fuzzy matching."""
    decomposed = unicodedata.normalize("NFKD", s).translate(_VOICE_CHAR_MAP)
    return re.sub(r'[̀-ͯ]', '', decomposed).lower().replace(' ', '')


_VOICE_ASCII_MAP: dict[str, str] = {_ascii_voice_key(v): v for v in _VOICE_CHOICES}


def _normalize_voice(name: str) -> str:
    """Normalize voice name, accepting ASCII fallback (e.g. 'Ngoc Lan' -> 'Ngọc Lan')."""
    if name in _VOICE_CHOICES:
        return name
    key = _ascii_voice_key(name)
    if key in _VOICE_ASCII_MAP:
        return _VOICE_ASCII_MAP[key]
    raise argparse.ArgumentTypeError(
        f"Invalid voice '{name}'. Choose from: {', '.join(_VOICE_CHOICES)}"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Offline Vietnamese Subtitle Generator (100% Local Processing)"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable video processing mode (provide input via --input or positional)",
    )
    parser.add_argument(
        "--input",
        dest="input",
        help="Input video file path (optional if positional provided)",
    )
    parser.add_argument(
        "--output", help="Output video file path (for video processing)"
    )
    parser.add_argument(
        "videopath",
        nargs="?",
        help="Positional video file path used with --video flag",
    )
    parser.add_argument(
        "--jp",
        action="store_true",
        help="[DEPRECATED] Use --language ja instead. Treat source as Japanese (backward compatibility)",
    )
    parser.add_argument(
        "--language",
        "--lang",
        dest="language",
        choices=["en", "ja", "zh", "ko", "th", "id"],
        default=None,
        help="Source language code: en (English), ja (Japanese), zh (Chinese), ko (Korean), th (Thai), id (Indonesian). Default: en",
    )
    parser.add_argument(
        "--model",
        dest="model",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        default=None,
        help="Whisper model size: tiny (fastest), base (recommended), small, medium, large (large-v3, most accurate), turbo (large-v3-turbo: near-large accuracy, much faster). Default: base",
    )
    parser.add_argument(
        "--device",
        dest="device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Inference device for Whisper and NLLB (default: config, normally auto = GPU if usable, else CPU)",
    )
    parser.add_argument(
        "--export-srt",
        action="store_true",
        help="Export subtitle file (.srt) to /srt directory without creating video",
    )
    parser.add_argument(
        "--voiceover",
        action="store_true",
        help="Create video with Vietnamese TTS voiceover (original audio kept at ~25%% volume)",
    )
    parser.add_argument(
        "--srt",
        dest="srt_file",
        help="Path to existing SRT file (for voiceover mode, optional - will generate if not provided)",
    )
    parser.add_argument(
        "--voice",
        dest="voice",
        type=_normalize_voice,
        default="Ngọc Lan",
        metavar="VOICE",
        help=f"VieNeu v3 Turbo TTS voice. Also accepts ASCII (e.g., 'Ngoc Lan'). Default: Ngọc Lan",
    )

    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Initialize application
    app = VietnameseSubtitleGenerator(args.config)

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        app.stop()
        # Non-zero exit so callers (the GUI) never report an interrupted run as success
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def apply_common_options(force_language: bool = True) -> None:
        # Set forced source language: support --language flag or legacy --jp flag
        if args.language:
            app.forced_source_lang = args.language
        elif args.jp:
            logger.warning(
                "⚠️  --jp flag is deprecated. Please use --language ja instead."
            )
            app.forced_source_lang = "ja"
        elif force_language:
            app.forced_source_lang = "en"

        # Set forced model size if provided via --model flag
        if args.model:
            app.forced_model_size = args.model
            logger.info(f"Using model size: {args.model}")

        if args.device:
            app.forced_device = args.device
            app.config.setdefault("translation", {})["device"] = args.device

    try:
        input_file = args.input or args.videopath
        if (args.export_srt or args.video or args.voiceover) and not input_file:
            parser.print_usage()
            logger.error(
                "No input video provided. Use --input <file> or place the path after flags."
            )
            sys.exit(2)
        if input_file and not os.path.isfile(input_file):
            logger.error(f"Input file not found: {input_file}")
            sys.exit(2)

        if args.export_srt:
            # Export subtitle mode - only create .srt file
            srt_dir = prepare_srt_dir("srt")

            # Generate output .srt filename
            output_srt = srt_dir / f"{Path(input_file).stem}_vietnamese.srt"

            apply_common_options()
            if app.load_whisper_model():
                success = app.export_srt_only(input_file, str(output_srt))
                if success:
                    logger.info(f"Subtitle file saved to: {output_srt}")
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
        elif args.video:
            # Video processing mode - create video with subtitles
            if not args.output:
                args.output = f"{Path(input_file).stem}_vietnamese_subtitles.mp4"

            apply_common_options()
            if app.load_whisper_model():
                success = app.process_video_file(input_file, args.output)
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
        elif args.voiceover:
            # Voiceover mode - create video with Vietnamese TTS voiceover
            apply_common_options()

            # Set TTS voice based on --voice argument (VieNeu voices)
            app.tts_voice = args.voice
            logger.info(f"Using VieNeu voice: {args.voice}")

            # Check if SRT file is provided or needs to be generated
            srt_file = args.srt_file
            if not srt_file:
                # Generate SRT file first
                logger.info("No SRT file provided, generating subtitles first...")

                srt_dir = prepare_srt_dir("srt")
                srt_file = str(srt_dir / f"{Path(input_file).stem}_vietnamese.srt")

                if app.load_whisper_model():
                    if not app.export_srt_only(input_file, srt_file):
                        logger.error("Failed to generate SRT file")
                        sys.exit(1)
                else:
                    sys.exit(1)

            # Generate output filename
            if not args.output:
                args.output = str(
                    Path.home() / "Downloads" / f"{Path(input_file).stem}_voiceover.mp4"
                )

            logger.info("Creating voiceover video...")
            success = app.create_voiceover_video(input_file, srt_file, args.output)
            if success:
                logger.info(f"Voiceover video saved to: {args.output}")
            sys.exit(0 if success else 1)
        else:
            # Real-time mode (language auto-detected unless --language is given)
            apply_common_options(force_language=False)
            app.start()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
