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
import signal
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import yaml


# Import faster-whisper (optimized for speed and accuracy)
from faster_whisper import WhisperModel

from loguru import logger
import keyboard

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.audio_processor import AudioProcessor
from src.subtitle_overlay import SubtitleOverlay
from src.translator import TranslationService, TranslationResult


class VietnameseSubtitleGenerator:
    """Main application class for offline video subtitle generation (100% local processing)."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the subtitle generator application.

        Args:
            config_path: Path to the configuration file
        """
        # Clean up temporary files from previous runs
        self._cleanup_temp_files()

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        self.whisper_model = None
        self.device = None
        self.forced_source_lang = None
        self.forced_model_size = None  # User-selected model size
        self.audio_processor = None
        self.subtitle_overlay = None
        self.translation_service = None

        # Application state
        self.is_running = False
        self.is_paused = False

        # TTS voice setting (default: female Vietnamese voice)
        self.tts_voice = "vi-VN-HoaiMyNeural"  # Female voice (default)
        # Available voices: vi-VN-HoaiMyNeural (female), vi-VN-NamMinhNeural (male)

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

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files from previous runs."""
        temp_files = ["temp_audio.wav", "temp_subtitles.srt"]

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    # Use plain text without special unicode characters to avoid encoding issues
                    print(f"Removed existing temporary file: {temp_file}")
                except Exception as e:
                    print(f"Warning: Could not remove {temp_file}: {e}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
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

    def initialize_components(self) -> bool:
        """
        Initialize all application components.

        Returns:
            True if all components initialized successfully
        """
        try:
            # Initialize faster-whisper model (speech-to-text)
            logger.info("Loading faster-whisper model...")
            whisper_config = self.config["whisper"]
            # Use forced model size if set via --model flag, otherwise use config
            model_size = self.forced_model_size or whisper_config.get(
                "model_size", "base"
            )
            device = whisper_config.get("device", "auto")
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            self.device = device
            self.whisper_model = WhisperModel(
                model_size, device=device, compute_type="float32"
            )
            logger.info(f"faster-whisper model '{model_size}' loaded on {device}")

            # Initialize audio processor
            logger.info("Initializing audio processor...")
            try:
                self.audio_processor = AudioProcessor(self.config)
                self.audio_processor.set_callbacks(on_audio_chunk=self._on_audio_chunk)
            except Exception as e:
                logger.warning(
                    f"AudioProcessor initialization failed: {e}. Continuing without audio processor."
                )
                self.audio_processor = None

            # Initialize translation service
            logger.info("Initializing translation service...")
            self.translation_service = TranslationService(self.config)
            self.translation_service.set_callback(self._on_translation_complete)
            self.translation_service.start_async_processing()

            # Initialize subtitle overlay
            logger.info("Initializing subtitle overlay...")
            self.subtitle_overlay = SubtitleOverlay(self.config)

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    def setup_hotkeys(self) -> None:
        """Setup keyboard hotkeys for application control."""
        hotkey_config = self.config.get("hotkeys", {})

        try:
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
            # Transcribe audio using Whisper
            start_time = time.time()
            result = self.whisper_model.transcribe(
                audio_file_path,
                language=self.config["whisper"].get("language", "auto"),
                temperature=self.config["whisper"].get("temperature", 0.0),
                best_of=self.config["whisper"].get("best_of", 5),
                beam_size=self.config["whisper"].get("beam_size", 5),
                fp16=False if self.device == "cpu" else True,
            )

            transcription_time = time.time() - start_time

            # Extract text
            text = result["text"].strip()
            detected_language = result.get("language", "unknown")

            if text:
                logger.info(f"Transcribed ({detected_language}): {text}")
                self.stats["transcriptions"] += 1

                if self.config.get("debug", {}).get("show_processing_time", False):
                    logger.debug(f"Transcription took {transcription_time:.2f} seconds")

                # Queue for translation
                source_lang = self.forced_source_lang or detected_language
                self.translation_service.translate_async(
                    text,
                    source_language=source_lang,
                    callback=lambda result: self._on_translation_complete(result),
                )

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

        cache_stats = self.translation_service.get_cache_stats()

        logger.info("=== Statistics ===")
        logger.info(f"Runtime: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        logger.info(f"Transcriptions: {self.stats['transcriptions']}")
        logger.info(f"Translations: {self.stats['translations']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Cache usage: {cache_stats['cache_usage_percent']}%")
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

    def process_video_file(self, input_file: str, output_file: str) -> bool:
        """
        Process a video file and add Vietnamese subtitles.

        Args:
            input_file: Path to input video file
            output_file: Path to output video file

        Returns:
            True if processing successful
        """
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            import subprocess

            logger.info(f"Processing video file: {input_file}")

            # Load video
            video = VideoFileClip(input_file)
            audio = video.audio

            # Extract audio to temporary file
            temp_audio = "temp_audio.wav"
            audio.write_audiofile(temp_audio)

            # Transcribe entire audio
            logger.info("Transcribing audio (with progress bar)...")

            # Use forced source language for transcription if set
            transcribe_lang = self.forced_source_lang or self.config.get(
                "whisper", {}
            ).get("language", "auto")

            # Anti-repetition parameters for long videos
            anti_rep_config = self.config.get("anti_repetition", {})
            condition_on_prev = anti_rep_config.get("condition_on_previous_text", False)
            repetition_penalty = anti_rep_config.get("repetition_penalty", 1.2)
            no_speech_thresh = anti_rep_config.get("no_speech_threshold", 0.5)
            compression_ratio_thresh = anti_rep_config.get(
                "compression_ratio_threshold", 2.0
            )

            # faster-whisper transcription (optimized for long videos)
            # Try with repetition_penalty first, fallback without it if not supported
            try:
                segments, info = self.whisper_model.transcribe(
                    temp_audio,
                    language=transcribe_lang if transcribe_lang != "auto" else None,
                    beam_size=self.config.get("whisper", {}).get("beam_size", 5),
                    best_of=self.config.get("whisper", {}).get("best_of", 5),
                    temperature=0.0,
                    condition_on_previous_text=condition_on_prev,
                    compression_ratio_threshold=compression_ratio_thresh,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=no_speech_thresh,
                    vad_filter=True,
                    repetition_penalty=repetition_penalty,
                )
            except TypeError:
                logger.warning("repetition_penalty not supported, using fallback")
                segments, info = self.whisper_model.transcribe(
                    temp_audio,
                    language=transcribe_lang if transcribe_lang != "auto" else None,
                    beam_size=self.config.get("whisper", {}).get("beam_size", 5),
                    best_of=self.config.get("whisper", {}).get("best_of", 5),
                    temperature=0.0,
                    condition_on_previous_text=condition_on_prev,
                    compression_ratio_threshold=compression_ratio_thresh,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=no_speech_thresh,
                    vad_filter=True,
                )

            # Convert to standard format
            result = {"text": "", "segments": [], "language": info.language}

            for segment in segments:
                result["segments"].append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                )
                result["text"] += segment.text

            # Create subtitle file
            subtitle_file = "temp_subtitles.srt"
            self._create_srt_file(result, subtitle_file)

            # Add subtitles to video
            logger.info("Adding subtitles to video...")
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_file,
                    "-vf",
                    f"subtitles={subtitle_file}",
                    "-c:a",
                    "copy",
                    output_file,
                    "-y",
                ],
                check=True,
            )

            # Cleanup
            os.remove(temp_audio)
            os.remove(subtitle_file)

            logger.info(f"Video processing completed: {output_file}")
            return True

        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            return False

    def _create_srt_file(
        self, transcription_result: dict, output_file: str, is_export_mode: bool = False
    ) -> None:
        """
        Create SRT subtitle file from transcription result.
        Optimized with batch translation for better performance.
        Includes anti-repetition filtering for long videos.

        Args:
            transcription_result: Whisper transcription result
            output_file: Path to output SRT file
            is_export_mode: If True, use optimized batch size for export_srt (faster)
        """
        segments = transcription_result.get("segments", [])
        if not segments:
            logger.warning("No segments to translate")
            return

        # Anti-repetition: Filter repeated segments (Whisper hallucination)
        original_count = len(segments)
        segments = self.translation_service.detect_segment_repetition(
            segments,
            similarity_threshold=self.config.get("anti_repetition", {}).get(
                "segment_similarity_threshold", 0.85
            ),
        )

        if len(segments) < original_count:
            logger.info(
                f"🔄 Anti-repetition: Filtered {original_count - len(segments)} "
                f"repetitive segments"
            )

        source_lang = self.forced_source_lang or "auto"

        # Extract all texts for batch processing
        texts = [seg["text"].strip() for seg in segments]

        logger.info(f"Translating {len(segments)} segments (batched, sequential)...")

        # Use larger batch size for export mode (faster throughput)
        if is_export_mode:
            batch_size = self.config.get("translation", {}).get("export_batch_size", 64)
            logger.info(f"Using optimized batch size for export: {batch_size}")
        else:
            batch_size = self.config.get("performance", {}).get(
                "translation_batch_size", 32
            )

        translated_segments = [None] * len(texts)

        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batches.append((i, batch_texts))

        with tqdm(total=len(batches), desc="Translating batches", unit="batch") as pbar:
            for batch_idx, batch_texts in batches:
                try:
                    batch_results = self.translation_service.translate_batch(
                        batch_texts, source_language=source_lang
                    )

                    for j, result in enumerate(batch_results):
                        if result and result.translated_text:
                            translated_segments[batch_idx + j] = result.translated_text
                        else:
                            translated_segments[batch_idx + j] = batch_texts[j]

                except Exception as e:
                    logger.warning(f"Batch translation failed: {e}")
                    for j, text in enumerate(batch_texts):
                        translated_segments[batch_idx + j] = text

                pbar.update(1)

        logger.info(
            f"✅ Batched translation completed: {len(translated_segments)} segments"
        )

        # Write to file preserving order
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                start_time = self._format_srt_time(segment["start"])
                end_time = self._format_srt_time(segment["end"])
                text = translated_segments[i]

                f.write(f"{i + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")

    def _format_srt_time(self, seconds: float) -> str:
        """
        Format time for SRT subtitle format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

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

            # Extract audio
            logger.info("Extracting audio...")
            temp_audio = "temp_audio.wav"
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    input_file,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    temp_audio,
                    "-y",
                ],
                check=True,
                capture_output=True,
            )

            # Transcribe audio (with export optimization)
            logger.info("Transcribing audio (optimized for speed)...")
            transcribe_lang = self.forced_source_lang or self.config.get(
                "whisper", {}
            ).get("language", "auto")

            # Use export-optimized parameters for faster transcription
            beam_size = self.config.get("whisper", {}).get("export_beam_size", 1)
            best_of = self.config.get("whisper", {}).get("export_best_of", 1)

            # Anti-repetition parameters for long videos
            anti_rep_config = self.config.get("anti_repetition", {})
            condition_on_prev = anti_rep_config.get("condition_on_previous_text", False)
            repetition_penalty = anti_rep_config.get("repetition_penalty", 1.2)
            no_speech_thresh = anti_rep_config.get("no_speech_threshold", 0.5)
            compression_ratio_thresh = anti_rep_config.get(
                "compression_ratio_threshold", 2.0
            )

            logger.info(
                f"Anti-repetition: condition_on_previous_text={condition_on_prev}, "
                f"repetition_penalty={repetition_penalty}"
            )

            # faster-whisper transcription (optimized for long videos)
            # Try with repetition_penalty first, fallback without it if not supported
            try:
                segments, info = self.whisper_model.transcribe(
                    temp_audio,
                    language=transcribe_lang if transcribe_lang != "auto" else None,
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=0.0,
                    condition_on_previous_text=condition_on_prev,
                    compression_ratio_threshold=compression_ratio_thresh,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=no_speech_thresh,
                    vad_filter=True,
                    repetition_penalty=repetition_penalty,
                )
            except TypeError:
                logger.warning("repetition_penalty not supported, using fallback")
                segments, info = self.whisper_model.transcribe(
                    temp_audio,
                    language=transcribe_lang if transcribe_lang != "auto" else None,
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=0.0,
                    condition_on_previous_text=condition_on_prev,
                    compression_ratio_threshold=compression_ratio_thresh,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=no_speech_thresh,
                    vad_filter=True,
                )

            # Convert to standard format
            result = {"text": "", "segments": [], "language": info.language}

            for segment in segments:
                result["segments"].append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                    }
                )
                result["text"] += segment.text

            # Create subtitle file (with optimized batch size for export)
            logger.info("Creating subtitle file with translations...")
            self._create_srt_file(result, output_srt_path, is_export_mode=True)

            # Cleanup temp audio
            if os.path.exists(temp_audio):
                os.remove(temp_audio)

            logger.info(f"Subtitle file created successfully: {output_srt_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting subtitle file: {e}")
            # Cleanup on error
            if os.path.exists("temp_audio.wav"):
                os.remove("temp_audio.wav")
            return False

    def create_voiceover_video(
        self, input_file: str, srt_file: str, output_file: str
    ) -> bool:
        """
        Create video with Vietnamese TTS voiceover from SRT file.

        This method:
        1. Parses SRT file to get text and timings
        2. Generates Vietnamese TTS audio for each segment using edge-tts
        3. Mutes original video audio
        4. Merges TTS audio with muted video
        5. Burns subtitles into video

        Args:
            input_file: Path to input video file
            srt_file: Path to SRT subtitle file
            output_file: Path to output video file

        Returns:
            True if successful, False otherwise
        """
        try:
            import edge_tts
            import asyncio
            from pydub import AudioSegment
            import tempfile
            import shutil

            logger.info(f"Creating voiceover video from: {input_file}")
            logger.info(f"Using SRT file: {srt_file}")
            logger.info(f"Using TTS voice: {self.tts_voice}")

            # Parse SRT file
            segments = self._parse_srt_file(srt_file)
            if not segments:
                logger.error("No segments found in SRT file")
                return False

            logger.info(f"Found {len(segments)} subtitle segments")

            # Create temp directory for audio files
            temp_dir = tempfile.mkdtemp(prefix="voiceover_")

            try:
                # Get video duration using ffprobe
                duration_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    input_file,
                ]
                result = subprocess.run(duration_cmd, capture_output=True, text=True)
                video_duration = float(result.stdout.strip()) * 1000  # Convert to ms

                # Create a silent base audio track
                logger.info("Creating base audio track...")
                base_audio = AudioSegment.silent(duration=int(video_duration))

                # Define async function to generate TTS using edge-tts
                async def generate_segment(
                    index: int,
                    text: str,
                    output_dir: str,
                    voice: str,
                    semaphore: asyncio.Semaphore,
                ) -> bool:
                    """Generate TTS audio for a single segment with semaphore limit"""
                    async with semaphore:
                        try:
                            output_path = os.path.join(output_dir, f"tts_{index}.mp3")
                            # Use -15% rate natively for standard "reading" speed without robotic artifacts
                            communicate = edge_tts.Communicate(text, voice, rate="-15%")
                            await communicate.save(output_path)
                            return True
                        except Exception as e:
                            logger.warning(
                                f"Failed to generate TTS for segment {index}: {e}"
                            )
                            return False

                async def process_batch_tts(segments, output_dir, voice):
                    """Process all segments in parallel with rate limiting"""
                    # Limit to 5 concurrent connections to avoid issues
                    semaphore = asyncio.Semaphore(5)
                    tasks = []

                    for i, segment in enumerate(segments):
                        text = segment["text"].strip()
                        if text:
                            tasks.append(
                                generate_segment(i, text, output_dir, voice, semaphore)
                            )

                    if tasks:
                        # Use tqdm for generation progress
                        results = []
                        for f in tqdm(
                            asyncio.as_completed(tasks),
                            total=len(tasks),
                            desc="Downloading TTS Audio",
                        ):
                            results.append(await f)
                        return results
                    return []

                # Run parallel generation
                logger.info("Starting parallel TTS generation...")
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        process_batch_tts(segments, temp_dir, self.tts_voice)
                    )
                    loop.close()
                except Exception as e:
                    logger.error(f"Async generation failed: {e}")
                    return False

                # Sequential Mixing Phase
                logger.info("Mixing audio segments...")
                for i, segment in enumerate(tqdm(segments, desc="Processing Audio")):
                    text = segment["text"].strip()
                    start_ms = int(segment["start"] * 1000)
                    end_ms = int(segment["end"] * 1000)

                    if not text:
                        continue

                    try:
                        tts_path = os.path.join(temp_dir, f"tts_{i}.mp3")

                        if not os.path.exists(tts_path):
                            logger.warning(
                                f"TTS file missing for segment {i}, skipping"
                            )
                            continue

                        # Load TTS audio
                        tts_audio = AudioSegment.from_mp3(tts_path)

                        # Add small silence padding for smoother transitions (50ms each side)
                        silence = AudioSegment.silent(duration=50)
                        tts_audio = silence + tts_audio + silence

                        # Calculate available duration for this segment
                        available_duration = end_ms - start_ms
                        current_duration = len(tts_audio)

                        # Determine tempo change
                        # Since we already generated at -15% speed (approx 0.85x), defaults to 1.0
                        # Only speed up if the generated audio is too long for the slot
                        final_speed = 1.0

                        if current_duration > available_duration:
                            # Audio is longer than segment, must speed up
                            # Calculate required speed to fit, max 1.7x (avoid chipmunk voice)
                            required_speed = current_duration / available_duration
                            final_speed = min(required_speed, 1.7)

                        # Apply tempo change if needed (only for speeding up or extreme fitting)
                        if abs(final_speed - 1.0) > 0.01:
                            # Use ffmpeg to change tempo
                            processed_path = os.path.join(
                                temp_dir, f"tts_{i}_processed.mp3"
                            )
                            speed_cmd = [
                                "ffmpeg",
                                "-i",
                                tts_path,
                                "-filter:a",
                                f"atempo={final_speed}",
                                "-y",
                                processed_path,
                            ]
                            subprocess.run(speed_cmd, capture_output=True)
                            if os.path.exists(processed_path):
                                tts_audio = AudioSegment.from_mp3(processed_path)

                        # If fast-forwarded audio is still slightly longer (due to ffmpeg precision), trim it
                        if len(tts_audio) > available_duration:
                            tts_audio = tts_audio[:available_duration]

                        # Overlay TTS audio at the correct position
                        base_audio = base_audio.overlay(tts_audio, position=start_ms)

                    except Exception as e:
                        logger.warning(f"Failed to process segment {i}: {e}")
                        continue

                # Export combined audio
                combined_audio_path = os.path.join(temp_dir, "combined_audio.mp3")
                logger.info("Exporting combined audio...")
                base_audio.export(combined_audio_path, format="mp3")

                # Create video with mixed audio (original + TTS voiceover)
                logger.info("Creating final video with voiceover...")

                # Use ffmpeg to: keep original audio (lower volume) + mix TTS voiceover
                # [0:a]volume=0.3[a0];[1:a]volume=1.0[a1];[a0][a1]amix=inputs=2
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i",
                    input_file,  # Input video with original audio
                    "-i",
                    combined_audio_path,  # TTS voiceover audio
                    "-filter_complex",
                    "[0:a]volume=0.3[original];[1:a]volume=1.2[voiceover];[original][voiceover]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map",
                    "0:v:0",  # Take video from first input
                    "-map",
                    "[aout]",  # Take mixed audio
                    "-c:v",
                    "libx264",  # Video codec
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-y",  # Overwrite output
                    output_file,
                ]

                logger.info(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
                subprocess.run(ffmpeg_cmd, check=True)

                logger.info(f"Voiceover video created successfully: {output_file}")
                return True

            finally:
                # Cleanup temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

        except ImportError as e:
            logger.error(f"Missing required library: {e}")
            logger.error("Please install: pip install edge-tts pydub")
            return False
        except Exception as e:
            logger.error(f"Error creating voiceover video: {e}")
            return False

    def _parse_srt_file(self, srt_path: str) -> list:
        """
        Parse SRT file and return list of segments with timing and text.

        Args:
            srt_path: Path to SRT file

        Returns:
            List of dicts with 'start', 'end', 'text' keys
        """
        import re

        segments = []

        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split by double newlines to get individual subtitle blocks
            blocks = re.split(r"\n\n+", content.strip())

            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) >= 3:
                    # Parse timing line (format: 00:00:00,000 --> 00:00:00,000)
                    timing_match = re.match(
                        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
                        lines[1],
                    )
                    if timing_match:
                        start_time = self._parse_srt_timestamp(timing_match.group(1))
                        end_time = self._parse_srt_timestamp(timing_match.group(2))
                        text = " ".join(lines[2:])

                        segments.append(
                            {"start": start_time, "end": end_time, "text": text}
                        )

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


def prepare_srt_dir(path: str = "srt", clean: bool = True) -> Path:
    """Prepare SRT directory; optionally remove existing contents first."""
    srt_dir = Path(path)
    if clean and srt_dir.exists():
        try:
            shutil.rmtree(srt_dir)
            logger.info("🗑️  Cleaned up old subtitle files")
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not clean srt directory: {e}")

    srt_dir.mkdir(parents=True, exist_ok=True)
    return srt_dir


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
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model size: tiny (fastest), base (recommended), small, medium, large (most accurate). Default: base",
    )
    parser.add_argument(
        "--export-srt",
        action="store_true",
        help="Export subtitle file (.srt) to /srt directory without creating video",
    )
    parser.add_argument(
        "--voiceover",
        action="store_true",
        help="Create video with Vietnamese TTS voiceover (mutes original audio)",
    )
    parser.add_argument(
        "--srt",
        dest="srt_file",
        help="Path to existing SRT file (for voiceover mode, optional - will generate if not provided)",
    )
    parser.add_argument(
        "--voice",
        dest="voice",
        choices=["female", "male"],
        default="female",
        help="TTS voice gender: female (vi-VN-HoaiMyNeural) or male (vi-VN-NamMinhNeural). Default: female",
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
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.export_srt:
            # Export subtitle mode - only create .srt file
            input_file = args.input or args.videopath
            if not input_file:
                parser.print_usage()
                logger.error(
                    "No input video provided. Use --input <file> or place the path after flags."
                )
                sys.exit(2)

            # Prepare `srt` dir: remove old files and recreate
            srt_dir = prepare_srt_dir("srt", clean=True)

            # Generate output .srt filename
            output_srt = srt_dir / f"{Path(input_file).stem}_vietnamese.srt"

            # Set forced source language: support --language flag or legacy --jp flag
            if args.language:
                app.forced_source_lang = args.language
            elif args.jp:
                logger.warning(
                    "⚠️  --jp flag is deprecated. Please use --language ja instead."
                )
                app.forced_source_lang = "ja"
            else:
                app.forced_source_lang = "en"

            # Set forced model size if provided via --model flag
            if args.model:
                app.forced_model_size = args.model
                logger.info(f"Using model size: {args.model}")

            if app.initialize_components():
                success = app.export_srt_only(input_file, str(output_srt))
                if success:
                    logger.info(f"Subtitle file saved to: {output_srt}")
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
        elif args.video:
            # Video processing mode - create video with subtitles
            input_file = args.input or args.videopath
            if not input_file:
                parser.print_usage()
                logger.error(
                    "No input video provided. Use --input <file> or place the path after flags."
                )
                sys.exit(2)

            if not args.output:
                args.output = f"{Path(input_file).stem}_vietnamese_subtitles.mp4"

            # Set forced source language for video: support --language flag or legacy --jp flag
            if args.language:
                app.forced_source_lang = args.language
            elif args.jp:
                logger.warning(
                    "⚠️  --jp flag is deprecated. Please use --language ja instead."
                )
                app.forced_source_lang = "ja"
            else:
                app.forced_source_lang = "en"

            # Set forced model size if provided via --model flag
            if args.model:
                app.forced_model_size = args.model
                logger.info(f"Using model size: {args.model}")

            if app.initialize_components():
                success = app.process_video_file(input_file, args.output)
                sys.exit(0 if success else 1)
            else:
                sys.exit(1)
        elif args.voiceover:
            # Voiceover mode - create video with Vietnamese TTS voiceover
            input_file = args.input or args.videopath
            if not input_file:
                parser.print_usage()
                logger.error(
                    "No input video provided. Use --input <file> or place the path after flags."
                )
                sys.exit(2)

            # Set forced source language
            if args.language:
                app.forced_source_lang = args.language
            elif args.jp:
                logger.warning(
                    "⚠️  --jp flag is deprecated. Please use --language ja instead."
                )
                app.forced_source_lang = "ja"
            else:
                app.forced_source_lang = "en"

            # Set forced model size if provided via --model flag
            if args.model:
                app.forced_model_size = args.model
                logger.info(f"Using model size: {args.model}")

            # Set TTS voice based on --voice argument
            if args.voice == "male":
                app.tts_voice = "vi-VN-NamMinhNeural"
                logger.info("Using male voice: vi-VN-NamMinhNeural")
            else:
                app.tts_voice = "vi-VN-HoaiMyNeural"
                logger.info("Using female voice: vi-VN-HoaiMyNeural")

            # Check if SRT file is provided or needs to be generated
            srt_file = args.srt_file
            if not srt_file:
                # Generate SRT file first
                logger.info("No SRT file provided, generating subtitles first...")

                # Prepare `srt` dir for voiceover: remove and recreate
                srt_dir = prepare_srt_dir("srt", clean=True)
                srt_file = str(srt_dir / f"{Path(input_file).stem}_vietnamese.srt")

                if app.initialize_components():
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
            # Real-time mode
            app.start()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
