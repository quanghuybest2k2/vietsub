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
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import yaml


try:
    import whisper

    _HAS_WHISPER = True
except ImportError:
    whisper = None
    _HAS_WHISPER = False
from loguru import logger

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

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
        self.translation_service = None

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
            # Initialize Whisper model (speech-to-text)
            if not _HAS_WHISPER:
                raise RuntimeError(
                    "openai-whisper not installed. Install with 'pip install openai-whisper'."
                )
            logger.info("Loading Whisper model...")
            whisper_config = self.config["whisper"]
            model_size = whisper_config.get("model_size", "base")
            device = whisper_config.get("device", "auto")
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
            self.device = device
            self.whisper_model = whisper.load_model(model_size, device=device)
            logger.info(f"Whisper model '{model_size}' loaded on {device}")

            # Initialize translation service
            logger.info("Initializing translation service...")
            self.translation_service = TranslationService(self.config)

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False

    def _create_srt_file(self, transcription_result: dict, output_file: str) -> None:
        """
        Create SRT subtitle file from transcription result.
        Optimized with batch translation for better performance.

        Args:
            transcription_result: Whisper transcription result
            output_file: Path to output SRT file
        """
        segments = transcription_result.get("segments", [])
        if not segments:
            logger.warning("No segments to translate")
            return

        source_lang = self.forced_source_lang or "auto"

        # Extract all texts for batch processing
        texts = [seg["text"].strip() for seg in segments]

        logger.info(f"Translating {len(segments)} segments in optimized batches...")

        # Use batch translation for much better performance
        try:
            # Batch size optimized for model throughput
            batch_size = 16  # Process 16 segments at a time
            translated_segments = []

            for i in tqdm(
                range(0, len(texts), batch_size),
                desc="Translating batches",
                unit="batch",
            ):
                batch_texts = texts[i : i + batch_size]
                batch_results = self.translation_service.translate_batch(
                    batch_texts, source_language=source_lang
                )

                # Extract translated text or fallback to original
                for j, result in enumerate(batch_results):
                    if result and result.translated_text:
                        translated_segments.append(result.translated_text)
                    else:
                        translated_segments.append(batch_texts[j])

            logger.info(f"Translation completed: {len(translated_segments)} segments")

        except Exception as e:
            logger.error(
                f"Batch translation failed: {e}, falling back to individual translation"
            )
            # Fallback to single translation
            translated_segments = []
            for text in tqdm(texts, desc="Translating (fallback)", unit="seg"):
                try:
                    result = self.translation_service.translate_text(
                        text, source_language=source_lang
                    )
                    if result:
                        translated_segments.append(result.translated_text)
                    else:
                        translated_segments.append(text)
                except Exception as e:
                    logger.warning(f"Translation failed for '{text[:30]}...': {e}")
                    translated_segments.append(text)

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

            # Transcribe audio
            logger.info("Transcribing audio...")
            transcribe_lang = self.forced_source_lang or self.config.get(
                "whisper", {}
            ).get("language", "auto")

            result = self.whisper_model.transcribe(
                temp_audio,
                language=transcribe_lang,
                verbose=False,
                fp16=False if self.device == "cpu" else True,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )

            # Create subtitle file
            logger.info("Creating subtitle file with translations...")
            self._create_srt_file(result, output_srt_path)

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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Offline Vietnamese Subtitle Generator - SRT Export Only (100% Local Processing)"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--input",
        dest="input",
        help="Input video file path (optional if positional provided)",
    )
    parser.add_argument(
        "videopath",
        nargs="?",
        help="Positional video file path",
    )
    parser.add_argument(
        "--jp",
        action="store_true",
        help="Treat source language as Japanese (default: English)",
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
        # Export subtitle mode - only create .srt file
        input_file = args.input or args.videopath
        if not input_file:
            parser.print_usage()
            logger.error(
                "No input video provided. Use --input <file> or place the path after flags."
            )
            sys.exit(2)

        # Clean up old srt directory before processing
        import shutil

        srt_dir = Path("srt")
        if srt_dir.exists():
            try:
                shutil.rmtree(srt_dir)
                logger.info("🗑️  Cleaned up old subtitle files")
            except Exception as e:
                logger.warning(f"⚠️  Warning: Could not clean srt directory: {e}")

        # Create /srt directory
        srt_dir.mkdir(exist_ok=True)

        # Generate output .srt filename
        output_srt = srt_dir / f"{Path(input_file).stem}_vietnamese.srt"

        # Set forced source language: ja if --jp else en
        app.forced_source_lang = "ja" if args.jp else "en"

        if app.initialize_components():
            success = app.export_srt_only(input_file, str(output_srt))
            if success:
                logger.info(f"Subtitle file saved to: {output_srt}")
            sys.exit(0 if success else 1)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
