"""
Audio Processing Module for Real-time Speech Recognition

This module handles microphone input, audio preprocessing, and audio chunk management
for real-time speech-to-text transcription using faster-whisper.
"""

import pyaudio
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable
import wave
import tempfile
import os
from loguru import logger
import scipy.signal
from collections import deque


class AudioProcessor:
    """Handles real-time audio capture and preprocessing for speech recognition."""

    def __init__(self, config: dict):
        """
        Initialize the AudioProcessor with configuration settings.

        Args:
            config: Audio configuration dictionary
        """
        self.config = config["audio"]
        self.sample_rate = self.config["sample_rate"]
        self.chunk_size = self.config["chunk_size"]
        self.channels = self.config["channels"]
        self.device_index = self.config.get("device_index")

        # Audio processing settings
        self.noise_reduction = self.config.get("noise_reduction", True)
        self.volume_threshold = self.config.get("volume_threshold", 0.01)
        self.silence_timeout = self.config.get("silence_timeout", 2.0)

        # PyAudio instance
        self.pyaudio_instance = None
        self.stream = None

        # Threading and queues
        self.audio_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.is_recording = False
        self.is_paused = False

        # Audio buffer for chunk assembly
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 10))  # 10 seconds max
        self.last_audio_time = time.time()

        # Callbacks
        self.on_audio_chunk: Optional[Callable] = None
        self.on_silence_detected: Optional[Callable] = None

        logger.info("AudioProcessor initialized")

    def initialize_audio(self) -> bool:
        """
        Initialize PyAudio and detect available audio devices.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.pyaudio_instance = pyaudio.PyAudio()

            # List available audio devices
            logger.info("Available audio devices:")
            device_count = self.pyaudio_instance.get_device_count()

            default_device = self.pyaudio_instance.get_default_input_device_info()
            logger.info(f"Default input device: {default_device['name']}")

            for i in range(device_count):
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    logger.info(
                        f"  [{i}] {device_info['name']} - {device_info['maxInputChannels']} channels"
                    )

            # Use specified device or default
            if self.device_index is None:
                self.device_index = default_device["index"]
                logger.info(f"Using default device: {default_device['name']}")
            else:
                device_info = self.pyaudio_instance.get_device_info_by_index(
                    self.device_index
                )
                logger.info(f"Using specified device: {device_info['name']}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            return False

    def start_recording(self) -> bool:
        """
        Start audio recording from microphone.

        Returns:
            True if recording started successfully, False otherwise
        """
        if not self.pyaudio_instance:
            if not self.initialize_audio():
                return False

        try:
            # Create audio stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )

            self.is_recording = True
            self.stream.start_stream()

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._process_audio_chunks, daemon=True
            )
            self.processing_thread.start()

            logger.info("Audio recording started")
            return True

        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            return False

    def stop_recording(self):
        """Stop audio recording and cleanup resources."""
        self.is_recording = False

        if hasattr(self, "stream") and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if (
            hasattr(self, "processing_thread")
            and self.processing_thread
            and self.processing_thread.is_alive()
        ):
            self.processing_thread.join(timeout=2.0)

        logger.info("Audio recording stopped")

    def pause_recording(self):
        """Pause audio processing without stopping the stream."""
        self.is_paused = True
        logger.info("Audio processing paused")

    def resume_recording(self):
        """Resume audio processing."""
        self.is_paused = False
        logger.info("Audio processing resumed")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback function for handling incoming audio data.

        Args:
            in_data: Raw audio data
            frame_count: Number of frames
            time_info: Timing information
            status: Stream status

        Returns:
            Tuple of (None, pyaudio.paContinue)
        """
        if not self.is_paused and self.is_recording:
            try:
                # Convert bytes to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.int16)

                # Add to queue for processing
                if not self.audio_queue.full():
                    self.audio_queue.put((audio_data, time.time()))
                else:
                    logger.warning("Audio queue full, dropping frame")

            except Exception as e:
                logger.error(f"Error in audio callback: {e}")

        return (None, pyaudio.paContinue)

    def _process_audio_chunks(self):
        """Process audio chunks from the queue in a separate thread."""
        while self.is_recording:
            try:
                # Get audio data with timeout
                audio_data, timestamp = self.audio_queue.get(timeout=0.1)

                # Add to buffer
                self.audio_buffer.extend(audio_data)
                self.last_audio_time = timestamp

                # Apply preprocessing
                processed_audio = self._preprocess_audio(audio_data)

                # Check for speech/silence
                volume_level = self._calculate_volume(processed_audio)

                if volume_level > self.volume_threshold:
                    # Speech detected, continue accumulating
                    pass
                else:
                    # Check for silence timeout
                    silence_duration = time.time() - self.last_audio_time
                    if (
                        silence_duration > self.silence_timeout
                        and len(self.audio_buffer) > 0
                    ):
                        # Process accumulated audio chunk
                        self._process_accumulated_audio()

                # Process chunk if buffer is getting full
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                if buffer_duration >= 5.0:  # 5 seconds of audio
                    self._process_accumulated_audio()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")

    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply audio preprocessing techniques.

        Args:
            audio_data: Raw audio data as numpy array

        Returns:
            Preprocessed audio data
        """
        # Convert to float32 for processing
        audio_float = audio_data.astype(np.float32) / 32768.0

        if self.noise_reduction:
            # Simple high-pass filter to reduce low-frequency noise
            nyquist = self.sample_rate / 2
            low_cutoff = 300 / nyquist  # 300 Hz high-pass
            b, a = scipy.signal.butter(4, low_cutoff, btype="high")
            audio_float = scipy.signal.filtfilt(b, a, audio_float)

            # Normalize audio
            max_val = np.max(np.abs(audio_float))
            if max_val > 0:
                audio_float = audio_float / max_val * 0.8  # Prevent clipping

        return audio_float

    def _calculate_volume(self, audio_data: np.ndarray) -> float:
        """
        Calculate the RMS volume of audio data.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            RMS volume level
        """
        return np.sqrt(np.mean(audio_data**2))

    def _process_accumulated_audio(self):
        """Process the accumulated audio buffer and trigger callback."""
        if len(self.audio_buffer) == 0:
            return

        # Convert deque to numpy array
        audio_array = np.array(list(self.audio_buffer))

        # Save to temporary WAV file for Whisper processing
        temp_file = self._save_audio_to_temp_file(audio_array)

        # Trigger callback with audio file path
        if self.on_audio_chunk and temp_file:
            try:
                self.on_audio_chunk(temp_file)
            except Exception as e:
                logger.error(f"Error in audio chunk callback: {e}")
            finally:
                # Cleanup temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass

        # Clear buffer
        self.audio_buffer.clear()

    def _save_audio_to_temp_file(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Save audio data to a temporary WAV file.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Path to temporary WAV file or None if failed
        """
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)

            # Convert float audio back to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Save as WAV file
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            return temp_path

        except Exception as e:
            logger.error(f"Failed to save audio to temp file: {e}")
            return None

    def get_device_list(self) -> list:
        """
        Get list of available audio input devices.

        Returns:
            List of tuples (device_index, device_name)
        """
        if not self.pyaudio_instance:
            self.initialize_audio()

        devices = []
        if self.pyaudio_instance:
            device_count = self.pyaudio_instance.get_device_count()
            for i in range(device_count):
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    devices.append((i, device_info["name"]))

        return devices

    def set_callbacks(
        self, on_audio_chunk: Callable = None, on_silence_detected: Callable = None
    ):
        """
        Set callback functions for audio events.

        Args:
            on_audio_chunk: Callback when audio chunk is ready for processing
            on_silence_detected: Callback when silence is detected
        """
        self.on_audio_chunk = on_audio_chunk
        self.on_silence_detected = on_silence_detected

    def cleanup(self):
        """Cleanup PyAudio resources."""
        self.stop_recording()

        if hasattr(self, "pyaudio_instance") and self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None

        logger.info("AudioProcessor cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def test_audio_processor():
    """Test function for the AudioProcessor class."""
    import yaml

    # Load test configuration
    config = {
        "audio": {
            "sample_rate": 16000,
            "chunk_size": 1024,
            "channels": 1,
            "device_index": None,
            "noise_reduction": True,
            "volume_threshold": 0.01,
            "silence_timeout": 2.0,
        }
    }

    def on_audio_chunk(file_path):
        print(f"Audio chunk ready: {file_path}")
        # Here you would typically pass the file to Whisper for transcription

    # Create and test audio processor
    processor = AudioProcessor(config)
    processor.set_callbacks(on_audio_chunk=on_audio_chunk)

    if processor.start_recording():
        print("Recording started. Speak something...")
        try:
            time.sleep(10)  # Record for 10 seconds
        except KeyboardInterrupt:
            pass
        finally:
            processor.stop_recording()
    else:
        print("Failed to start recording")


if __name__ == "__main__":
    test_audio_processor()
