# Offline Vietnamese Subtitle Generator

100% offline video subtitle generation system powered by OpenAI Whisper and local translation models. No internet connection required after initial setup.

## Features

- **Offline Speech Recognition**: OpenAI Whisper-based accurate transcription (100% local)
- **Advanced Translation**: NLLB-200-distilled-600M for high-quality translation (max 1024 tokens)
- **Auto Chunk Processing**: Automatically splits long texts to handle videos >3 hours
- **Smart Fallback**: Multi-layer fallback (NLLB → opus-mt → Argos) ensures reliability
- **Video Processing**: Add Vietnamese subtitles to video files with timing synchronization
- **Voice-Over Generation**: Create videos with natural Vietnamese voice-over (TTS) from subtitles
- **Subtitle Export**: Export standalone .srt subtitle files without creating video
- **Multi-language Support**: EN, JA, ZH, KO, TH, ID → VI (direct translation)
- **Modern GUI**: Easy-to-use PyQt6 interface with progress tracking and pause/resume
- **100% Free & Open Source**: No API keys, subscriptions, or rate limits
- **Complete Privacy**: All processing done locally on your machine

## Demo

![Application Demo](images/demo.png)

## Requirements

- Python 3.12.6 ([download here](https://www.python.org/downloads/release/python-3126/) - Windows installer (64-bit))
- FFmpeg ([download here](https://www.gyan.dev/ffmpeg/builds/) - ffmpeg-git-essentials.7z)
- No internet connection required (after installing dependencies)

## Installation

1. **Clone and navigate to project directory**

   ```bash
   cd vietsub
   python -m venv venv
   ```

2. **Activate virtual environment**

   ```bash
   source venv/Scripts/activate
   ```

   To deactivate:

   ```bash
   deactivate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### GUI Mode (Recommended)

Launch the modern PyQt6 interface with drag-and-drop support:

```bash
python app_tk.py
```

or

```bash
./run.sh
```

Features:

- Select video file via file browser or drag-and-drop
- Choose source language: **English, Japanese, Chinese (Simplified), Korean, Thai, Indonesian**
- Real-time progress tracking with runtime display
- Pause/Resume/Stop/Reset controls
- **Two processing modes:**
  - **Start Processing**: Creates video with embedded subtitles (saved to Downloads)
  - **Export SRT**: Generates subtitle file only (.srt format, saved to Downloads)
- Output automatically saved to Downloads folder with auto-open
- View logs and download SRT file

### Command Line Mode

**Create video with subtitles** (English source, default):

```bash
python main.py --video --input videos/english.mp4 --output videos/output_with_subtitles_en.mp4
# With custom model: add --model {tiny,base,small,medium,large}
python main.py --video --model base --input videos/english.mp4 --output videos/output_with_subtitles_en.mp4
```

**Create video with subtitles** (other languages):

```bash
# Japanese
python main.py --video --language ja --input videos/japanese.mp4 --output videos/output_ja.mp4

# Chinese (Simplified)
python main.py --video --language zh --input videos/chinese.mp4 --output videos/output_zh.mp4

# Korean
python main.py --video --language ko --input videos/korean.mp4 --output videos/output_ko.mp4

# Thai
python main.py --video --language th --input videos/thai.mp4 --output videos/output_th.mp4

# Indonesian
python main.py --video --language id --input videos/indonesian.mp4 --output videos/output_id.mp4

# With custom model (add to any command above):
python main.py --video --model small --language ja --input videos/japanese.mp4 --output videos/output_ja.mp4
```

**Legacy flag (still supported):**

```bash
# Old way (deprecated but still works)
python main.py --video --jp --input videos/japanese.mp4 --output videos/output.mp4
```

**Export subtitle file only** (.srt format, saved to /srt directory):

```bash
# English (default)
python main.py --export-srt --input videos/english.mp4

# Other languages
python main.py --export-srt --language ja --input videos/japanese.mp4
python main.py --export-srt --language zh --input videos/chinese.mp4
python main.py --export-srt --language ko --input videos/korean.mp4

# With custom model (add --model {tiny,base,small,medium,large}):
python main.py --export-srt --model base --language ja --input videos/japanese.mp4
```

**Generate Video with Voice-Over** (Vietnamese TTS):

```bash
# Basic usage (auto-generates subtitles first)
python main.py --voiceover --input videos/english.mp4

# With existing SRT file
python main.py --voiceover --input videos/english.mp4 --srt subtitles.srt

# Custom output path
python main.py --voiceover --input videos/english.mp4 --output videos/voiceover.mp4
```

**Available language codes:**

- `en` - English (default)
- `ja` - Japanese (日本語)
- `zh` - Chinese Simplified (中文简体)
- `ko` - Korean (한국어)
- `th` - Thai (ไทย)
- `id` - Indonesian (Bahasa Indonesia)

**Available model sizes** (optional, default: base):

- `tiny` - Fastest (~0.5 min/hour), lowest accuracy
- `base` - Recommended (~1 min/hour), good balance
- `small` - Better accuracy (~2 min/hour)
- `medium` - High quality (~5 min/hour)
- `large` - Best accuracy (~10 min/hour), slowest

Full path example:

```bash
python main.py --video --language ja --input "/d/video_recording/test/japanese.mp4" --output "videos/output.mp4"
```

Note: All processing is done 100% offline. Internet is not required after dependencies are installed.

## Configuration

Edit `config/config.yaml` to customize:

- **Whisper model size**: tiny, base, small, medium, large (affects accuracy vs speed)
- **Translation workers**: Number of parallel threads (default: 4, optimal for most CPUs)
  - 4 workers: Best for 4-8 core CPUs (recommended)
  - 6-8 workers: For 8+ core CPUs with 16GB+ RAM
  - Batch processing automatically optimizes throughput
- **Cache size**: Translation cache size (default: 200, increased for batch processing)
- **Subtitle appearance**: Font, size, color, position, background opacity
- **Video output**: Codec, bitrate, FPS settings
- All settings optimized for offline processing with batch translation

## Project Structure

```
vietsub/
├── app_tk.py                # Modern PyQt6 GUI interface
├── main.py                  # Command-line entry point
├── config/config.yaml       # Configuration file (offline settings)
├── requirements.txt         # Dependencies
├── src/
│   ├── audio_processor.py   # Audio extraction & preprocessing
│   ├── subtitle_overlay.py  # Subtitle rendering & video processing
│   └── translator.py        # Offline translation services (100% local)
├── tests/
│   └── test_parallel_translation.py
├── videos/                  # Sample videos directory
└── logs/                    # Application logs
```

## Dependencies

- `openai-whisper` - Speech recognition (offline)
- `transformers` - Local neural translation models (Helsinki-NLP)
- `argostranslate` - Offline statistical translation
- `PyQt6` - Modern GUI framework
- `opencv-python` - Video processing
- `torch` - Deep learning framework (CPU/CUDA support)
- `pyyaml` - Configuration parsing
- `loguru` - Logging
- `tqdm` - Progress bars
- `ffmpeg-python` - Video/audio manipulation

## Troubleshooting

**First-time setup slow**: Local translation models will be downloaded automatically on first run (one-time only)

**Translation quality issues**:

- Install Argos Translate language packages: `argostranslate-package-updater`
- Models are downloaded automatically to `~/.cache/huggingface/` and `~/.local/share/argos-translate/`

**Performance issues**:

- Use smaller Whisper model (`tiny` or `base`) in `config/config.yaml` for faster processing
- Adjust `translation_workers` (default: 4) based on your CPU cores
- Recommended: 4 workers for 4-8 cores, 6-8 workers for 8+ cores

**Processing paused/stuck**:

- Use Stop button to cancel and Reset to start over
- Check logs for detailed error messages

**FFmpeg errors**:

- Ensure FFmpeg is installed and in system PATH
- Download from: https://www.gyan.dev/ffmpeg/builds/

## Performance Tips

- **NLLB Model**: Uses facebook/nllb-200-distilled-600M for high-quality translation with 1024 token support
- **Batch Translation**: Optimized batch processing with size 8 for stable long-video handling
- **Auto Chunking**: Automatically splits long texts (>1024 tokens) for videos >3 hours
- **Smart Caching**: Automatic caching of translations reduces redundant processing
- **Parallel Processing**: Multi-threaded translation with optimized lock scope for maximum throughput
- **CPU Usage**: Translation workers utilize multiple CPU cores efficiently (default: 4 workers)
- **Memory**: Models require ~2-4GB RAM (NLLB: ~1.2GB, Whisper varies by size)
- **First Run**: Allow 5-10 minutes for automatic model downloads (one-time setup)
- **Subsequent Runs**: Fully offline with no internet dependency
- **Speed**: Process ~1 minute of video per minute on modern CPUs (base Whisper model)

## Privacy & Offline Benefits

✅ **100% Local Processing** - No data sent to external servers  
✅ **No Rate Limits** - Process unlimited videos without restrictions  
✅ **No API Costs** - Completely free to use  
✅ **Works Offline** - Process videos anywhere without internet  
✅ **Private** - Your video content never leaves your machine

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Maintainer**: quanghuybest2k2  
**Repository**: [vietsub](https://github.com/quanghuybest2k2/vietsub)
