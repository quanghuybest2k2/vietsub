# Offline Vietnamese Subtitle Generator

100% offline video subtitle generation system powered by OpenAI Whisper and local translation models. No internet connection required after initial setup.

## Features

- **Offline Speech Recognition**: OpenAI Whisper-based accurate transcription (100% local)
- **Offline Translation**: Local neural models (Helsinki-NLP, Argos Translate) - no internet needed
- **Subtitle Export**: Export standalone .srt subtitle files with Vietnamese translation
- **Multi-language Support**: English and Japanese source audio
- **Modern GUI**: Easy-to-use PySide6 interface with progress tracking
- **100% Free & Open Source**: No API keys, subscriptions, or rate limits
- **Complete Privacy**: All processing done locally on your machine

## Demo

![Application Demo](images/demo.png)

## Requirements

- Python 3.12.6
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

**📖 [Complete GUI Usage Guide](GUI_USAGE_GUIDE.md)** - Detailed step-by-step instructions with screenshots

Features:

- Select video file via file browser
- Choose source language (English/Japanese)
- Real-time progress tracking with runtime display
- **Export SRT**: Generates Vietnamese subtitle file (.srt format, saved to Downloads)
- Output automatically saved to Downloads folder with auto-open
- View logs for detailed processing information

### Command Line Mode

**Export subtitle file** (English source, default):

```bash
python main.py --input videos/english.mp4
```

**Export subtitle file** (Japanese source):

```bash
python main.py --jp --input videos/japanese.mp4
```

Full path example:

```bash
python main.py --jp --input "/d/video_recording/test/japanese.mp4"
```

Output: Vietnamese subtitle file (.srt) saved to `/srt` directory

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
├── app_tk.py                # Modern PySide6 GUI interface
├── main.py                  # Command-line entry point
├── config/config.yaml       # Configuration file (offline settings)
├── requirements.txt         # Dependencies
├── src/
│   └── translator.py        # Offline translation services (100% local)
├── tests/
│   └── test_parallel_translation.py
├── srt/                     # Generated subtitle files
├── videos/                  # Sample videos directory
└── logs/                    # Application logs
```

## Dependencies

- `openai-whisper` - Speech recognition (offline)
- `transformers` - Local neural translation models (Helsinki-NLP)
- `argostranslate` - Offline statistical translation
- `PySide6` - Modern GUI framework
- `torch` - Deep learning framework (CPU/CUDA support)
- `pyyaml` - Configuration parsing
- `loguru` - Logging
- `tqdm` - Progress bars
- `ffmpeg` - Audio extraction

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

- **Batch Translation**: Optimized batch processing translates 8-16 segments simultaneously (3-4x faster)
- **Smart Caching**: Automatic caching of translations reduces redundant processing
- **Parallel Processing**: Multi-threaded translation with optimized lock scope for maximum throughput
- **CPU Usage**: Translation workers utilize multiple CPU cores efficiently (default: 4 workers)
- **Memory**: Models require ~2-4GB RAM depending on Whisper model size
- **First Run**: Allow 5-10 minutes for automatic model downloads (one-time setup)
- **Subsequent Runs**: Fully offline with no internet dependency
- **Speed**: Process ~1 minute of video per minute on modern CPUs (base model, 3-4x faster than v1.0)

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
