# GUI Usage Guide

Complete guide for using the Vietnamese Subtitle Generator GUI application.

## Launching the Application

Start the graphical interface:

```bash
python app_tk.py
```

## Main Interface Overview

The application features a modern, user-friendly interface with the following sections:

### 1. **Title Bar**

- Application title: "🎬 Vietnamese Subtitle Generator"
- Window controls: Minimize, Maximize/Restore, Close

### 2. **Video Selection Section**

- **File path input**: Displays selected video file path
- **Browse button**: Opens file picker to select video
- **Supported formats**: MP4, MKV, MOV, AVI

### 3. **Language Selection**

- **Dropdown menu**: Choose source language
  - English (default)
  - Japanese

### 4. **Control Buttons**

#### Start Processing Button (Green)

- **Purpose**: Creates video with embedded Vietnamese subtitles
- **Output**: Video file saved to Downloads folder
- **Process**: Extract audio → Transcribe → Translate → Embed subtitles into video

#### Export SRT Button (Purple)

- **Purpose**: Generates only the subtitle file (.srt format)
- **Output**: Subtitle file saved to Downloads folder
- **Process**: Extract audio → Transcribe → Translate → Save as .srt
- **Benefit**: Faster than full video processing, useful for manual editing or use with media players

#### Open Logs Button

- Opens the application log file for detailed debugging information

#### Reset Button

- Clears all fields and resets the interface to initial state
- Only available when no processing is running

### 5. **Processing Logs Panel**

- Real-time display of processing progress
- Shows transcription and translation progress
- Displays errors and warnings
- Dark theme for easy reading

### 6. **Status Bar**

- **Left**: Current status message
- **Right**: Runtime counter (shows elapsed time during processing)
- **Bottom-right**: Resize grip for window resizing

## Step-by-Step Usage

### Option A: Create Video with Subtitles

1. **Launch the application**

   ```bash
   python app_tk.py
   ```

2. **Select your video file**

   - Click the "Browse" button
   - Navigate to your video file
   - Or paste the file path directly into the input field

3. **Choose source language**

   - Select "English" for English audio (default)
   - Select "Japanese" for Japanese audio

4. **Start processing**

   - Click the green "▶ Start Processing" button
   - Monitor progress in the logs panel
   - Runtime counter shows elapsed time

5. **Wait for completion**

   - Processing time varies based on video length and hardware
   - Typical: ~1 minute of processing per 1 minute of video
   - The application will show "✅ Processing completed successfully!" when done

6. **Access your video**
   - Output video is automatically saved to your Downloads folder
   - Filename format: `{original_name}_vietnamese_subtitles.mp4`
   - Downloads folder opens automatically
   - Video includes embedded Vietnamese subtitles

### Option B: Export Subtitle File Only

1. **Launch and select video** (same as steps 1-3 above)

2. **Export subtitle**

   - Click the purple "💾 Export SRT" button instead
   - This is faster than creating a full video

3. **Wait for completion**

   - Processing extracts audio, transcribes, and translates
   - No video encoding required (faster process)

4. **Access your subtitle file**
   - File saved to your Downloads folder
   - Filename format: `{original_name}_vietnamese.srt`
   - Downloads folder opens automatically
   - Use with any media player (VLC, MPC-HC, etc.)

## Processing Controls

### Pause/Resume

- While processing, the Start button changes to "■ Stop"
- Click to pause the current operation
- Button changes to "▶ Continue"
- Click again to resume processing
- Elapsed time is preserved

### Stop Processing

- Click the Stop button to cancel the current operation
- Process will terminate gracefully
- Partial output files may be created

### Reset Interface

- Click "🔄 Reset" button to clear everything
- Only available when not processing
- Clears video path, logs, and status

## Understanding the Logs

The logs panel shows detailed information:

### Normal Processing Messages

```
Starting Vietnamese Subtitle Generation...
Loading Whisper model...
Extracting audio...
Transcribing audio...
Translating X segments in optimized batches...
✅ Processing completed successfully!
```

### Progress Indicators

- Progress bars show transcription and translation progress
- Percentage completion for each stage
- Real-time status updates

### Error Messages

- Red ❌ icon indicates errors
- Check logs for detailed error information
- Common issues and solutions provided

## Keyboard Shortcuts

Currently, the GUI uses mouse-only controls. Keyboard shortcuts coming in future updates.

## Advanced Features

### Custom Output Location

- Currently outputs to Downloads folder
- To change: modify the code in `app_tk.py` worker function
- Or use command-line mode for custom output paths

### Configuration

- Edit `config/config.yaml` to adjust processing parameters
- See main README.md for configuration details
- Restart application after changing config

## Privacy & Security

- **100% Offline**: All processing done locally on your computer
- **No data transmission**: Your videos never leave your machine
- **No tracking**: No analytics or telemetry collected
- **Open source**: Full transparency of code operations

## Getting Help

If you encounter issues:

1. Check the logs panel for error details
2. Review this guide and main README.md
3. Check `logs/subtitle_generator.log` for detailed logs
4. Open an issue on GitHub with error details
5. Include relevant log excerpts when reporting issues

---

**Version**: 2.0  
**Last Updated**: December 6, 2025  
**Related**: See [EXPORT_SRT_GUIDE.md](EXPORT_SRT_GUIDE.md) for Vietnamese guide
