# Vietnamese Subtitle Generator — Agent Guide

## Project Overview

100% offline video subtitle generation system. Transcribes speech via faster-whisper, translates to Vietnamese using local NLLB/opus-mt/Argos models, and burns subtitles into video. Also supports Vietnamese TTS voiceover via VieNeu.

**Tech stack:** Python 3.12, PySide6 (GUI), faster-whisper, HuggingFace Transformers, Argos Translate, moviepy, OpenCV, loguru.

## ⚠️ MANDATORY RULE — Venv Required

> **Every** Python command **MUST** be run inside the `venv` virtual environment. NEVER use the global/system Python interpreter.
> Running packages globally makes the machine heavy and causes dependency conflicts.

### How to activate venv

```bash
# Git Bash / Linux
source venv/Scripts/activate

# Or if the above doesn't exist:
source venv/bin/activate
```

After activation, you'll see `(venv)` in the terminal prompt. Verify:

```bash
which python    # should show venv/Scripts/python
pip list        # should show packages from venv only
```

If `venv` doesn't exist yet, create it first:

```bash
python -m venv venv
```

> **Agent instruction:** Always check if venv is active before running any Python-related command. If not active, run `source venv/Scripts/activate` first.

## Quick Start

```bash
source venv/Scripts/activate   # activate venv first!
pip install -r requirements.txt
python app_tk.py               # GUI mode (recommended)
python main.py --video --input videos/example.mp4  # CLI mode
```

**Prerequisites:** Python 3.12.6, FFmpeg (in PATH), espeak-ng (optional for TTS).

## Architecture

```
app_tk.py (PySide6 GUI) ──spawns subprocess──► main.py (CLI + core logic)
                                                    │
                                                    ├── src/audio_processor.py   (PyAudio mic capture)
                                                    ├── src/translator.py        (TranslationService + TranslationCache)
                                                    └── src/subtitle_overlay.py  (OpenCV real-time overlay)
```

**Key design decision:** The GUI runs `main.py` as a **subprocess** (not importing it directly). Communication is via `queue.Queue` + `threading.Event` for pause/resume/stop.

## CLI Flags (`main.py`)

| Flag                                        | Purpose                                                |
| ------------------------------------------- | ------------------------------------------------------ |
| `--video --input FILE --output FILE`        | Full pipeline: transcribe → translate → burn subtitles |
| `--export-srt --input FILE`                 | Export .srt subtitle file only                         |
| `--voiceover --input FILE`                  | Create video with Vietnamese TTS voiceover             |
| `--language en\|ja\|zh\|ko\|th\|id`         | Source language (default: en)                          |
| `--model tiny\|base\|small\|medium\|large`  | Whisper model size                                     |
| `--voice Binh\|Tuyen\|Vinh\|Doan\|Ly\|Ngoc` | VieNeu TTS voice                                       |

## Translation Pipeline (fallback chain)

1. **NLLB-200-distilled-600M** (primary, ~600MB model, best quality)
2. **opus-mt** (HuggingFace pipeline, 512 tokens)
3. **Argos Translate** (offline C++ backend, last resort)

All translation methods are thread-safe via `_model_lock`. Results cached in `TranslationCache` (LRU, 1000 entries).

## Anti-Repetition System

Prevents Whisper hallucination in long videos:

- `repetition_penalty: 1.2` — reduces duplicate text
- `no_speech_threshold: 0.5` — skips non-speech segments
- `detect_segment_repetition()` — Jaccard similarity filter (threshold 0.85), max 2 consecutive repeats

## Key Conventions

- **Logging:** `loguru` throughout — use `logger.info()`, `logger.error()`, etc.
- **Config:** `config/config.yaml` accessed via `self.config["section"]["key"]`
- **Type hints:** Required for all function signatures (PEP 484)
- **Docstrings:** Google-style with `Args:` / `Returns:`
- **Naming:** `snake_case` for functions/vars, `PascalCase` for classes
- **Error handling:** `try/except` at every integration point with graceful fallbacks
- **Async translation:** Use `translate_async()` with callbacks for non-blocking operation

## Project Structure

```
vietsub/
├── main.py              # CLI + orchestrator (~900 lines)
├── app_tk.py            # PySide6 GUI (~2900 lines)
├── config/config.yaml   # Central configuration
├── lang/en.json, vi.json # UI localization strings
├── src/
│   ├── audio_processor.py  # Real-time mic capture
│   ├── translator.py       # Translation engine + cache
│   └── subtitle_overlay.py # OpenCV overlay rendering
└── srt/                # Generated subtitle output directory
```

## Important Pitfalls

1. **NLLB model** downloads ~600MB on first run — needs disk space and initial internet despite "offline" label.
2. **VieNeu TTS** depends on a custom third-party PyPI index (`pnnbao97.github.io`). If unavailable, voiceover mode breaks.
3. **Concurrent temp files:** `temp_audio.wav` and `temp_subtitles.srt` used without process-specific naming.
4. **ffmpeg** must be in PATH — no graceful error if missing.
5. **`keyboard` library** needs admin/sudo on Linux for hotkeys.
6. **GUI subprocess** encoding issues possible — the GUI sets `PYTHONIOENCODING=utf-8` and uses `errors="replace"`.
7. **Beam size** differs by mode: `beam_size=1` for export (speed), `beam_size=5` for video (quality).

## Available Chat Commands

| Command        | Description                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------- |
| `/ai-commit`   | Auto-generate a Conventional Commit message from staged changes and commit immediately (no confirmation). |
| `/code-review` | Review staged git changes with severity ratings (critical/high/medium/low) and fix suggestions.           |

## Performance Config Tips

- **Export mode:** `beam_size: 1, best_of: 1, batch_size: 64` for speed
- **Video mode:** `beam_size: 5, best_of: 5, batch_size: 32` for quality
- **GPU:** Auto-detected via `torch.cuda.is_available()`. Falls back to CPU.
- **Workers:** Configurable via `config.yaml` → `performance` section (translation_workers, max_workers, etc.)
