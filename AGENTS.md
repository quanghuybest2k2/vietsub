# Vietnamese Subtitle Generator — Agent Guide

## Project Overview

100% offline video subtitle generation system. Transcribes speech via faster-whisper, translates to Vietnamese using local NLLB/opus-mt/Argos models, and burns subtitles into video. Also supports Vietnamese TTS voiceover via VieNeu.

**Tech stack:** Python 3.12, PySide6 (GUI), faster-whisper + CTranslate2, HuggingFace Transformers, Argos Translate, PyAV/imageio-ffmpeg, OpenCV, loguru.

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

## ⚠️ RULE — Sync requirements.txt After Any pip Change

> After every `pip install` or `pip uninstall` inside the venv, **must** update `requirements.txt` immediately.
> Only list direct dependencies — do NOT dump the entire `pip freeze`.
> If a package requires a special `--extra-index-url` (torch CUDA, vieneu), include the index URL next to it.

## Quick Start

```bash
source venv/Scripts/activate   # activate venv first!
pip install -r requirements.txt
python app_tk.py               # GUI mode (recommended)
python main.py --video --input videos/example.mp4  # CLI mode
```

**Prerequisites:** Python 3.12.6, FFmpeg (PATH, or the imageio-ffmpeg bundled binary as fallback), espeak-ng (optional for TTS).

## Architecture

```
app_tk.py (PySide6 GUI) ──spawns subprocess──► main.py (CLI + core logic)
                                                    │
                                                    ├── src/audio_processor.py   (PyAudio mic capture)
                                                    ├── src/translator.py        (TranslationService + TranslationCache)
                                                    └── src/subtitle_overlay.py  (OpenCV real-time overlay)
```

**Key design decision:** The GUI runs `main.py` as a **subprocess** (not importing it directly), with `cwd` = project dir, `stdin=DEVNULL` and (POSIX) its own process group. Output reaches the UI via `queue.Queue`; lines ending in `\r` are progress-bar refreshes that replace the previous log line. Pause/resume really suspends the job (`SIGSTOP`/`SIGCONT` on the process group; on Windows `NtSuspendProcess` suspends main.py only, so a running ffmpeg keeps going); stop terminates the whole tree (process group on POSIX, `taskkill /T` on Windows).

**Model lifecycle (file modes):** Whisper loads first, is released after transcription, then NLLB loads (lazy `get_translation_service()`); both are released before VieNeu TTS. Realtime-only modules (`audio_processor`, `subtitle_overlay`, `keyboard`) are imported lazily, so file modes don't need PyAudio/OpenCV.

## CLI Flags (`main.py`)

| Flag                                                             | Purpose                                                |
| ---------------------------------------------------------------- | ------------------------------------------------------ |
| `--video --input FILE --output FILE`                             | Full pipeline: transcribe → translate → burn subtitles |
| `--export-srt --input FILE`                                      | Export .srt subtitle file only                         |
| `--voiceover --input FILE`                                       | Create video with Vietnamese TTS voiceover             |
| `--language en\|ja\|zh\|ko\|th\|id`                              | Source language (default: en)                          |
| `--model tiny\|base\|small\|medium\|large\|turbo`                | Whisper model size (`turbo` = large-v3-turbo)          |
| `--device auto\|cuda\|cpu`                                       | Inference device (default: config, `auto`)             |
| `--voice Ngọc Lan\|Ngọc Linh\|Trúc Ly\|Mỹ Duyên\|Xuân Vĩnh\|...` | VieNeu v3 Turbo TTS voice (default: Ngọc Lan)          |

## Translation Pipeline (fallback chain)

1. **NLLB-200-distilled-600M via CTranslate2** (primary). `models/nllb-200-distilled-600M-ct2-int8` is converted once from the official checkpoint (needs torch); int8 on CPU/Pascal, int8_float16/float16 on newer GPUs. 4-7x faster than Transformers fp32 on CPU, same chrF on FLORES-200.
2. **NLLB-200-distilled-600M via Transformers** (GPU fp16 on Volta+, fp32 otherwise; CPU fp32)
3. **opus-mt** (HuggingFace pipeline, EN→VI and JA→EN→VI)
4. **Argos Translate** (set up lazily, last resort)

Batches are sorted by length (`translation.batch_size`, default 32), beam size `translation.beam_size` (default 4), output capped at `3 × source tokens + 16`. Empty/non-speech lines (`[Music]`, `♪`) are never sent to the model. The "already Vietnamese" heuristic only runs when the source language is `auto`. All inference is serialized via `_model_lock` (HF fast tokenizers are not thread-safe). Results cached in `TranslationCache` (LRU, 1000 entries).

**Offline loading:** always load Hugging Face models by their **local snapshot path** (`_hf_from_pretrained`). transformers 4.57.3 calls the Hub API (`model_info`) when a tokenizer is loaded by repo id and fails offline even if cached.

## Anti-Repetition System

Prevents Whisper hallucination in long videos:

- `repetition_penalty: 1.0` (off) — 1.2 penalised legitimately repeated words ("say" → "see", "done" → "down") and lowered confidence
- `condition_on_previous_text: false`, VAD filter, `no_speech_threshold: 0.5`
- `compression_ratio_threshold` only matters when `whisper.temperature` is a list (fallback enabled); with a single temperature it is inert
- `detect_segment_repetition()` — Jaccard similarity filter (words, or character bigrams for CJK; threshold 0.85), max `max_allowed_repetitions` consecutive repeats
- `_merge_fragments()` — joins a segment cut mid-sentence with the next one (config `srt:`) so NLLB translates whole sentences

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
├── main.py              # CLI + orchestrator
├── app_tk.py            # PySide6 GUI
├── AGENTS.md            # Agent instructions (this file)
├── run.sh               # Convenience launcher script
├── config/
│   ├── config.yaml      # Central configuration
│   └── user_preferences.json  # User language/theme prefs
├── images/              # Screenshots & demo images
├── lang/                # UI localization (en.json, vi.json)
├── src/
│   ├── audio_processor.py  # Real-time mic capture
│   ├── translator.py       # Translation engine + cache
│   └── subtitle_overlay.py # OpenCV overlay rendering
├── srt/                 # Generated subtitle output directory
├── videos/              # Sample videos directory
└── logs/                # Application logs
```

## Important Pitfalls

1. **NLLB model** downloads ~2.5GB (`pytorch_model.bin`) on first run, then is converted to CTranslate2 int8 (~620MB in `models/`, needs torch + ~3GB RAM) — needs disk space and initial internet despite "offline" label.
2. **VieNeu TTS v3 Turbo** — auto-downloads model from HuggingFace on first use (~2GB). Needs internet for initial setup. Runs on GPU (PyTorch) or CPU (ONNX) automatically.
3. **GPU compute types:** never hard-code `float16`. Pascal GPUs (GTX 10xx, Quadro Pxxx, CC 6.x) reject it in CTranslate2. `resolve_compute_candidates()` (src/translator.py) asks CTranslate2 what the device supports; CPU uses int8.
4. **CUDA libraries load lazily:** CTranslate2 dlopens cuBLAS on the first inference, not at model load. Both loaders run a warm-up decode on CUDA and fall back to CPU on failure. Importing torch first (`_preload_torch_cuda`) exposes the cuBLAS shipped with torch.
5. **ffmpeg** from PATH, else the `imageio-ffmpeg` bundled binary (`get_ffmpeg_exe()`); checked before transcription. Transcription itself decodes media with PyAV (no ffmpeg needed). NVENC is probed with a 1-frame encode (needs a matching driver) and falls back to libx264.
6. **Temp files** live in per-run `tempfile.mkdtemp()` folders; `srt/` is no longer wiped, so concurrent runs don't clobber each other.
7. **`keyboard` library** needs admin/sudo on Linux for hotkeys.
8. **GUI subprocess** encoding issues possible — the GUI sets `PYTHONIOENCODING=utf-8` and uses `errors="replace"`.
9. **Beam size:** `beam_size` (`--video`), `export_beam_size` (export/voiceover) and `realtime_beam_size` all default to 1. Beam 5 was tested: better on some names, but longer cues with whole-second timestamps (up to ~2.5 s off) on the samples and ~1.5x slower.
10. **Exit codes:** `main.py` exits `128 + signal` when interrupted — the GUI relies on non-zero codes to not report a cancelled job as success.

## Available Chat Commands

| Command        | Description                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------- |
| `/ai-commit`   | Auto-generate a Conventional Commit message from staged changes and commit immediately (no confirmation). |
| `/code-review` | Review staged git changes with severity ratings (critical/high/medium/low) and fix suggestions.           |

## Performance Config Tips

- **Beam size:** greedy (1) everywhere by default; int8 greedy is ~1.9-2x faster than the old float32 greedy on CPU
- **GPU:** Auto-detected via `ctranslate2.get_cuda_device_count()` (no torch needed); compute type via `get_supported_compute_types()`; CPU fallback after a failed warm-up
- **CPU:** int8 is 2-2.4x faster than float32 for Whisper base; `cpu_threads: 4` is as fast as 8 for base
- **Translation:** `translation.beam_size: 4` (quality) or 2 (≈1.7x faster, -1.3 chrF on JA→VI); `batch_size: 32`
- **Burn-in:** `video.preset: veryfast` (≈1.4x faster than `medium`, SSIM 0.989 vs 0.992); `video.encoder: auto` uses NVENC when the probe succeeds
