import threading
import queue
import traceback
import subprocess
import signal
import sys
import os
import io
import re
import tempfile
from pathlib import Path
import time
import json
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QFrame,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QSizeGrip,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QFont, QTextCursor
import shutil
import qtawesome as qta


# Load translations from JSON files
def load_translations():
    """Load translation files from lang directory"""
    translations = {}
    lang_dir = Path(__file__).parent / "lang"

    for lang_file in ["en.json", "vi.json"]:
        lang_code = lang_file.replace(".json", "")
        file_path = lang_dir / lang_file

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                translations[lang_code] = json.load(f)
        except Exception as e:
            print(f"Error loading {lang_file}: {e}")
            translations[lang_code] = {}

    return translations


def load_user_preferences():
    """Load user preferences from config file"""
    config_dir = Path(__file__).parent / "config"
    config_file = config_dir / "user_preferences.json"

    default_prefs = {"language": "en", "theme": "light"}

    try:
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading preferences: {e}")

    return default_prefs


def save_user_preferences(language, theme):
    """Save user preferences to config file"""
    config_dir = Path(__file__).parent / "config"
    config_file = config_dir / "user_preferences.json"

    try:
        config_dir.mkdir(exist_ok=True)
        preferences = {"language": language, "theme": theme}

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(preferences, f, indent=2)
    except Exception as e:
        print(f"Error saving preferences: {e}")


TRANSLATIONS = load_translations()


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = str(PROJECT_DIR / "main.py")
SRT_DIR = PROJECT_DIR / "srt"
# Combo index -> --model value (new entries are appended so saved indices keep their meaning)
MODEL_LIST = ["base", "tiny", "small", "medium", "large", "turbo"]
LANG_MAP = {
    "English": "en",
    "Japanese": "ja",
    "Chinese (Simplified)": "zh",
    "Korean": "ko",
    "Thai": "th",
    "Indonesian": "id",
}
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _srt_path_for(input_path: str) -> Path:
    """SRT file main.py writes for a given video."""
    return SRT_DIR / f"{Path(input_path).stem}_vietnamese.srt"


def _suspend_process(process, suspend: bool) -> None:
    """Really pause/resume a job (stopping the output reader alone let it keep running).

    POSIX: SIGSTOP/SIGCONT to the job's process group, so ffmpeg children stop too.
    Windows: NtSuspendProcess/NtResumeProcess on main.py itself.
    """
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_SUSPEND_RESUME = 0x0800
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_SUSPEND_RESUME, False, process.pid
            )
            if handle:
                try:
                    if suspend:
                        ctypes.windll.ntdll.NtSuspendProcess(handle)
                    else:
                        ctypes.windll.ntdll.NtResumeProcess(handle)
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
        else:
            os.killpg(
                os.getpgid(process.pid), signal.SIGSTOP if suspend else signal.SIGCONT
            )
    except Exception as e:
        print(f"Could not {'suspend' if suspend else 'resume'} process: {e}")


def _terminate_process(process, timeout: float = 3.0) -> None:
    """Terminate a job and its children (ffmpeg), even while it is suspended."""
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            # Kill the whole tree: TerminateProcess on main.py alone left its
            # ffmpeg child encoding on in the background
            result = subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
            )
            if result.returncode != 0:
                process.terminate()
        else:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
            # A stopped process only acts on SIGTERM after SIGCONT
            os.killpg(pgid, signal.SIGCONT)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=timeout)
        except Exception:
            pass
    except (ProcessLookupError, OSError):
        pass


def _build_command(mode_flag: str, input_path: str, source_lang: str, model_size: str) -> list:
    cmd = [sys.executable, MAIN_SCRIPT, mode_flag, "--input", input_path]

    # Add language flag for non-English languages
    if source_lang and source_lang != "en":
        cmd.extend(["--language", source_lang])

    # Add model size flag if specified
    if model_size:
        cmd.extend(["--model", model_size])
    return cmd


def _run_main_subprocess(
    cmd: list,
    msg_queue: queue.Queue,
    pause_event: threading.Event = None,
    current_process_holder: list = None,
    stop_event: threading.Event = None,
):
    """Run main.py and stream its output into the GUI queue.

    Returns:
        main.py exit code, or None if the user cancelled the job
    """
    msg_queue.put(f"Command: {' '.join(cmd)}\n")
    msg_queue.put("=" * 80 + "\n")

    # Set environment to force UTF-8 encoding for subprocess
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # Per-job temp root owned by the GUI: removed below even when the job is
    # killed before main.py could clean its own temp folders
    job_tmp = tempfile.mkdtemp(prefix="vietsub_job_")
    for name in ("TMPDIR", "TEMP", "TMP"):
        env[name] = job_tmp

    group_kwargs = {}
    if os.name != "nt":
        # Own process group (inside the GUI's session), so pause/stop also reach
        # the ffmpeg children, and the kernel still hangs up a job left stopped
        # if the GUI dies without closing it
        if sys.version_info >= (3, 11):
            group_kwargs["process_group"] = 0
        else:
            group_kwargs["preexec_fn"] = os.setpgrp

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # Children must never read the terminal (a background GUI froze on SIGTTOU)
        stdin=subprocess.DEVNULL,
        # config/, srt/ and logs/ resolve to the project wherever the GUI was started
        cwd=str(PROJECT_DIR),
        env=env,
        **group_kwargs,
    )

    # Store process reference for external termination
    if current_process_holder is not None:
        current_process_holder[0] = process

    # newline="" keeps "\r", so progress-bar refreshes can replace the last log line
    stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="")
    paused_notified = False
    try:
        for raw in iter(stream.readline, ""):
            if stop_event and stop_event.is_set():
                break

            # Safety net for platforms where suspending failed: stop reading
            if pause_event and not pause_event.is_set():
                if not paused_notified:
                    msg_queue.put("__PAUSED__")
                    paused_notified = True
                while not pause_event.is_set() and not (stop_event and stop_event.is_set()):
                    pause_event.wait(timeout=0.1)
                if stop_event and stop_event.is_set():
                    break
                msg_queue.put("__RESUMED__")
                paused_notified = False

            is_progress = raw.endswith("\r")
            # Clean ANSI escape codes that cause garbled text
            clean_line = _ANSI_RE.sub("", raw.strip("\r\n"))
            if not clean_line.strip():
                continue
            msg_queue.put(f"__PROGRESS__:{clean_line}" if is_progress else clean_line)
    finally:
        if stop_event and stop_event.is_set():
            msg_queue.put("\n⚠ Processing terminated by user")
            _terminate_process(process)
        stream.close()
        return_code = process.wait()
        shutil.rmtree(job_tmp, ignore_errors=True)

    if stop_event and stop_event.is_set():
        return None
    return return_code


def worker_process_export_srt(
    input_path: str,
    msg_queue: queue.Queue,
    source_lang: str = None,
    model_size: str = None,
    pause_event: threading.Event = None,
    current_process_holder: list = None,
    stop_event: threading.Event = None,
):
    """Run main.py with --export-srt flag to export subtitle file only"""
    try:
        cmd = _build_command("--export-srt", input_path, source_lang, model_size)
        return_code = _run_main_subprocess(
            cmd, msg_queue, pause_event, current_process_holder, stop_event
        )
        if return_code is None:
            return  # cancelled: the UI has already been reset
        if return_code != 0:
            msg_queue.put("__DONE__:False:")
            return

        # Find the generated .srt file in srt/ directory
        try:
            srt_file = _srt_path_for(input_path)
            if not srt_file.exists():
                # Try to find any .srt file with the video name
                srt_files = list(SRT_DIR.glob(f"{Path(input_path).stem}*.srt"))
                srt_file = srt_files[0] if srt_files else None

            if srt_file and srt_file.exists():
                # Copy to Downloads
                downloads = Path.home() / "Downloads"
                downloads.mkdir(parents=True, exist_ok=True)
                dest_file = downloads / srt_file.name
                shutil.copy2(srt_file, dest_file)

                msg_queue.put(f"\n✓ Subtitle file copied to: {dest_file}")
                msg_queue.put(f"__DONE__:True:{dest_file}")
            else:
                msg_queue.put("\n⚠ Warning: Subtitle file not found in srt/ directory")
                msg_queue.put("__DONE__:True:")
        except Exception as e:
            msg_queue.put(f"\n⚠ Warning: Could not copy to Downloads: {e}")
            msg_queue.put("__DONE__:True:")

    except Exception as e:
        msg_queue.put(f"Exception in worker: {e}\n")
        msg_queue.put(traceback.format_exc())
        msg_queue.put("__DONE__:False:")


def worker_process_voiceover(
    input_path: str,
    msg_queue: queue.Queue,
    source_lang: str = None,
    model_size: str = None,
    pause_event: threading.Event = None,
    current_process_holder: list = None,
    stop_event: threading.Event = None,
    voice: str = None,
):
    """Run main.py with --voiceover flag to create video with Vietnamese TTS"""
    try:
        cmd = _build_command("--voiceover", input_path, source_lang, model_size)

        # Add voice flag if specified
        if voice:
            cmd.extend(["--voice", voice])

        # Add output path
        downloads = Path.home() / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        output_file = downloads / f"{Path(input_path).stem}_voiceover.mp4"
        cmd.extend(["--output", str(output_file)])

        return_code = _run_main_subprocess(
            cmd, msg_queue, pause_event, current_process_holder, stop_event
        )
        if return_code is None:
            return  # cancelled: the UI has already been reset
        if return_code == 0:
            msg_queue.put(f"__DONE__:True:{output_file}")
        else:
            msg_queue.put("__DONE__:False:")

    except Exception as e:
        msg_queue.put(f"Exception in worker: {e}\n")
        msg_queue.put(traceback.format_exc())
        msg_queue.put("__DONE__:False:")


def worker_process(
    input_path: str,
    msg_queue: queue.Queue,
    source_lang: str = None,
    model_size: str = None,
    pause_event: threading.Event = None,
    current_process_holder: list = None,
    stop_event: threading.Event = None,
):
    """Run main.py as subprocess to capture all terminal output including progress bars"""
    try:
        cmd = _build_command("--video", input_path, source_lang, model_size)

        # Add output path
        downloads = Path.home() / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        output_file = downloads / f"{Path(input_path).stem}_vietnamese_subtitles.mp4"
        cmd.extend(["--output", str(output_file)])

        return_code = _run_main_subprocess(
            cmd, msg_queue, pause_event, current_process_holder, stop_event
        )
        if return_code is None:
            return  # cancelled: the UI has already been reset
        if return_code == 0:
            msg_queue.put(f"__DONE__:True:{output_file}")
        else:
            msg_queue.put("__DONE__:False:")

    except Exception as e:
        msg_queue.put(f"Exception in worker: {e}\n")
        msg_queue.put(traceback.format_exc())
        msg_queue.put("__DONE__:False:")


class AppTk(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load user preferences
        prefs = load_user_preferences()

        # Current theme and language
        self.current_theme = prefs.get("theme", "light")  # light or dark
        self.current_language = prefs.get("language", "en")  # en or vi

        # Modern color schemes
        self.themes = {
            "light": {
                "bg": "#F5F6FA",
                "primary": "#4A90E2",
                "primary_dark": "#357ABD",
                "primary_hover": "#5BA3F5",
                "success": "#52C41A",
                "success_hover": "#66D42E",
                "danger": "#F5222D",
                "text_dark": "#2C3E50",
                "text_light": "#7F8C8D",
                "border": "#DFE4EA",
                "card_bg": "#FFFFFF",
                "title_bg": "#FFFFFF",
                "log_bg": "#1E1E1E",
                "log_text": "#D4D4D4",
                "input_bg": "#F8F9FA",
                "input_border": "#E1E8ED",
            },
            "dark": {
                "bg": "#1E1E1E",
                "primary": "#5BA3F5",
                "primary_dark": "#4A90E2",
                "primary_hover": "#6BB3FF",
                "success": "#66D42E",
                "success_hover": "#7AE842",
                "danger": "#FF4D4F",
                "text_dark": "#E0E0E0",
                "text_light": "#B0B0B0",
                "border": "#3A3A3A",
                "card_bg": "#2A2A2A",
                "title_bg": "#252525",
                "log_bg": "#0D0D0D",
                "log_text": "#D4D4D4",
                "input_bg": "#2A2A2A",
                "input_border": "#3A3A3A",
            },
        }

        self.colors = self.themes[self.current_theme]

        self.msg_queue = queue.Queue()
        self.worker = None
        self.current_process = None  # Track subprocess
        self.processing_start_time = None
        self.pause_event = threading.Event()
        self.pause_event.set()  # Initially not paused
        self.stop_event = threading.Event()  # Event to signal worker to stop
        self.is_paused = False
        self.paused_elapsed_time = 0  # Track elapsed time when paused
        self.button_state = "start"  # Track button state: start, stop, continue
        self.job_started_at = None  # Wall-clock start of the current job (not pause-adjusted)
        self._last_progress_key = None  # Key of the progress bar shown on the last log line

        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self._poll_queue)
        self.timer.start(100)

        self.srt_timer = QTimer()
        self.srt_timer.timeout.connect(self._check_for_srt)
        self.srt_timer.start(1000)

        # Timer to update runtime label
        self.runtime_timer = QTimer()
        self.runtime_timer.timeout.connect(self._update_runtime)

    def init_ui(self):
        self.setWindowTitle("Vietnamese Subtitle Generator")
        self.setGeometry(100, 100, 900, 730)
        self.setMinimumSize(560, 400)

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        frame_geometry = self.frameGeometry()
        screen_center = self.screen().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

        self.drag_position = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {self.colors['bg']};
                border: 1px solid {self.colors['border']};
                border-radius: 10px;
            }}
        """
        )

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_widget.setLayout(main_layout)

        # Custom Title Bar
        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(45)
        self.title_bar.setStyleSheet(
            f"""
            QWidget {{
                background-color: {self.colors['title_bg']};
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                border-bottom: 1px solid {self.colors['border']};
            }}
        """
        )
        title_bar_layout = QHBoxLayout()
        title_bar_layout.setContentsMargins(15, 0, 10, 0)
        self.title_bar.setLayout(title_bar_layout)

        # Title
        self.title_label = QLabel(self.t('title'))
        self.title_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.title_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent; border: none;"
        )
        title_bar_layout.addWidget(self.title_label)

        title_bar_layout.addStretch()

        # Theme toggle button
        self.theme_btn = QPushButton("☀️" if self.current_theme == "dark" else "🌙")
        self.theme_btn.setFixedSize(35, 35)
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.setToolTip(self.t("theme"))
        hover_color = "#E8E8E8" if self.current_theme == "light" else "#3A3A3A"
        self.theme_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )
        self.theme_btn.clicked.connect(self.toggle_theme)
        title_bar_layout.addWidget(self.theme_btn)

        # Language toggle button
        self.lang_btn = QPushButton("EN" if self.current_language == "vi" else "VI")
        self.lang_btn.setFixedSize(35, 35)
        self.lang_btn.setCursor(Qt.PointingHandCursor)
        self.lang_btn.setToolTip(self.t("language"))
        self.lang_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )
        self.lang_btn.clicked.connect(self.toggle_language)
        title_bar_layout.addWidget(self.lang_btn)

        # Window control buttons
        self.btn_minimize = QPushButton("─")
        self.btn_minimize.setFixedSize(35, 35)
        self.btn_minimize.setCursor(Qt.PointingHandCursor)
        hover_color = "#E8E8E8" if self.current_theme == "light" else "#3A3A3A"
        self.btn_minimize.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )
        self.btn_minimize.clicked.connect(self.showMinimized)
        title_bar_layout.addWidget(self.btn_minimize)

        self.btn_maximize = QPushButton("□")
        self.btn_maximize.setFixedSize(35, 35)
        self.btn_maximize.setCursor(Qt.PointingHandCursor)
        self.btn_maximize.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )
        self.btn_maximize.clicked.connect(self.toggle_maximize)
        title_bar_layout.addWidget(self.btn_maximize)

        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(35, 35)
        self.btn_close.setCursor(Qt.PointingHandCursor)
        self.btn_close.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #E81123;
                color: white;
                border-radius: 4px;
            }}
        """
        )
        self.btn_close.clicked.connect(self.close)
        title_bar_layout.addWidget(self.btn_close)

        main_layout.addWidget(self.title_bar)

        # Content container
        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent; border: none;")
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(12, 10, 12, 12)
        content_layout.setSpacing(12)
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        # Subtitle
        self.subtitle_label = QLabel(self.t("subtitle"))
        self.subtitle_label.setFont(QFont("Segoe UI", 9))
        self.subtitle_label.setStyleSheet(
            f"color: {self.colors['text_light']}; background: transparent;"
        )
        content_layout.addWidget(self.subtitle_label)

        content_layout.addSpacing(6)

        # Card for file selection
        self.file_card = QFrame()
        self.file_card.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self.colors['card_bg']};
                border-radius: 8px;
                padding: 12px;
            }}
        """
        )
        file_layout = QVBoxLayout()
        file_layout.setSpacing(6)
        self.file_card.setLayout(file_layout)

        self.file_label = QLabel(self.t("video_file"))
        self.file_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.file_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        file_layout.addWidget(self.file_label)

        file_row = QHBoxLayout()
        file_row.setSpacing(8)

        self.entry = QLineEdit()
        self.entry.setFont(QFont("Segoe UI", 10))
        self.entry.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 10px 12px;
                color: {self.colors['text_dark']};
            }}
            QLineEdit:focus {{
                border: 2px solid {self.colors['primary']};
            }}
        """
        )
        file_row.addWidget(self.entry)

        self.browse_btn = QPushButton(self.t("browse"))
        self.browse_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.browse_btn.setCursor(Qt.PointingHandCursor)
        self.browse_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 11px 25px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['primary_dark']};
            }}
        """
        )
        self.browse_btn.clicked.connect(self.browse)
        # Keep browse button a fixed width so the entry can expand/shrink
        self.browse_btn.setFixedWidth(120)
        file_row.addWidget(self.browse_btn)

        file_layout.addLayout(file_row)
        content_layout.addWidget(self.file_card)

        # Card for language selection
        self.lang_card = QFrame()
        self.lang_card.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self.colors['card_bg']};
                border-radius: 8px;
                padding: 12px;
            }}
        """
        )
        lang_layout = QVBoxLayout()
        lang_layout.setSpacing(6)
        self.lang_card.setLayout(lang_layout)

        lang_header_layout = QHBoxLayout()
        lang_header_layout.setSpacing(4)

        self.lang_label = QLabel(self.t("source_language"))
        self.lang_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.lang_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        lang_header_layout.addWidget(self.lang_label)

        self.help_btn = QPushButton("?")
        self.help_btn.setFixedSize(16, 16)
        self.help_btn.setCursor(Qt.PointingHandCursor)
        self.help_btn.setToolTip(self.t("language_tooltip"))
        self.help_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QToolTip {{
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
                border: 1px solid {self.colors['border']};
                padding: 5px;
                border-radius: 3px;
                font-size: 10px;
            }}
        """
        )
        lang_header_layout.addWidget(self.help_btn)
        lang_header_layout.addStretch()

        lang_layout.addLayout(lang_header_layout)

        self.model_label = QLabel(self.t("whisper_model"))
        self.model_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.model_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )

        self.model_help_btn = QPushButton("?")
        self.model_help_btn.setFixedSize(16, 16)
        self.model_help_btn.setCursor(Qt.PointingHandCursor)
        self.model_help_btn.setToolTip(self.t("model_tooltip"))
        self.model_help_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QToolTip {{
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
                border: 1px solid {self.colors['border']};
                padding: 5px;
                border-radius: 3px;
                font-size: 10px;
            }}
        """
        )

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(
            [
                "English",
                "Japanese",
                "Chinese (Simplified)",
                "Korean",
                "Thai",
                "Indonesian",
            ]
        )
        self.lang_combo.setFont(QFont("Segoe UI", 9))
        self.lang_combo.setFixedHeight(36)
        # Allow combo to be reasonably sized but not force layout overflow
        self.lang_combo.setMinimumWidth(100)
        self.lang_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lang_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 8px 12px;
                padding-right: 35px;
                color: {self.colors['text_dark']};
            }}
            QComboBox:hover {{
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox:focus {{
                border: 2px solid {self.colors['primary']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 25px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border-style: solid;
                border-width: 6px 5px 0px 5px;
                border-color: {self.colors['text_dark']} transparent transparent transparent;
            }}
            QComboBox::down-arrow:hover {{
                border-color: {self.colors['primary']} transparent transparent transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['card_bg']};
                border: 2px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {self.colors['primary']};
                selection-color: white;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                min-height: 30px;
                margin: 2px;
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {self.colors['primary_hover']};
                color: white;
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {self.colors['primary']};
                color: white;
                border: 1px solid {self.colors['primary_dark']};
            }}
        """
        )

        # Model combobox
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            [
                self.t("model_base"),
                self.t("model_tiny"),
                self.t("model_small"),
                self.t("model_medium"),
                self.t("model_large"),
                self.t("model_turbo"),
            ]
        )
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setFont(QFont("Segoe UI", 9))
        self.model_combo.setFixedHeight(36)
        self.model_combo.setMinimumWidth(100)
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.model_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 8px 12px;
                padding-right: 35px;
                color: {self.colors['text_dark']};
            }}
            QComboBox:hover {{
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox:focus {{
                border: 2px solid {self.colors['primary']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 25px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border-style: solid;
                border-width: 6px 5px 0px 5px;
                border-color: {self.colors['text_dark']} transparent transparent transparent;
            }}
            QComboBox::down-arrow:hover {{
                border-color: {self.colors['primary']} transparent transparent transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['card_bg']};
                border: 2px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {self.colors['primary']};
                selection-color: white;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                min-height: 30px;
                margin: 2px;
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {self.colors['primary_hover']};
                color: white;
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {self.colors['primary']};
                color: white;
                border: 1px solid {self.colors['primary_dark']};
            }}
        """
        )

        combo_row_layout = QHBoxLayout()
        combo_row_layout.setSpacing(12)
        combo_row_layout.setContentsMargins(0, 0, 0, 0)

        lang_column = QVBoxLayout()
        lang_column.setSpacing(4)
        lang_column.setContentsMargins(0, 0, 0, 0)

        lang_col_header = QHBoxLayout()
        lang_col_header.setSpacing(4)
        lang_col_header.addWidget(self.lang_label)
        lang_col_header.addWidget(self.help_btn)
        lang_col_header.addStretch()
        lang_column.addLayout(lang_col_header)
        lang_column.addWidget(self.lang_combo)

        combo_row_layout.addLayout(lang_column, 1)

        model_column = QVBoxLayout()
        model_column.setSpacing(4)
        model_column.setContentsMargins(0, 0, 0, 0)

        model_col_header = QHBoxLayout()
        model_col_header.setSpacing(4)
        model_col_header.addWidget(self.model_label)
        model_col_header.addWidget(self.model_help_btn)
        model_col_header.addStretch()
        model_column.addLayout(model_col_header)
        model_column.addWidget(self.model_combo)

        combo_row_layout.addLayout(model_column, 1)

        # Voice selection column
        voice_column = QVBoxLayout()
        voice_column.setSpacing(4)
        voice_column.setContentsMargins(0, 0, 0, 0)

        # Voice label
        self.voice_label = QLabel(self.t("tts_voice"))
        self.voice_label.setFont(QFont("Segoe UI", 9))
        self.voice_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )

        # Voice help button
        self.voice_help_btn = QPushButton("?")
        self.voice_help_btn.setFixedSize(16, 16)
        self.voice_help_btn.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.voice_help_btn.setCursor(Qt.PointingHandCursor)
        self.voice_help_btn.setToolTip(self.t("voice_tooltip"))
        self.voice_help_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QToolTip {{
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
                border: 1px solid {self.colors['border']};
                padding: 5px;
                border-radius: 3px;
                font-size: 10px;
            }}
        """
        )

        # Voice combobox (VieNeu v3 Turbo voices)
        self.voice_combo = QComboBox()
        self.voice_combo.addItems(
            [
                self.t("voice_ngoc_lan"),
                self.t("voice_ngoc_linh"),
                self.t("voice_truc_ly"),
                self.t("voice_my_duyen"),
                self.t("voice_xuan_vinh"),
                self.t("voice_thai_son"),
                self.t("voice_gia_bao"),
                self.t("voice_duc_tri"),
                self.t("voice_trong_huu"),
                self.t("voice_binh_an"),
            ]
        )
        self.voice_combo.setCurrentIndex(0)  # Default: Ngọc Lan
        self.voice_combo.setMaxVisibleItems(10)
        self.voice_combo.setFont(QFont("Segoe UI", 9))
        self.voice_combo.setFixedHeight(36)
        self.voice_combo.setMinimumWidth(100)
        self.voice_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.voice_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 8px 12px;
                padding-right: 35px;
                color: {self.colors['text_dark']};
            }}
            QComboBox:hover {{
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox:focus {{
                border: 2px solid {self.colors['primary']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 25px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border-style: solid;
                border-width: 6px 5px 0px 5px;
                border-color: {self.colors['text_dark']} transparent transparent transparent;
            }}
            QComboBox::down-arrow:hover {{
                border-color: {self.colors['primary']} transparent transparent transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['card_bg']};
                border: 2px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {self.colors['primary']};
                selection-color: white;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                min-height: 30px;
                margin: 2px;
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {self.colors['primary_hover']};
                color: white;
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {self.colors['primary']};
                color: white;
                border: 1px solid {self.colors['primary_dark']};
            }}
        """
        )

        voice_col_header = QHBoxLayout()
        voice_col_header.setSpacing(4)
        voice_col_header.addWidget(self.voice_label)
        voice_col_header.addWidget(self.voice_help_btn)
        voice_col_header.addStretch()
        voice_column.addLayout(voice_col_header)
        voice_column.addWidget(self.voice_combo)

        combo_row_layout.addLayout(voice_column, 1)
        combo_row_layout.addStretch()

        lang_layout.addLayout(combo_row_layout)

        content_layout.addWidget(self.lang_card)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.start_btn = QPushButton(self.t("start_processing"))
        self.start_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['success']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
            }}
            QPushButton:hover:enabled {{
                background-color: {self.colors['success_hover']};
            }}
            QPushButton:pressed:enabled {{
                background-color: #46A815;
            }}
            QPushButton:disabled {{
                background-color: #95A5A6;
                color: #CCCCCC;
            }}
        """
        )
        self.start_btn.clicked.connect(self.handle_start_stop_continue)
        btn_layout.addWidget(self.start_btn)

        # Export SRT Only button
        self.export_srt_btn = QPushButton(self.t("export_srt"))
        self.export_srt_btn.setFont(QFont("Segoe UI", 10))
        self.export_srt_btn.setCursor(Qt.PointingHandCursor)
        self.export_srt_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #9B59B6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 14px 20px;
            }}
            QPushButton:hover {{
                background-color: #8E44AD;
            }}
            QPushButton:pressed {{
                background-color: #7D3C98;
            }}
        """
        )
        self.export_srt_btn.clicked.connect(self.start_export_srt)
        btn_layout.addWidget(self.export_srt_btn)

        # Voiceover button - Create video with Vietnamese TTS
        self.voiceover_btn = QPushButton(self.t("voiceover_btn"))
        self.voiceover_btn.setFont(QFont("Segoe UI", 10))
        self.voiceover_btn.setCursor(Qt.PointingHandCursor)
        self.voiceover_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #00ACC1;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 14px 20px;
            }}
            QPushButton:hover {{
                background-color: #00BCD4;
            }}
            QPushButton:pressed {{
                background-color: #0097A7;
            }}
        """
        )
        self.voiceover_btn.clicked.connect(self.start_voiceover)
        btn_layout.addWidget(self.voiceover_btn)

        btn_layout.addStretch()
        # Download SRT button (hidden until temp_subtitles.srt is detected)
        self.download_srt_btn = QPushButton(self.t("download_srt"))
        self.download_srt_btn.setFont(QFont("Segoe UI", 10))
        self.download_srt_btn.setCursor(Qt.PointingHandCursor)
        self.download_srt_btn.setVisible(False)
        self.download_srt_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #3A8DFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 18px;
            }}
            QPushButton:hover {{
                background-color: #5AA7FF;
            }}
        """
        )
        self.download_srt_btn.clicked.connect(self.download_srt)
        btn_layout.addWidget(self.download_srt_btn)
        content_layout.addLayout(btn_layout)

        # Logs section
        self.logs_label = QLabel(self.t("processing_logs"))
        self.logs_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.logs_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        content_layout.addWidget(self.logs_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(5000)  # bounded memory on long jobs
        self.log.setFont(QFont("Consolas", 9))
        self.log.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {self.colors['log_bg']};
                color: {self.colors['log_text']};
                border: none;
                border-radius: 4px;
                padding: 12px;
            }}
        """
        )
        # Allow log to expand but don't force a very large minimum height
        self.log.setMinimumHeight(150)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout.addWidget(self.log, stretch=1)

        # Log action buttons row
        log_btn_layout = QHBoxLayout()
        log_btn_layout.setSpacing(8)

        self.reset_btn = QPushButton(self.t("reset"))
        self.reset_btn.setFont(QFont("Segoe UI", 9))
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        reset_bg = "#E0E0E0" if self.current_theme == "light" else "#3A3A3A"
        reset_hover = "#D0D0D0" if self.current_theme == "light" else "#4A4A4A"
        reset_text = "#2C3E50" if self.current_theme == "light" else "#E0E0E0"
        self.reset_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {reset_bg};
                color: {reset_text};
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: {reset_hover};
            }}
        """
        )
        self.reset_btn.clicked.connect(self.reset_ui)
        log_btn_layout.addWidget(self.reset_btn)

        log_btn_layout.addStretch()

        self.log_btn = QPushButton(self.t("view_logs"))
        self.log_btn.setFont(QFont("Segoe UI", 9))
        self.log_btn.setCursor(Qt.PointingHandCursor)
        self.log_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #FFA726;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: #FB8C00;
            }}
        """
        )
        self.log_btn.clicked.connect(self.open_log_file)
        log_btn_layout.addWidget(self.log_btn)

        content_layout.addLayout(log_btn_layout)

        # Status bar
        self.status_label = QLabel(self.t("ready"))
        self.status_label.setFont(QFont("Segoe UI", 8))
        self.status_label.setStyleSheet(
            f"color: {self.colors['text_light']}; background: transparent;"
        )

        # Add a horizontal layout for the status bar and a QSizeGrip for resizing
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        # Runtime label shows elapsed time while processing (placed on right)
        self.runtime_label = QLabel(f"{self.t('runtime')}: 00:00:00")
        self.runtime_label.setFont(QFont("Segoe UI", 8))
        self.runtime_label.setStyleSheet(
            f"color: {self.colors['text_light']}; background: transparent;"
        )
        status_layout.addWidget(self.runtime_label)
        size_grip = QSizeGrip(self)
        status_layout.addWidget(size_grip, 0, Qt.AlignRight | Qt.AlignBottom)
        content_layout.addLayout(status_layout)

        # Set up qtawesome icons
        self._setup_icons()

    def t(self, key):
        """Get translation for current language"""
        return TRANSLATIONS[self.current_language].get(key, key)

    def _setup_icons(self):
        """Set up qtawesome icons — deferred to ensure QApp event loop is running"""
        QTimer.singleShot(0, self._apply_icons)

    def _apply_icons(self):
        """Actually apply qtawesome icons (event loop must be running)"""
        c = self.colors

        # Browse button
        self.browse_btn.setIcon(qta.icon('fa5s.folder-open', color='white'))

        # Start/Stop/Continue button - icon based on state
        if self.button_state == "stop":
            self.start_btn.setIcon(qta.icon('fa5s.stop', color='white'))
        else:
            self.start_btn.setIcon(qta.icon('fa5s.play', color='white'))

        # Export SRT button
        self.export_srt_btn.setIcon(qta.icon('fa5s.download', color='white'))

        # Voiceover button
        self.voiceover_btn.setIcon(qta.icon('fa5s.microphone', color='white'))

        # Download SRT button
        self.download_srt_btn.setIcon(qta.icon('fa5s.download', color='white'))

        # Reset button
        reset_color = "#2C3E50" if self.current_theme == "light" else "#E0E0E0"
        self.reset_btn.setIcon(qta.icon('fa5s.undo', color=reset_color))

        # View Logs button
        self.log_btn.setIcon(qta.icon('fa5s.file-alt', color='white'))

        # Theme toggle button (icon only, no text)
        icon_name = 'fa5s.sun' if self.current_theme == "dark" else 'fa5s.moon'
        self.theme_btn.setIcon(qta.icon(icon_name, color=c['text_light']))
        self.theme_btn.setText("")

        # Set icon size for consistency
        icon_size = QSize(16, 16)
        for btn in [self.browse_btn, self.start_btn, self.export_srt_btn,
                    self.voiceover_btn, self.download_srt_btn, self.reset_btn,
                    self.log_btn, self.theme_btn]:
            btn.setIconSize(icon_size)

    def toggle_theme(self):
        """Toggle between light and dark theme"""
        self.current_theme = "dark" if self.current_theme == "light" else "light"
        self.colors = self.themes[self.current_theme]
        self.update_theme()
        # Save preference
        save_user_preferences(self.current_language, self.current_theme)

    def toggle_language(self):
        """Toggle between English and Vietnamese"""
        self.current_language = "vi" if self.current_language == "en" else "en"
        self.update_ui_texts()
        # Save preference
        save_user_preferences(self.current_language, self.current_theme)

    def update_theme(self):
        """Update all UI elements with new theme colors"""
        self._apply_icons()

        hover_color = "#E8E8E8" if self.current_theme == "light" else "#3A3A3A"

        self.centralWidget().setStyleSheet(
            f"""
            QWidget {{
                background-color: {self.colors['bg']};
                border: 1px solid {self.colors['border']};
                border-radius: 10px;
            }}
        """
        )

        # Update all text colors and backgrounds
        self.title_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent; border: none;"
        )
        self.subtitle_label.setStyleSheet(
            f"color: {self.colors['text_light']}; background: transparent;"
        )
        self.file_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        self.lang_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        self.model_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        self.logs_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        self.status_label.setStyleSheet(
            f"color: {self.colors['text_light']}; background: transparent;"
        )
        self.runtime_label.setStyleSheet(
            f"color: {self.colors['text_light']}; background: transparent;"
        )

        # Update input field
        self.entry.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 10px 12px;
                color: {self.colors['text_dark']};
            }}
            QLineEdit:focus {{
                border: 2px solid {self.colors['primary']};
            }}
        """
        )

        # Update combo box
        self.lang_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 8px 12px;
                padding-right: 35px;
                color: {self.colors['text_dark']};
            }}
            QComboBox:hover {{
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox:focus {{
                border: 2px solid {self.colors['primary']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 25px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border-style: solid;
                border-width: 6px 5px 0px 5px;
                border-color: {self.colors['text_dark']} transparent transparent transparent;
            }}
            QComboBox::down-arrow:hover {{
                border-color: {self.colors['primary']} transparent transparent transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['card_bg']};
                border: 2px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {self.colors['primary']};
                selection-color: white;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                min-height: 30px;
                margin: 2px;
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {self.colors['primary_hover']};
                color: white;
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {self.colors['primary']};
                color: white;
                border: 1px solid {self.colors['primary_dark']};
            }}
        """
        )

        # Update log text area
        self.log.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {self.colors['log_bg']};
                color: {self.colors['log_text']};
                border: none;
                border-radius: 4px;
                padding: 12px;
            }}
        """
        )

        # Update buttons
        self.browse_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 11px 25px;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.colors['primary_dark']};
            }}
        """
        )

        # Update help button
        self.help_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QToolTip {{
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
                border: 1px solid {self.colors['border']};
                padding: 5px;
                border-radius: 3px;
                font-size: 10px;
            }}
        """
        )

        # Update model help button
        self.model_help_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QToolTip {{
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
                border: 1px solid {self.colors['border']};
                padding: 5px;
                border-radius: 3px;
                font-size: 10px;
            }}
        """
        )

        # Update model combo box
        self.model_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 8px 12px;
                padding-right: 35px;
                color: {self.colors['text_dark']};
            }}
            QComboBox:hover {{
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox:focus {{
                border: 2px solid {self.colors['primary']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 25px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border-style: solid;
                border-width: 6px 5px 0px 5px;
                border-color: {self.colors['text_dark']} transparent transparent transparent;
            }}
            QComboBox::down-arrow:hover {{
                border-color: {self.colors['primary']} transparent transparent transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['card_bg']};
                border: 2px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {self.colors['primary']};
                selection-color: white;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                min-height: 30px;
                margin: 2px;
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {self.colors['primary_hover']};
                color: white;
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {self.colors['primary']};
                color: white;
                border: 1px solid {self.colors['primary_dark']};
            }}
        """
        )

        # Update voice combo box
        self.voice_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {self.colors['input_bg']};
                border: 1px solid {self.colors['input_border']};
                border-radius: 4px;
                padding: 8px 12px;
                padding-right: 35px;
                color: {self.colors['text_dark']};
            }}
            QComboBox:hover {{
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox:focus {{
                border: 2px solid {self.colors['primary']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 25px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
                border-style: solid;
                border-width: 6px 5px 0px 5px;
                border-color: {self.colors['text_dark']} transparent transparent transparent;
            }}
            QComboBox::down-arrow:hover {{
                border-color: {self.colors['primary']} transparent transparent transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {self.colors['card_bg']};
                border: 2px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                selection-background-color: {self.colors['primary']};
                selection-color: white;
                outline: none;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 8px 12px;
                border: 1px solid {self.colors['border']};
                border-radius: 3px;
                min-height: 30px;
                margin: 2px;
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
            }}
            QComboBox QAbstractItemView::item:hover {{
                background-color: {self.colors['primary_hover']};
                color: white;
                border: 1px solid {self.colors['primary']};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background-color: {self.colors['primary']};
                color: white;
                border: 1px solid {self.colors['primary_dark']};
            }}
        """
        )

        # Update voice label
        self.voice_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )

        # Update voice help button
        self.voice_help_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 9px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.colors['primary_hover']};
            }}
            QToolTip {{
                background-color: {self.colors['card_bg']};
                color: {self.colors['text_dark']};
                border: 1px solid {self.colors['border']};
                padding: 5px;
                border-radius: 3px;
                font-size: 10px;
            }}
        """
        )

        # Update start button (check current state)
        current_text = self.start_btn.text()
        if "Stop" in current_text or "Dừng" in current_text:
            self.start_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.colors['danger']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 20px;
                }}
                QPushButton:hover {{
                    background-color: #FF4D4F;
                }}
                QPushButton:pressed {{
                    background-color: #CF1322;
                }}
            """
            )
        else:
            self.start_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.colors['success']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 20px;
                }}
                QPushButton:hover:enabled {{
                    background-color: {self.colors['success_hover']};
                }}
                QPushButton:pressed:enabled {{
                    background-color: #46A815;
                }}
                QPushButton:disabled {{
                    background-color: #95A5A6;
                    color: #CCCCCC;
                }}
            """
            )

        self.log_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #FFA726;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: #FB8C00;
            }}
        """
        )

        reset_bg = "#E0E0E0" if self.current_theme == "light" else "#3A3A3A"
        reset_hover = "#D0D0D0" if self.current_theme == "light" else "#4A4A4A"
        reset_text = "#2C3E50" if self.current_theme == "light" else "#E0E0E0"
        self.reset_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {reset_bg};
                color: {reset_text};
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
            }}
            QPushButton:hover {{
                background-color: {reset_hover};
            }}
        """
        )

        self.export_srt_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #9B59B6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 14px 20px;
            }}
            QPushButton:hover {{
                background-color: #8E44AD;
            }}
            QPushButton:pressed {{
                background-color: #7D3C98;
            }}
        """
        )

        self.voiceover_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #00ACC1;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 14px 20px;
            }}
            QPushButton:hover {{
                background-color: #00BCD4;
            }}
            QPushButton:pressed {{
                background-color: #0097A7;
            }}
        """
        )

        self.download_srt_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #3A8DFF;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 18px;
            }}
            QPushButton:hover {{
                background-color: #5AA7FF;
            }}
        """
        )

        # Update theme and language buttons
        self.theme_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )

        self.lang_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )

        # Update card frames
        self.file_card.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self.colors['card_bg']};
                border-radius: 8px;
                padding: 18px;
            }}
        """
        )

        self.lang_card.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self.colors['card_bg']};
                border-radius: 8px;
                padding: 18px;
            }}
        """
        )

        # Update title bar
        self.title_bar.setStyleSheet(
            f"""
            QWidget {{
                background-color: {self.colors['title_bg']};
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                border-bottom: 1px solid {self.colors['border']};
            }}
        """
        )

        # Update window control buttons
        self.btn_minimize.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )

        self.btn_maximize.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
                border-radius: 4px;
            }}
        """
        )

        self.btn_close.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                color: {self.colors['text_light']};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #E81123;
                color: white;
                border-radius: 4px;
            }}
        """
        )

    def update_ui_texts(self):
        """Update all text labels with current language"""
        # Update language button
        self.lang_btn.setText("EN" if self.current_language == "vi" else "VI")

        # Update all labels and buttons
        self.title_label.setText(self.t('title'))
        self.subtitle_label.setText(self.t("subtitle"))
        self.file_label.setText(self.t("video_file"))
        self.browse_btn.setText(self.t("browse"))
        self.lang_label.setText(self.t("source_language"))
        self.help_btn.setToolTip(self.t("language_tooltip"))
        self.model_label.setText(self.t("whisper_model"))
        self.model_help_btn.setToolTip(self.t("model_tooltip"))
        self.voice_label.setText(self.t("tts_voice"))
        self.voice_help_btn.setToolTip(self.t("voice_tooltip"))

        # Update model combobox items
        current_model_index = self.model_combo.currentIndex()
        self.model_combo.clear()
        self.model_combo.addItems(
            [
                self.t("model_base"),
                self.t("model_tiny"),
                self.t("model_small"),
                self.t("model_medium"),
                self.t("model_large"),
                self.t("model_turbo"),
            ]
        )
        self.model_combo.setCurrentIndex(current_model_index)

        # Update voice combobox items (VieNeu voices)
        current_voice_index = self.voice_combo.currentIndex()
        self.voice_combo.clear()
        self.voice_combo.addItems(
            [
                self.t("voice_ngoc_lan"),
                self.t("voice_ngoc_linh"),
                self.t("voice_truc_ly"),
                self.t("voice_my_duyen"),
                self.t("voice_xuan_vinh"),
                self.t("voice_thai_son"),
                self.t("voice_gia_bao"),
                self.t("voice_duc_tri"),
                self.t("voice_trong_huu"),
                self.t("voice_binh_an"),
            ]
        )
        self.voice_combo.setCurrentIndex(current_voice_index)

        # Update button text based on current state
        current_btn_text = self.start_btn.text()
        if "Start" in current_btn_text or "Bắt" in current_btn_text:
            self.start_btn.setText(self.t("start_processing"))
            self.button_state = "start"
            self.start_btn.setIcon(qta.icon('fa5s.play', color='white'))
        elif "Stop" in current_btn_text or "Dừng" in current_btn_text:
            self.start_btn.setText(self.t("stop"))
            self.button_state = "stop"
            self.start_btn.setIcon(qta.icon('fa5s.stop', color='white'))
        elif "Continue" in current_btn_text or "Tiếp" in current_btn_text:
            self.start_btn.setText(self.t("continue"))
            self.button_state = "continue"
            self.start_btn.setIcon(qta.icon('fa5s.play', color='white'))

        self.log_btn.setText(self.t("view_logs"))
        self.reset_btn.setText(self.t("reset"))
        self.export_srt_btn.setText(self.t("export_srt"))
        self.voiceover_btn.setText(self.t("voiceover_btn"))
        self.download_srt_btn.setText(self.t("download_srt"))
        self.logs_label.setText(self.t("processing_logs"))

        # Update status if it's at default
        current_status = self.status_label.text()
        if "Ready" in current_status or "Sẵn sàng" in current_status:
            self.status_label.setText(self.t("ready"))
        elif "Exporting" in current_status or "xuất" in current_status:
            self.status_label.setText(self.t("exporting"))
        elif "Processing" in current_status or "Đang xử lý" in current_status:
            self.status_label.setText(self.t("processing"))
        elif (
            "Creating voiceover" in current_status
            or "video thuyết minh" in current_status
        ):
            self.status_label.setText(self.t("voiceover_processing"))

        # Update runtime label
        runtime_text = self.runtime_label.text()
        if ":" in runtime_text:
            time_part = (
                runtime_text.split(":", 1)[1].strip()
                if ":" in runtime_text
                else "00:00:00"
            )
            self.runtime_label.setText(f"{self.t('runtime')}: {time_part}")

        # Update tooltips
        self.theme_btn.setToolTip(self.t("theme"))
        self.lang_btn.setToolTip(self.t("language"))

    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.LeftButton:
            # Only allow dragging from top 45px (title bar area)
            if event.position().y() < 45:
                self.drag_position = (
                    event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                )
                event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging"""
        if event.buttons() == Qt.LeftButton and self.drag_position is not None:
            if event.position().y() < 45:  # Only drag from title bar
                self.move(event.globalPosition().toPoint() - self.drag_position)
                event.accept()

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop dragging"""
        self.drag_position = None

    def browse(self):
        file_dialog = QFileDialog()
        path, _ = file_dialog.getOpenFileName(
            self,
            self.t("select_video"),
            "",
            "Video files (*.mp4 *.mkv *.mov *.avi);;MP4 files (*.mp4);;MKV files (*.mkv);;All files (*.*)",
        )
        if path:
            self.entry.setText(path)
            if not self._job_running():  # never enable Start next to a running job
                self.start_btn.setEnabled(True)
            self.status_label.setText(f"{self.t('selected')}: {Path(path).name}")
            self.log.append(f"{self.t('video_selected')}: {path}")

    def handle_start_stop_continue(self):
        """Handle Start/Stop/Continue button click"""
        if self.button_state == "start":
            self.start_processing()
        elif self.button_state == "stop":
            self.pause_processing()
        elif self.button_state == "continue":
            self.resume_processing()

    def _job_running(self) -> bool:
        """True while a main.py job (any of the three buttons) is running."""
        try:
            return self.worker is not None and self.worker.is_alive()
        except Exception:
            return False

    def _selected_options(self):
        """Return (language code, model name) chosen in the UI."""
        selected_code = LANG_MAP.get(self.lang_combo.currentText(), "en")
        model_index = self.model_combo.currentIndex()
        selected_model = (
            MODEL_LIST[model_index] if 0 <= model_index < len(MODEL_LIST) else "base"
        )
        return selected_code, selected_model

    def _set_stop_button(self):
        """Turn the main button into "Stop" (pause) for the running job."""
        self.start_btn.setEnabled(True)
        self.start_btn.setText(self.t("stop"))
        self.button_state = "stop"
        self.start_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['danger']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: #FF4D4F;
            }}
            QPushButton:pressed {{
                background-color: #CF1322;
            }}
        """
        )

    def _launch_job(self, target, input_path, extra_args=()):
        """Common start logic: one job at a time, pausable and cancellable."""
        # Only one job at a time: the jobs share the GPU/CPU and srt/ outputs
        self.export_srt_btn.setEnabled(False)
        self.voiceover_btn.setEnabled(False)
        self._set_stop_button()
        self._last_progress_key = None

        selected_code, selected_model = self._selected_options()

        # Reset pause state
        self.pause_event.set()
        self.is_paused = False
        self.paused_elapsed_time = 0
        self.stop_event.clear()  # Clear stop flag

        # Create holder for subprocess reference
        current_process_holder = [None]
        self.current_process = current_process_holder

        self.worker = threading.Thread(
            target=target,
            args=(
                input_path,
                self.msg_queue,
                selected_code,
                selected_model,
                self.pause_event,
                current_process_holder,
                self.stop_event,
                *extra_args,
            ),
        )
        self.worker.daemon = True
        self.worker.start()

        # Start runtime timer
        self.job_started_at = time.time()
        try:
            self.processing_start_time = time.time()
            self.runtime_timer.start(1000)
        except Exception:
            self.processing_start_time = None

    def start_export_srt(self):
        """Start exporting SRT file only (without creating video)"""
        input_path = self.entry.text()
        if not input_path:
            QMessageBox.warning(
                self, self.t("no_file_selected"), self.t("select_export_warning")
            )
            return
        if self._job_running():
            return

        self.status_label.setText(self.t("exporting"))
        self.log.clear()
        self.log.append(f"{self.t('starting_export')}\n")
        self._launch_job(worker_process_export_srt, input_path)

    def start_voiceover(self):
        """Start creating voiceover video with Vietnamese TTS"""
        input_path = self.entry.text()
        if not input_path:
            QMessageBox.warning(
                self, self.t("no_file_selected"), self.t("select_voiceover_warning")
            )
            return
        if self._job_running():
            return

        self.status_label.setText(self.t("voiceover_processing"))
        self.log.clear()
        self.log.append(f"{self.t('voiceover_starting')}\n")

        # Read selected voice from UI (VieNeu v3 Turbo voices)
        # Format: "Ngọc Lan (Nữ - Dịu dàng)" → extract "Ngọc Lan"
        selected_label = self.voice_combo.currentText()
        selected_voice = selected_label.split(" (")[0] if " (" in selected_label else "Ngọc Lan"

        self._launch_job(worker_process_voiceover, input_path, (selected_voice,))

    def start_processing(self):
        """Start the processing task"""
        input_path = self.entry.text()
        if not input_path:
            QMessageBox.warning(
                self, self.t("no_file_selected"), self.t("select_video_warning")
            )
            return
        if self._job_running():
            return

        self.status_label.setText(self.t("processing"))
        self.log.clear()
        self.log.append(f"{self.t('starting')}\n")
        self._launch_job(worker_process, input_path)

    def pause_processing(self):
        """Pause the processing task"""
        if self.worker and self.worker.is_alive():
            self.pause_event.clear()  # Signal worker to pause
            self.is_paused = True
            # Really stop the job (and its ffmpeg children), not just its output
            if self.current_process and self.current_process[0] is not None:
                _suspend_process(self.current_process[0], True)

            # Save elapsed time when pausing
            if self.processing_start_time:
                self.paused_elapsed_time = time.time() - self.processing_start_time

            # Stop runtime timer
            self.runtime_timer.stop()

            # Change button to Continue
            self.start_btn.setText(self.t("continue"))
            self.button_state = "continue"
            self.start_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.colors['success']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 20px;
                }}
                QPushButton:hover {{
                    background-color: {self.colors['success_hover']};
                }}
                QPushButton:pressed {{
                    background-color: #46A815;
                }}
            """
            )
            self.status_label.setText(self.t("paused"))
            self.log.append(f"\n{self.t('paused_user')}")

    def resume_processing(self):
        """Resume the processing task"""
        if self.worker and self.worker.is_alive():
            if self.current_process and self.current_process[0] is not None:
                _suspend_process(self.current_process[0], False)
            self.pause_event.set()  # Signal worker to resume
            self.is_paused = False

            # Adjust start time to account for paused duration
            if self.processing_start_time:
                self.processing_start_time = time.time() - self.paused_elapsed_time

            # Restart runtime timer
            self.runtime_timer.start(1000)

            # Change button back to Stop
            self.start_btn.setText(self.t("stop"))
            self.button_state = "stop"
            self.start_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.colors['danger']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 10px 20px;
                }}
                QPushButton:hover {{
                    background-color: #FF4D4F;
                }}
                QPushButton:pressed {{
                    background-color: #CF1322;
                }}
            """
            )
            self.status_label.setText(self.t("resumed"))
            self.log.append(f"{self.t('resumed')}\n")

    def open_log_file(self):
        """Open the log file with default text editor"""
        log_file = PROJECT_DIR / "logs" / "subtitle_generator.log"  # main.py runs in PROJECT_DIR

        if not log_file.exists():
            QMessageBox.warning(
                self,
                self.t("log_not_found"),
                self.t("log_not_found_msg"),
            )
            return

        try:
            # Open with default application based on OS
            if sys.platform == "win32":
                os.startfile(log_file)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(log_file)])
            else:  # Linux
                subprocess.run(["xdg-open", str(log_file)])

            self.status_label.setText(f"{self.t('opened_log')}: {log_file}")
        except Exception as e:
            QMessageBox.critical(
                self,
                self.t("error_opening_log"),
                f"{self.t('error_opening_log_msg')}\n{e}",
            )

    def open_downloads_folder(self):
        """Open the Downloads folder"""
        downloads = Path.home() / "Downloads"
        try:
            if sys.platform == "win32":
                os.startfile(downloads)
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(downloads)])
            else:  # Linux
                subprocess.run(["xdg-open", str(downloads)])
            self.status_label.setText(f"{self.t('opened_folder')}: {downloads}")
        except Exception as e:
            print(f"Could not open Downloads folder: {e}")

    def _current_srt_path(self):
        """SRT written by the job started in this session for the selected video, if any."""
        input_path = self.entry.text().strip()
        started = getattr(self, "job_started_at", None)
        if not input_path or not started:
            return None
        srt_path = _srt_path_for(input_path)
        try:
            if srt_path.exists() and srt_path.stat().st_mtime >= started - 1:
                return srt_path
        except OSError:
            pass
        return None

    def _check_for_srt(self):
        """Show the download button once the current job has written its SRT (kept in srt/)."""
        try:
            exists = self._current_srt_path() is not None
            # Only show button when file exists
            if exists and not self.download_srt_btn.isVisible():
                self.download_srt_btn.setVisible(True)
                self.status_label.setText(self.t("srt_ready"))
                self.log.append(self.t("temp_srt_found"))
            elif not exists and self.download_srt_btn.isVisible():
                # Hide when removed
                self.download_srt_btn.setVisible(False)
        except Exception:
            # Non-critical; do not spam logs
            pass

    def download_srt(self):
        """Open a save dialog and copy the job's SRT (srt/<video>_vietnamese.srt) there."""
        srt_path = self._current_srt_path()
        if srt_path is None:
            QMessageBox.warning(
                self, self.t("srt_not_found"), self.t("srt_not_available")
            )
            return

        # Ask user where to save
        dialog = QFileDialog()
        save_path, _ = dialog.getSaveFileName(
            self,
            self.t("save_subtitle"),
            str(Path.home() / "Downloads" / srt_path.name),
            "SRT files (*.srt);;All files (*.*)",
        )
        if not save_path:
            return

        try:
            shutil.copyfile(str(srt_path), save_path)
            QMessageBox.information(
                self, self.t("saved"), f"{self.t('saved_subtitle')}\n{save_path}"
            )
            self.status_label.setText(
                f"{self.t('saved_status')}: {Path(save_path).name}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, self.t("save_failed"), f"{self.t('save_failed_msg')}\n{e}"
            )

    def _update_runtime(self):
        """Update the runtime label each second while processing is active."""
        try:
            if not self.processing_start_time:
                self.runtime_label.setText(f"{self.t('runtime')}: 00:00:00")
                return

            elapsed = int(time.time() - self.processing_start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.runtime_label.setText(
                f"{self.t('runtime')}: {hours:02d}:{minutes:02d}:{seconds:02d}"
            )
        except Exception:
            pass

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg.startswith("__DONE__:"):
                    self._last_progress_key = None
                    # Format: __DONE__:<True|False>:[optional_path]
                    # Cannot use split(":", 2) because paths contain colons on Windows (e.g. C:\...)
                    suffix = msg[len("__DONE__:"):]
                    colon_pos = suffix.find(":")
                    if colon_pos != -1:
                        ok = suffix[:colon_pos] == "True"
                        out = suffix[colon_pos + 1:]
                    else:
                        ok = suffix == "True"
                        out = ""
                    if ok:
                        self.log.append("\n" + "=" * 80)
                        self.log.append(self.t("completed"))
                        self.log.append("=" * 80)
                        # Check if this is an SRT export or video processing
                        if out and out.endswith(".srt"):
                            self.status_label.setText(self.t("srt_exported"))
                        else:
                            self.status_label.setText(self.t("completed_status"))
                        # Open Downloads folder
                        self.open_downloads_folder()
                        # Stop runtime timer
                        try:
                            self.runtime_timer.stop()
                        except Exception:
                            pass
                    else:
                        self.log.append("\n" + "=" * 80)
                        self.log.append(self.t("failed"))
                        self.log.append("=" * 80)
                        self.status_label.setText(self.t("failed_status"))
                        QMessageBox.critical(
                            self,
                            self.t("error_title"),
                            self.t("error_message"),
                        )

                        # Stop runtime timer on failure as well
                        try:
                            self.runtime_timer.stop()
                        except Exception:
                            pass

                    # Restore start button
                    self.is_paused = False
                    self.start_btn.setEnabled(True)
                    self.export_srt_btn.setEnabled(True)
                    self.voiceover_btn.setEnabled(True)
                    self.start_btn.setText(self.t("start_processing"))
                    self.button_state = "start"
                    self.start_btn.setStyleSheet(
                        f"""
                        QPushButton {{
                            background-color: {self.colors['success']};
                            color: white;
                            border: none;
                            border-radius: 4px;
                            padding: 10px 20px;
                        }}
                        QPushButton:hover:enabled {{
                            background-color: {self.colors['success_hover']};
                        }}
                        QPushButton:pressed:enabled {{
                            background-color: #46A815;
                        }}
                    """
                    )
                elif msg == "__PAUSED__":
                    # Handle paused notification from worker
                    pass
                elif msg == "__RESUMED__":
                    # Handle resumed notification from worker
                    pass
                else:
                    self._append_log(msg)
        except queue.Empty:
            pass

    def _append_log(self, msg: str) -> None:
        """Append a log line; a progress-bar refresh replaces the previous one.

        tqdm/ffmpeg redraw their bars with '\\r'. Appending every redraw buried
        real warnings under thousands of near-identical lines.
        """
        is_progress = msg.startswith("__PROGRESS__:")
        if is_progress:
            msg = msg[len("__PROGRESS__:"):]
        key = re.match(r"[^:|=]*", msg).group(0).strip()[:40]
        last_key = getattr(self, "_last_progress_key", None)

        document = self.log.document()
        if (
            last_key is not None
            and key == last_key
            # the last line must still be our progress line (the GUI may have
            # appended its own message since)
            and document.lastBlock().text().startswith(last_key)
        ):
            # Same bar as the last line: overwrite it (final "\n" state included).
            # A document cursor edits without moving the view or the user's selection.
            scrollbar = self.log.verticalScrollBar()
            at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
            cursor = QTextCursor(document)
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.insertText(msg)
            if at_bottom:
                scrollbar.setValue(scrollbar.maximum())
        else:
            self.log.append(msg)
        self._last_progress_key = key if is_progress else None

    def reset_ui(self):
        """Reset the UI to initial state: clear path, logs, status, and hide download button.

        If a processing worker is running, warn the user and do not reset.
        """
        try:
            running = (
                self.worker is not None
                and getattr(self.worker, "is_alive", lambda: False)()
            )
        except Exception:
            running = False

        if running:
            # If paused, show warning with confirmation
            if self.is_paused:
                reply = QMessageBox.warning(
                    self,
                    self.t("warning"),
                    self.t("warning_paused"),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return
                # User confirmed, terminate the worker and subprocess
                try:
                    # Signal worker to stop
                    self.stop_event.set()
                    self.pause_event.set()  # Unblock if waiting

                    # Terminate subprocess (and its ffmpeg children) if it exists,
                    # even while it is suspended by Pause
                    if self.current_process and self.current_process[0] is not None:
                        _terminate_process(self.current_process[0])

                    # Wait a bit for worker thread to finish
                    if self.worker and self.worker.is_alive():
                        self.worker.join(timeout=2)

                    self.worker = None
                    self.current_process = None
                except Exception as e:
                    print(f"Error terminating process: {e}")
            else:
                # Running and not paused
                QMessageBox.warning(
                    self,
                    self.t("task_running"),
                    self.t("task_running_msg"),
                )
                return

        # Clear UI fields
        self.entry.clear()
        self.start_btn.setEnabled(False)
        # A cancelled job never posts __DONE__, so re-enable the job buttons here
        self.export_srt_btn.setEnabled(True)
        self.voiceover_btn.setEnabled(True)
        self.start_btn.setText(self.t("start_processing"))
        self.button_state = "start"
        self.start_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self.colors['success']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
            }}
            QPushButton:hover:enabled {{
                background-color: {self.colors['success_hover']};
            }}
            QPushButton:pressed:enabled {{
                background-color: #46A815;
            }}
        """
        )

        # Clear logs and status
        self.log.clear()
        self.status_label.setText(self.t("ready"))

        # Clear the selected file path
        self.video_path = None
        # Reset runtime
        try:
            self.processing_start_time = None
            self.paused_elapsed_time = 0
            self.is_paused = False
            self.pause_event.set()
            self.runtime_timer.stop()
            self.runtime_label.setText(f"{self.t('runtime')}: 00:00:00")
        except Exception:
            pass

        # Hide download button if visible
        try:
            if self.download_srt_btn.isVisible():
                self.download_srt_btn.setVisible(False)
        except Exception:
            pass

        # Reset language selection to default
        try:
            self.lang_combo.setCurrentIndex(0)
        except Exception:
            pass

    def closeEvent(self, event):
        """Handle window close event"""
        try:
            running = (
                self.worker is not None
                and getattr(self.worker, "is_alive", lambda: False)()
            )
        except Exception:
            running = False

        if running:
            prompt = self.t("exit_running")
        else:
            prompt = self.t("exit_prompt")

        reply = QMessageBox.question(
            self,
            self.t("confirm_exit"),
            prompt,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            if running:
                # Do not leave main.py (and ffmpeg) running headless after the
                # window is gone: it would keep the GPU busy and collide with
                # the next run.
                self.stop_event.set()
                self.pause_event.set()
                if self.current_process and self.current_process[0] is not None:
                    _terminate_process(self.current_process[0])
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    window = AppTk()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
