import threading
import queue
import traceback
import subprocess
import sys
import os
import re
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
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
import shutil


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


def worker_process_export_srt(
    input_path: str,
    msg_queue: queue.Queue,
    source_lang: str = None,
    pause_event: threading.Event = None,
    current_process_holder: list = None,
    stop_event: threading.Event = None,
):
    """Run main.py to export subtitle file only"""
    try:
        # Build command for exporting SRT
        cmd = [sys.executable, "main.py", "--input", input_path]

        # Add language flag if Japanese
        if source_lang == "ja":
            cmd.insert(2, "--jp")

        msg_queue.put(f"Command: {' '.join(cmd)}\n")
        msg_queue.put("=" * 80 + "\n")

        # Set environment to force UTF-8 encoding for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Run subprocess and capture output in real-time with proper encoding
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        # Store process reference for external termination
        if current_process_holder is not None:
            current_process_holder[0] = process

        # Read output line by line
        paused_notified = False
        srt_file = None
        for line in iter(process.stdout.readline, ""):
            # Check if we should stop
            if stop_event and stop_event.is_set():
                msg_queue.put("\n⚠ Processing terminated by user")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                break

            if line:
                # Check if paused
                if pause_event and not pause_event.is_set():
                    if not paused_notified:
                        msg_queue.put("__PAUSED__")
                        paused_notified = True

                    # Wait until resumed or stopped
                    while not pause_event.is_set():
                        if stop_event and stop_event.is_set():
                            msg_queue.put("\n⚠ Processing terminated by user")
                            process.terminate()
                            try:
                                process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            return
                        pause_event.wait(timeout=0.1)

                    msg_queue.put("__RESUMED__")
                    paused_notified = False

                # Clean ANSI escape codes that cause garbled text
                clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line.rstrip())
                msg_queue.put(clean_line)

                # Look for output file path in the logs
                if (
                    "Subtitle file saved to:" in clean_line
                    or "Subtitle file created successfully:" in clean_line
                ):
                    # Extract the file path
                    parts = clean_line.split(":", 1)
                    if len(parts) > 1:
                        srt_file = parts[1].strip()

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            # Find the generated .srt file in srt/ directory
            try:
                import shutil

                # Get video filename without extension
                video_stem = Path(input_path).stem
                srt_dir = Path("srt")

                # Look for the generated .srt file
                expected_srt = srt_dir / f"{video_stem}_vietnamese.srt"

                if expected_srt.exists():
                    srt_file = str(expected_srt)
                else:
                    # Try to find any .srt file with the video name
                    srt_files = list(srt_dir.glob(f"{video_stem}*.srt"))
                    if srt_files:
                        srt_file = str(srt_files[0])

                if srt_file and Path(srt_file).exists():
                    # Copy to Downloads
                    downloads = Path.home() / "Downloads"
                    downloads.mkdir(parents=True, exist_ok=True)
                    dest_file = downloads / Path(srt_file).name

                    shutil.copy2(srt_file, dest_file)

                    msg_queue.put(f"\n✓ Subtitle file copied to: {dest_file}")
                    msg_queue.put(f"__DONE__:True:{dest_file}")
                else:
                    msg_queue.put(
                        "\n⚠ Warning: Subtitle file not found in srt/ directory"
                    )
                    msg_queue.put(f"__DONE__:True:")
            except Exception as e:
                msg_queue.put(f"\n⚠ Warning: Could not copy to Downloads: {e}")
                msg_queue.put(f"__DONE__:True:")
        else:
            msg_queue.put(f"__DONE__:False:")

    except Exception as e:
        msg_queue.put(f"Exception in worker: {e}\n")
        msg_queue.put(traceback.format_exc())
        msg_queue.put("__DONE__:False:")


def worker_process(
    input_path: str,
    msg_queue: queue.Queue,
    source_lang: str = None,
    pause_event: threading.Event = None,
    current_process_holder: list = None,
    stop_event: threading.Event = None,
):
    """Run main.py as subprocess to capture all terminal output including progress bars"""
    try:
        # Build command
        cmd = [sys.executable, "main.py", "--video", "--input", input_path]

        # Add language flag for non-English languages
        if source_lang and source_lang != "en":
            cmd.insert(2, "--language")
            cmd.insert(3, source_lang)

        # Add output path
        downloads = Path.home() / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem
        output_file = downloads / f"{stem}_vietnamese_subtitles.mp4"
        cmd.extend(["--output", str(output_file)])

        msg_queue.put(f"Command: {' '.join(cmd)}\n")
        msg_queue.put("=" * 80 + "\n")

        # Set environment to force UTF-8 encoding for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # Run subprocess and capture output in real-time with proper encoding
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        # Store process reference for external termination
        if current_process_holder is not None:
            current_process_holder[0] = process

        # Read output line by line
        paused_notified = False
        for line in iter(process.stdout.readline, ""):
            # Check if we should stop
            if stop_event and stop_event.is_set():
                msg_queue.put("\n⚠ Processing terminated by user")
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                break

            if line:
                # Check if paused
                if pause_event and not pause_event.is_set():
                    if not paused_notified:
                        msg_queue.put("__PAUSED__")
                        paused_notified = True

                    # Wait until resumed or stopped
                    while not pause_event.is_set():
                        if stop_event and stop_event.is_set():
                            msg_queue.put("\n⚠ Processing terminated by user")
                            process.terminate()
                            try:
                                process.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            return
                        pause_event.wait(timeout=0.1)

                    msg_queue.put("__RESUMED__")
                    paused_notified = False

                # Clean ANSI escape codes that cause garbled text
                clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line.rstrip())
                msg_queue.put(clean_line)

        process.stdout.close()
        return_code = process.wait()

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

        self.init_ui()

        # Setup timer to poll queue
        self.timer = QTimer()
        self.timer.timeout.connect(self._poll_queue)
        self.timer.start(100)

        # Timer to check for presence of temporary SRT file
        self.srt_timer = QTimer()
        self.srt_timer.timeout.connect(self._check_for_srt)
        self.srt_timer.start(1000)

        # Timer to update runtime label
        self.runtime_timer = QTimer()
        self.runtime_timer.timeout.connect(self._update_runtime)

    def init_ui(self):
        self.setWindowTitle("Vietnamese Subtitle Generator")
        self.setGeometry(100, 100, 900, 730)
        # Allow the window to be resized by the user and set a sensible minimum
        self.setMinimumSize(640, 480)

        # Remove default title bar for custom look
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        # Center window on screen
        frame_geometry = self.frameGeometry()
        screen_center = self.screen().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())

        # Variables for window dragging
        self.drag_position = None

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Set background color with border
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

        # Title with icon
        self.title_label = QLabel(f"🎬 {self.t('title')}")
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
        content_layout.setContentsMargins(20, 15, 20, 20)
        content_layout.setSpacing(15)
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget)

        # Subtitle
        self.subtitle_label = QLabel(self.t("subtitle"))
        self.subtitle_label.setFont(QFont("Segoe UI", 9))
        self.subtitle_label.setStyleSheet(
            f"color: {self.colors['text_light']}; background: transparent;"
        )
        content_layout.addWidget(self.subtitle_label)

        content_layout.addSpacing(10)

        # Card for file selection
        self.file_card = QFrame()
        self.file_card.setStyleSheet(
            f"""
            QFrame {{
                background-color: {self.colors['card_bg']};
                border-radius: 8px;
                padding: 18px;
            }}
        """
        )
        file_layout = QVBoxLayout()
        file_layout.setSpacing(8)
        self.file_card.setLayout(file_layout)

        self.file_label = QLabel(self.t("video_file"))
        self.file_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.file_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        file_layout.addWidget(self.file_label)

        file_row = QHBoxLayout()
        file_row.setSpacing(12)

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
                padding: 18px;
            }}
        """
        )
        lang_layout = QVBoxLayout()
        lang_layout.setSpacing(8)
        self.lang_card.setLayout(lang_layout)

        # Create horizontal layout for label and help button
        lang_header_layout = QHBoxLayout()
        lang_header_layout.setSpacing(8)

        self.lang_label = QLabel(self.t("source_language"))
        self.lang_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lang_label.setStyleSheet(
            f"color: {self.colors['text_dark']}; background: transparent;"
        )
        lang_header_layout.addWidget(self.lang_label)

        # Help button with tooltip
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
        self.lang_combo.setFont(QFont("Segoe UI", 10))
        self.lang_combo.setFixedHeight(40)
        # Allow combo to be reasonably sized but not force layout overflow
        self.lang_combo.setMinimumWidth(160)
        self.lang_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
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
        lang_layout.addWidget(self.lang_combo, alignment=Qt.AlignLeft)

        content_layout.addWidget(self.lang_card)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        # The Start Processing button has been removed from the UI.

        self.log_btn = QPushButton(self.t("view_logs"))
        self.log_btn.setFont(QFont("Segoe UI", 10))
        self.log_btn.setCursor(Qt.PointingHandCursor)
        self.log_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #FFA726;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 14px 25px;
            }}
            QPushButton:hover {{
                background-color: #FB8C00;
            }}
        """
        )
        self.log_btn.clicked.connect(self.open_log_file)
        btn_layout.addWidget(self.log_btn)

        self.reset_btn = QPushButton(self.t("reset"))
        self.reset_btn.setFont(QFont("Segoe UI", 10))
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
                padding: 14px 20px;
            }}
            QPushButton:hover {{
                background-color: {reset_hover};
            }}
        """
        )
        self.reset_btn.clicked.connect(self.reset_ui)
        btn_layout.addWidget(self.reset_btn)

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

    def t(self, key):
        """Get translation for current language"""
        return TRANSLATIONS[self.current_language].get(key, key)

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
        # Update theme button icon
        self.theme_btn.setText("☀️" if self.current_theme == "dark" else "🌙")

        # Update hover colors based on theme
        hover_color = "#E8E8E8" if self.current_theme == "light" else "#3A3A3A"

        # Update main widget background
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

        # Export button uses its dedicated style (below).

        self.log_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #FFA726;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 14px 25px;
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
                padding: 14px 20px;
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
        self.title_label.setText(f"🎬 {self.t('title')}")
        self.subtitle_label.setText(self.t("subtitle"))
        self.file_label.setText(self.t("video_file"))
        self.browse_btn.setText(self.t("browse"))
        self.lang_label.setText(self.t("source_language"))
        self.help_btn.setToolTip(self.t("language_tooltip"))

        # Ensure Export button shows correct label
        self.export_srt_btn.setText(self.t("export_srt"))
        self.log_btn.setText(self.t("view_logs"))
        self.reset_btn.setText(self.t("reset"))
        self.export_srt_btn.setText(self.t("export_srt"))
        self.download_srt_btn.setText(self.t("download_srt"))
        self.logs_label.setText(self.t("processing_logs"))

        # Update status if it's at default
        current_status = self.status_label.text()
        if "Ready" in current_status or "Sẵn sàng" in current_status:
            self.status_label.setText(self.t("ready"))

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
            self.export_srt_btn.setEnabled(True)
            self.status_label.setText(f"{self.t('selected')}: {Path(path).name}")
            self.log.append(f"{self.t('video_selected')}: {path}")

    def handle_start_stop_continue(self):
        """Handle Start/Stop/Continue button click"""
        # This handler is deprecated — Start button was removed. No-op.
        return

    def start_export_srt(self):
        """Start exporting SRT file only (without creating video)"""
        input_path = self.entry.text()
        if not input_path:
            QMessageBox.warning(
                self, self.t("no_file_selected"), self.t("select_export_warning")
            )
            return

        # Disable the Export button during processing
        self.export_srt_btn.setEnabled(False)
        self.status_label.setText(self.t("exporting"))
        self.log.clear()
        self.log.append(f"{self.t('starting_export')}\n")

        # Read selected language from UI
        selected_label = self.lang_combo.currentText()
        lang_map = {
            "English": "en",
            "Japanese": "ja",
            "Chinese (Simplified)": "zh",
            "Korean": "ko",
            "Thai": "th",
            "Indonesian": "id",
        }
        selected_code = lang_map.get(selected_label, "en")

        # Reset pause state
        self.pause_event.set()
        self.is_paused = False
        self.paused_elapsed_time = 0
        self.stop_event.clear()

        # Create holder for subprocess reference
        current_process_holder = [None]
        self.current_process = current_process_holder

        self.worker = threading.Thread(
            target=worker_process_export_srt,
            args=(
                input_path,
                self.msg_queue,
                selected_code,
                self.pause_event,
                current_process_holder,
                self.stop_event,
            ),
        )
        self.worker.daemon = True
        self.worker.start()

        # Start runtime timer
        try:
            self.processing_start_time = time.time()
            self.runtime_timer.start(1000)
        except Exception:
            self.processing_start_time = None

    # `start_processing` removed — button hidden and start functionality deprecated.

    # Pause/resume UI removed — export is single-action; worker still supports pause via events.

    def open_log_file(self):
        """Open the log file with default text editor"""
        log_file = Path("logs/subtitle_generator.log")

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

    def _check_for_srt(self):
        """Check periodically for the presence of `temp_subtitles.srt` and toggle the download button."""
        try:
            srt_path = Path("temp_subtitles.srt")
            exists = srt_path.exists()
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
        """Open a save dialog and copy `temp_subtitles.srt` to the chosen location."""
        srt_path = Path("temp_subtitles.srt")
        if not srt_path.exists():
            QMessageBox.warning(
                self, self.t("srt_not_found"), self.t("srt_not_available")
            )
            return

        # Ask user where to save
        dialog = QFileDialog()
        save_path, _ = dialog.getSaveFileName(
            self,
            self.t("save_subtitle"),
            str(Path.home() / "Downloads" / "temp_subtitles.srt"),
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
                    parts = msg.split(":", 2)
                    ok = parts[1] == "True"
                    out = parts[2] if len(parts) > 2 else ""
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
                    # Restore export button state
                    self.is_paused = False
                    self.export_srt_btn.setEnabled(True)
                    self.export_srt_btn.setText(self.t("export_srt"))
                elif msg == "__PAUSED__":
                    # Handle paused notification from worker
                    pass
                elif msg == "__RESUMED__":
                    # Handle resumed notification from worker
                    pass
                else:
                    # Display raw output to preserve progress bars and formatting
                    self.log.append(msg)
        except queue.Empty:
            pass

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

                    # Terminate subprocess if exists
                    if self.current_process and self.current_process[0] is not None:
                        proc = self.current_process[0]
                        if proc.poll() is None:  # Still running
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                proc.kill()

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
        # Reset export button state
        try:
            self.export_srt_btn.setEnabled(False)
            self.export_srt_btn.setText(self.t("export_srt"))
            self.export_srt_btn.setStyleSheet(
                f"""
                QPushButton {{
                    background-color: {self.colors['primary']};
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 14px 20px;
                }}
                QPushButton:hover {{
                    background-color: {self.colors['primary_hover']};
                }}
            """
            )
        except Exception:
            pass

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
