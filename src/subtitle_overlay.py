"""
Subtitle Overlay Module for Real-time Video Display

This module handles the creation and display of subtitle overlays on screen
or video streams using OpenCV and supports various subtitle formatting options.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime, timedelta
import threading
import time
from dataclasses import dataclass
from loguru import logger
import textwrap

import os


@dataclass
class SubtitleLine:
    """Represents a single subtitle line with timing and styling information."""

    text: str
    start_time: datetime
    duration: float
    position: Tuple[int, int] = (0, 0)
    font_size: int = 24
    color: Tuple[int, int, int] = (255, 255, 255)  # RGB
    background_color: Tuple[int, int, int] = (0, 0, 0)  # RGB
    background_opacity: float = 0.7
    fade_in: float = 0.5
    fade_out: float = 0.5


class SubtitleOverlay:
    """Manages subtitle display overlays for real-time video streams."""

    def __init__(self, config: dict):
        """
        Initialize the SubtitleOverlay with configuration settings.

        Args:
            config: Subtitle configuration dictionary
        """
        self.config = config["subtitles"]

        # Display settings
        self.font_family = self.config.get("font_family", "Arial")
        self.font_size = self.config.get("font_size", 24)
        self.font_color = tuple(self.config.get("font_color", [255, 255, 255]))
        self.background_color = tuple(self.config.get("background_color", [0, 0, 0]))
        self.background_opacity = self.config.get("background_opacity", 0.7)

        # Positioning
        self.position = self.config.get("position", "bottom")
        self.margin = self.config.get("margin", 50)
        self.max_width = self.config.get("max_width", 0.8)

        # Timing
        self.display_duration = self.config.get("display_duration", 5.0)
        self.fade_duration = self.config.get("fade_duration", 0.5)

        # Text formatting
        self.max_chars_per_line = self.config.get("max_chars_per_line", 50)
        self.max_lines = self.config.get("max_lines", 2)
        self.text_align = self.config.get("text_align", "center")

        # Active subtitles
        self.active_subtitles: List[SubtitleLine] = []
        self.subtitle_lock = threading.Lock()

        # Screen/window settings
        self.screen_width = 1920
        self.screen_height = 1080
        self.window_name = "Vietnamese Subtitles"
        self.overlay_window = None
        self.is_fullscreen = False

        # Font loading
        self.font_path = self._find_system_font()

        logger.info("SubtitleOverlay initialized")

    def _find_system_font(self) -> Optional[str]:
        """
        Find a suitable system font for subtitle rendering.

        Returns:
            Path to font file or None if not found
        """
        # Common font paths for different systems
        font_paths = [
            # Windows
            f"C:/Windows/Fonts/{self.font_family}.ttf",
            f"C:/Windows/Fonts/{self.font_family}bd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            # Linux
            f"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            f"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            # macOS
            f"/System/Library/Fonts/{self.font_family}.ttc",
            "/System/Library/Fonts/Arial.ttf",
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                logger.info(f"Using font: {font_path}")
                return font_path

        logger.warning("No suitable system font found, using default")
        return None

    def add_subtitle(self, text: str, duration: Optional[float] = None) -> None:
        """
        Add a new subtitle to be displayed.

        Args:
            text: The subtitle text to display
            duration: Display duration in seconds (uses default if None)
        """
        if not text.strip():
            return

        # Format text to fit within line limits
        formatted_lines = self._format_text(text)

        with self.subtitle_lock:
            # Create subtitle line
            subtitle = SubtitleLine(
                text="\n".join(formatted_lines),
                start_time=datetime.now(),
                duration=duration or self.display_duration,
                font_size=self.font_size,
                color=self.font_color,
                background_color=self.background_color,
                background_opacity=self.background_opacity,
                fade_in=self.fade_duration,
                fade_out=self.fade_duration,
            )

            # Add to active subtitles
            self.active_subtitles.append(subtitle)

            # Clean up old subtitles
            self._cleanup_expired_subtitles()

        logger.info(f"Added subtitle: {text[:50]}...")

    def _format_text(self, text: str) -> List[str]:
        """
        Format text to fit within line and character limits.

        Args:
            text: Input text to format

        Returns:
            List of formatted text lines
        """
        # Split long text into multiple lines
        wrapped_lines = textwrap.wrap(
            text,
            width=self.max_chars_per_line,
            break_long_words=False,
            break_on_hyphens=True,
        )

        # Limit to maximum number of lines
        if len(wrapped_lines) > self.max_lines:
            wrapped_lines = wrapped_lines[: self.max_lines - 1]
            # Add ellipsis to indicate truncation
            if wrapped_lines:
                wrapped_lines[-1] += "..."

        return wrapped_lines

    def _cleanup_expired_subtitles(self) -> None:
        """Remove expired subtitles from the active list."""
        current_time = datetime.now()
        self.active_subtitles = [
            subtitle
            for subtitle in self.active_subtitles
            if (current_time - subtitle.start_time).total_seconds()
            < subtitle.duration + subtitle.fade_out
        ]

    def create_overlay_window(self) -> bool:
        """
        Create a transparent overlay window for subtitle display.

        Returns:
            True if window created successfully, False otherwise
        """
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)

            # Create transparent background
            overlay = np.zeros(
                (self.screen_height, self.screen_width, 3), dtype=np.uint8
            )
            cv2.imshow(self.window_name, overlay)

            self.overlay_window = self.window_name
            self.is_fullscreen = True

            logger.info("Overlay window created")
            return True

        except Exception as e:
            logger.error(f"Failed to create overlay window: {e}")
            return False

    def render_subtitles(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render active subtitles onto a frame or create overlay.

        Args:
            frame: Input video frame (optional, creates overlay if None)

        Returns:
            Frame with rendered subtitles
        """
        if frame is None:
            # Create transparent overlay
            frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

        frame_height, frame_width = frame.shape[:2]

        with self.subtitle_lock:
            current_time = datetime.now()

            # Render each active subtitle
            for subtitle in self.active_subtitles:
                elapsed_time = (current_time - subtitle.start_time).total_seconds()

                # Skip if not yet started or already finished
                if (
                    elapsed_time < 0
                    or elapsed_time > subtitle.duration + subtitle.fade_out
                ):
                    continue

                # Calculate opacity based on fade in/out
                opacity = self._calculate_opacity(elapsed_time, subtitle)

                if opacity > 0:
                    frame = self._render_subtitle_text(
                        frame, subtitle, opacity, frame_width, frame_height
                    )

        return frame

    def _calculate_opacity(self, elapsed_time: float, subtitle: SubtitleLine) -> float:
        """
        Calculate subtitle opacity based on timing and fade settings.

        Args:
            elapsed_time: Time elapsed since subtitle start
            subtitle: Subtitle line object

        Returns:
            Opacity value between 0.0 and 1.0
        """
        if elapsed_time < 0:
            return 0.0

        # Fade in
        if elapsed_time < subtitle.fade_in:
            return elapsed_time / subtitle.fade_in

        # Full display
        if elapsed_time < subtitle.duration - subtitle.fade_out:
            return 1.0

        # Fade out
        if elapsed_time < subtitle.duration:
            fade_progress = (subtitle.duration - elapsed_time) / subtitle.fade_out
            return max(0.0, fade_progress)

        return 0.0

    def _render_subtitle_text(
        self,
        frame: np.ndarray,
        subtitle: SubtitleLine,
        opacity: float,
        frame_width: int,
        frame_height: int,
    ) -> np.ndarray:
        """
        Render subtitle text onto the frame with proper positioning and styling.

        Args:
            frame: Video frame to render on
            subtitle: Subtitle to render
            opacity: Current opacity level
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Frame with rendered subtitle
        """
        # Split text into lines
        lines = subtitle.text.split("\n")

        # Calculate text dimensions
        font_scale = subtitle.font_size / 24.0  # Base scale
        thickness = max(1, int(font_scale * 2))

        # Measure text size
        line_heights = []
        line_widths = []

        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            line_widths.append(text_width)
            line_heights.append(text_height + baseline)

        max_text_width = max(line_widths) if line_widths else 0
        total_text_height = sum(line_heights)

        # Calculate position
        x_pos, y_pos = self._calculate_text_position(
            max_text_width, total_text_height, frame_width, frame_height
        )

        # Render background
        if self.background_opacity > 0:
            self._render_text_background(
                frame,
                x_pos,
                y_pos,
                max_text_width,
                total_text_height,
                subtitle.background_color,
                opacity * self.background_opacity,
            )

        # Render text lines
        current_y = y_pos
        for i, line in enumerate(lines):
            if not line.strip():
                current_y += line_heights[i] if i < len(line_heights) else 20
                continue

            # Calculate line position
            line_x = x_pos
            if self.text_align == "center":
                line_x = x_pos + (max_text_width - line_widths[i]) // 2
            elif self.text_align == "right":
                line_x = x_pos + max_text_width - line_widths[i]

            # Apply opacity to text color
            text_color = tuple(int(c * opacity) for c in subtitle.color)

            # Render text with outline for better visibility
            outline_thickness = max(1, thickness + 1)

            # Draw outline
            cv2.putText(
                frame,
                line,
                (line_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                outline_thickness,
                cv2.LINE_AA,
            )

            # Draw main text
            cv2.putText(
                frame,
                line,
                (line_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

            current_y += line_heights[i] if i < len(line_heights) else 20

        return frame

    def _calculate_text_position(
        self, text_width: int, text_height: int, frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """
        Calculate the position for subtitle text based on configuration.

        Args:
            text_width: Width of the text
            text_height: Height of the text
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            (x, y) position for the text
        """
        # Calculate maximum width based on frame
        max_text_width = int(frame_width * self.max_width)
        text_width = min(text_width, max_text_width)

        # Horizontal positioning (always centered)
        x_pos = (frame_width - text_width) // 2

        # Vertical positioning
        if self.position == "top":
            y_pos = self.margin + text_height
        elif self.position == "center":
            y_pos = (frame_height + text_height) // 2
        else:  # bottom
            y_pos = frame_height - self.margin

        return x_pos, y_pos

    def _render_text_background(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int],
        opacity: float,
    ) -> None:
        """
        Render semi-transparent background for subtitle text.

        Args:
            frame: Video frame to render on
            x, y: Background position
            width, height: Background dimensions
            color: Background color (RGB)
            opacity: Background opacity
        """
        # Create background overlay
        overlay = frame.copy()

        # Add padding around text
        padding = 10
        bg_x1 = max(0, x - padding)
        bg_y1 = max(0, y - height - padding)
        bg_x2 = min(frame.shape[1], x + width + padding)
        bg_y2 = min(frame.shape[0], y + padding)

        # Draw background rectangle
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Blend with original frame
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

    def start_overlay_display(self) -> None:
        """Start the overlay display in a separate thread."""
        if not self.create_overlay_window():
            logger.error("Failed to create overlay window")
            return

        display_thread = threading.Thread(
            target=self._overlay_display_loop, daemon=True
        )
        display_thread.start()
        logger.info("Overlay display started")

    def _overlay_display_loop(self) -> None:
        """Main loop for overlay display."""
        while True:
            try:
                # Create overlay frame
                overlay_frame = self.render_subtitles()

                # Display frame
                if self.overlay_window:
                    cv2.imshow(self.window_name, overlay_frame)

                # Check for key press (ESC to exit)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break

                # Small delay to prevent high CPU usage
                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error in overlay display loop: {e}")
                break

        self.close_overlay_window()

    def close_overlay_window(self) -> None:
        """Close the overlay window and cleanup resources."""
        if self.overlay_window:
            cv2.destroyWindow(self.window_name)
            self.overlay_window = None
            logger.info("Overlay window closed")

    def clear_subtitles(self) -> None:
        """Clear all active subtitles."""
        with self.subtitle_lock:
            self.active_subtitles.clear()
        logger.info("All subtitles cleared")

    def get_active_subtitle_count(self) -> int:
        """
        Get the number of currently active subtitles.

        Returns:
            Number of active subtitles
        """
        with self.subtitle_lock:
            return len(self.active_subtitles)

    def set_screen_size(self, width: int, height: int) -> None:
        """
        Set the screen/window size for subtitle positioning.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self.screen_width = width
        self.screen_height = height
        logger.info(f"Screen size set to {width}x{height}")

    def cleanup(self) -> None:
        """Cleanup resources and close windows."""
        self.close_overlay_window()
        self.clear_subtitles()
        logger.info("SubtitleOverlay cleanup completed")


def test_subtitle_overlay():
    """Test function for the SubtitleOverlay class."""
    import yaml
    import time

    # Test configuration
    config = {
        "subtitles": {
            "font_family": "Arial",
            "font_size": 32,
            "font_color": [255, 255, 0],  # Yellow
            "background_color": [0, 0, 0],
            "background_opacity": 0.8,
            "position": "bottom",
            "margin": 100,
            "max_width": 0.9,
            "display_duration": 3.0,
            "fade_duration": 0.5,
            "max_chars_per_line": 60,
            "max_lines": 2,
            "text_align": "center",
        }
    }

    # Create subtitle overlay
    overlay = SubtitleOverlay(config)

    # Start overlay display
    overlay.start_overlay_display()

    # Add test subtitles
    test_subtitles = [
        "Xin chào! Đây là phụ đề tiếng Việt thời gian thực.",
        "This is a real-time Vietnamese subtitle system.",
        "Hệ thống nhận diện giọng nói và dịch tự động sang tiếng Việt.",
        "Testing longer subtitle text that should wrap to multiple lines automatically.",
    ]

    try:
        for i, text in enumerate(test_subtitles):
            overlay.add_subtitle(text)
            print(f"Added subtitle {i + 1}: {text}")
            time.sleep(4)  # Wait between subtitles

        # Keep display running
        time.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        overlay.cleanup()


if __name__ == "__main__":
    test_subtitle_overlay()
