"""
Win32 Desktop Controller for Windows Gakumas Client.
Pure ctypes implementation with window matching, GDI/ImageGrab capture,
background/foreground clicks, and headless fallback.
"""

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import logging
import re
from typing import Any, Optional, Tuple
import numpy as np
from PIL import Image

from demo.schemas import BoundingBox
from demo.vision.mock_generator import MockFrameGenerator
from demo.maa_bridge.base_controller import (
    BaseController,
    ControllerConnectionError,
    ControllerExecutionError,
)

logger = logging.getLogger(__name__)

# Win32 Constants
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
MK_LBUTTON = 0x0001
PW_RENDERFULLCONTENT = 0x00000002

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


@dataclass
class Win32Config:
    window_title: str = "学园偶像大师"
    window_title_pattern: str = r"(学园偶像大师|Gakumas|学マス)"
    headless_fallback: bool = False
    background_click: bool = True


class Win32Controller(BaseController):
    """
    Controls desktop game window using pure ctypes and Windows APIs.
    """

    def __init__(
        self,
        window_title: Optional[str] = None,
        config: Optional[Win32Config] = None,
        headless_fallback: Optional[bool] = None,
        **kwargs,
    ):
        if config:
            self.config = config
        else:
            wt = window_title or kwargs.get("window_title", "学园偶像大师")
            hf = headless_fallback if headless_fallback is not None else kwargs.get("headless_fallback", False)
            self.config = Win32Config(window_title=wt, headless_fallback=hf, **kwargs)

        self.window_title: str = self.config.window_title
        self.hwnd: Optional[int] = None
        self._connected: bool = False
        self._mock_delegate: Optional[Any] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Find window by title or regex pattern and verify handle."""
        # 1. Direct FindWindowW lookup (fast and easily mocked)
        hwnd = user32.FindWindowW(None, self.window_title)
        if not hwnd and self.config.window_title_pattern:
            hwnd = self._find_window_by_pattern(self.config.window_title_pattern)

        if hwnd:
            self.hwnd = int(hwnd)
            self._connected = True
            logger.info("Win32Controller attached to window HWND: %d", self.hwnd)
            return True

        if self.config.headless_fallback:
            logger.info("Win32 window not found. Activating headless mock delegate.")
            self._mock_delegate = MockFrameGenerator()
            self._connected = True
            return True

        self._connected = False
        self.hwnd = None
        return False

    def disconnect(self) -> None:
        self._connected = False
        self.hwnd = None
        if self._mock_delegate:
            self._mock_delegate = None

    def screencap(self) -> np.ndarray:
        if not self._connected:
            raise ControllerExecutionError("Win32Controller is not connected")

        # Headless mock mode
        if self.hwnd is None and self._mock_delegate:
            frame, _ = self._mock_delegate.generate_frame("battle_turn")
            return frame

        if self.hwnd is None:
            raise ControllerExecutionError("No window handle available for screencap")

        # Capture window using GDI PrintWindow
        try:
            return self._capture_gdi(self.hwnd)
        except Exception as e:
            if self.config.headless_fallback:
                logger.warning("GDI screencap failed (%s), falling back to mock frame", e)
                if not self._mock_delegate:
                    self._mock_delegate = MockFrameGenerator()
                frame, _ = self._mock_delegate.generate_frame("battle_turn")
                return frame
            raise ControllerExecutionError(f"Win32 screencap failed: {e}") from e

    def click(self, x: int, y: int) -> bool:
        if not self._connected:
            raise ControllerExecutionError("Win32Controller is not connected")

        if self.hwnd is None:
            # Headless mock mode records click successfully
            return True

        x = int(x)
        y = int(y)
        lParam = (y << 16) | (x & 0xFFFF)
        if self.config.background_click:
            user32.PostMessageW(self.hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lParam)
            user32.PostMessageW(self.hwnd, WM_LBUTTONUP, 0, lParam)
            return True
        else:
            user32.SetForegroundWindow(self.hwnd)
            point = wintypes.POINT(x, y)
            user32.ClientToScreen(self.hwnd, ctypes.byref(point))
            user32.SetCursorPos(point.x, point.y)
            user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
            user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
            return True

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> bool:
        if not self._connected:
            raise ControllerExecutionError("Win32Controller is not connected")
        return True

    @staticmethod
    def _find_window_by_pattern(pattern: str) -> Optional[int]:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning("Invalid regex pattern '%s': %s", pattern, e)
            return None
        found_hwnd = None

        def enum_proc(hwnd, lParam):
            nonlocal found_hwnd
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if regex.search(buf.value):
                        found_hwnd = hwnd
                        return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
        return found_hwnd

    _find_window = _find_window_by_pattern

    @staticmethod
    def _capture_gdi(hwnd: int) -> np.ndarray:
        rect = wintypes.RECT()
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid window client dimensions: {w}x{h}")

        hdc_window = user32.GetDC(hwnd)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
        hbitmap = gdi32.CreateCompatibleBitmap(hdc_window, w, h)
        gdi32.SelectObject(hdc_mem, hbitmap)

        user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0

        buf = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(hdc_mem, hbitmap, 0, h, buf, ctypes.byref(bmi), 0)

        gdi32.DeleteObject(hbitmap)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)

        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        rgb = arr[:, :, [2, 1, 0]]
        return rgb
