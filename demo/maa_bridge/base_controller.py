"""
Abstract Base Controller Protocol for Gakumas System.
Defines unified interface for Mock, ADB, and Win32 backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from demo.schemas import BoundingBox


class ControllerError(Exception):
    """Base exception for controller operations."""
    pass


class ControllerConnectionError(ControllerError):
    """Raised when controller connection fails."""
    pass


class ControllerExecutionError(ControllerError):
    """Raised when click, swipe, or capture command fails."""
    pass


class BaseController(ABC):
    """
    Abstract Base Class for hardware and simulated game controllers.
    Guarantees consistent screencap and input semantics across all run modes.
    """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if controller is actively connected."""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to target device, window, or mock environment."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect and release resources."""
        pass

    @abstractmethod
    def screencap(self) -> np.ndarray:
        """
        Capture current frame as RGB uint8 array (shape: (H, W, 3)).
        Raises ControllerExecutionError if not connected or capture fails.
        """
        pass

    @abstractmethod
    def click(self, x: int, y: int) -> bool:
        """Send single tap/click event at (x, y). Returns True on success."""
        pass

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> bool:
        """Send swipe/drag event from (x1, y1) to (x2, y2). Returns True on success."""
        pass

    @property
    def resolution(self) -> Tuple[int, int]:
        """Return standard target resolution (width, height), default (1080, 1920)."""
        return (1080, 1920)

    def click_bbox(self, bbox: BoundingBox) -> bool:
        """Convenience method to click center of a bounding box."""
        cx, cy = bbox.center()
        return self.click(cx, cy)

    def long_press(self, x: int, y: int, duration_ms: int = 900) -> bool:
        """Send press-and-hold gesture at (x, y) for `duration_ms`. Returns True on success."""
        return self.swipe(int(x), int(y), int(x), int(y), duration_ms=int(duration_ms))

    def screencap_pil(self) -> Image.Image:
        """Convenience method returning PIL Image from screencap."""
        arr = self.screencap()
        return Image.fromarray(arr)

    def __enter__(self) -> "BaseController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
