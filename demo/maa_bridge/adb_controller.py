"""
ADB Network Controller for Android Emulator (192.168.0.115:5555).
Supports multi-tiered auto-discovery of adb.exe, dual-backend (MaaFramework & direct CLI),
root authentication handling, and graceful offline fallback.
"""

from dataclasses import dataclass, field
import io
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from demo.maa_bridge.base_controller import (
    BaseController,
    ControllerConnectionError,
    ControllerExecutionError,
)

logger = logging.getLogger(__name__)

# Try importing MaaFramework AdbController
try:
    from maa.controller import AdbController as MaaAdbController
    _HAS_MAA = True
except ImportError:
    MaaAdbController = None
    _HAS_MAA = False


@dataclass
class AdbConfig:
    host: str = "192.168.0.115"
    port: int = 5555
    adb_path: Optional[str] = None
    root_pwd: str = "root/114514"
    auto_root: bool = False
    connect_timeout: float = 3.0
    command_timeout: float = 8.0
    retry_count: int = 3
    prefer_maa: bool = True
    fallback_to_mock: bool = False


class AdbController(BaseController):
    """
    Connects to an Android emulator via ADB over TCP/IP.
    Supports 192.168.0.115:5555 with automatic ADB discovery and CLI/Maa fallbacks.
    """

    KNOWN_ADB_PATHS = [
        os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
        r"C:\Program Files\Netease\MuMu\nx_main\adb.exe",
        r"C:\Program Files\Netease\MuMu\nx_device\15.0\shell\adb.exe",
        r"C:\Program Files\Netease\MuMuPlayer-12.0\shell\adb.exe",
        r"C:\Program Files\Netease\MuMuPlayerGlobal-12.0\shell\adb.exe",
        r"C:\Program Files\MuMuPlayer-12.0\shell\adb.exe",
        r"C:\LDPlayer\LDPlayer9\adb.exe",
        r"C:\Program Files\Nox\bin\nox_adb.exe",
        r"C:\Program Files (x86)\Nox\bin\nox_adb.exe",
        r"C:\Program Files\Microvirt\MEmu\adb.exe",
    ]

    def __init__(
        self,
        host: Optional[Any] = None,
        port: Optional[int] = None,
        config: Optional[AdbConfig] = None,
        **kwargs,
    ):
        if isinstance(host, AdbConfig) and config is None:
            config = host
            host = None
        if config:
            self.config = config
        else:
            c_host = host if host is not None else kwargs.get("host", "192.168.0.115")
            c_port = port if port is not None else kwargs.get("port", 5555)
            self.config = AdbConfig(host=c_host, port=c_port, **kwargs)

        self.host: str = self.config.host
        self.port: int = self.config.port
        self.address: str = f"{self.host}:{self.port}"
        self.adb_path: str = self._resolve_adb_path(self.config.adb_path) or "adb"

        self._connected: bool = False
        self._is_connected: bool = False
        self._use_maa: bool = False
        self.maa_controller: Optional[Any] = None
        self._mock_delegate: Optional[Any] = None
        self._last_frame_size: Optional[Tuple[int, int]] = None

    @classmethod
    def _resolve_adb_path(cls, user_path: Optional[str] = None) -> Optional[str]:
        """Locate ADB executable across user argument, PATH, and common installation directories."""
        if user_path and Path(user_path).exists():
            return str(Path(user_path).resolve())
        env_adb = os.getenv("ADB_PATH")
        if env_adb and Path(env_adb).exists():
            return str(Path(env_adb).resolve())
        which_adb = shutil.which("adb")
        if which_adb:
            return which_adb
        for p in cls.KNOWN_ADB_PATHS:
            if Path(p).exists():
                return p
        return None

    @property
    def is_connected(self) -> bool:
        return self._connected or self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._connected = value
        self._is_connected = value

    def connect(self) -> bool:
        """Establish network ADB connection with timeout and root authentication check."""
        # 1. Non-blocking TCP socket pre-check to prevent hanging on offline devices
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.config.connect_timeout)
                s.connect((self.host, self.port))
        except (socket.timeout, OSError) as e:
            logger.warning("Cannot reach %s: %s", self.address, e)
            if self.config.fallback_to_mock:
                from demo.maa_bridge.mock_controller import MockController
                self._mock_delegate = MockController()
                self._mock_delegate.connect()
                self.is_connected = True
                return True
            self.is_connected = False
            return False

        # 2. Run adb connect
        try:
            res = subprocess.run(
                [self.adb_path, "connect", self.address],
                capture_output=True,
                text=True,
                timeout=self.config.connect_timeout + 2.0,
            )
            stdout_lower = (res.stdout or "").lower()
            if "connected" not in stdout_lower and "already connected" not in stdout_lower:
                logger.warning("ADB connect failed: %s", res.stdout.strip())
                self.is_connected = False
                return False
        except Exception as e:
            logger.warning("Error running adb connect: %s", e)
            self.is_connected = False
            return False

        # 3. Handle root if requested
        if self.config.auto_root:
            self._setup_root()

        # 4. Attempt MaaFramework binding if available and preferred
        if _HAS_MAA and self.config.prefer_maa and MaaAdbController:
            try:
                self.maa_controller = MaaAdbController(adb_path=self.adb_path, address=self.address)
                job = self.maa_controller.post_connection()
                job.wait()
                if job.succeeded and self.maa_controller.connected:
                    self._use_maa = True
                    logger.info("Successfully connected via MaaAdbController to %s", self.address)
            except Exception as e:
                logger.info("MaaAdbController init failed, falling back to CLI subprocess mode: %s", e)
                self._use_maa = False

        self.is_connected = True
        return True

    def _setup_root(self) -> None:
        """Attempt root privilege elevation via adb root or su password."""
        try:
            subprocess.run([self.adb_path, "-s", self.address, "root"], capture_output=True, timeout=3.0)
            time.sleep(0.5)
            subprocess.run([self.adb_path, "connect", self.address], capture_output=True, timeout=2.0)
        except Exception:
            pass

    def disconnect(self) -> None:
        self.is_connected = False
        if self._use_maa and self.maa_controller:
            try:
                self.maa_controller.post_inactive()
            except Exception:
                pass
            self.maa_controller = None
        if self._mock_delegate:
            self._mock_delegate.disconnect()
            self._mock_delegate = None

    def _reconnect_cli(self) -> bool:
        """Re-issue `adb connect` via CLI to recover dropped TCP/IP sessions or restarted adb servers."""
        try:
            res = subprocess.run(
                [self.adb_path, "connect", self.address],
                capture_output=True,
                text=True,
                timeout=self.config.connect_timeout + 2.0,
            )
            out = (res.stdout or "").lower()
            return "connected" in out or "already connected" in out
        except Exception as e:
            logger.debug("ADB CLI reconnect failed: %s", e)
            return False

    def _get_or_init_mock_delegate(self) -> Any:
        """Lazily create and connect a MockController fallback delegate."""
        if self._mock_delegate is None:
            from demo.maa_bridge.mock_controller import MockController
            self._mock_delegate = MockController()
            self._mock_delegate.connect()
        return self._mock_delegate

    def _run_cli_input(self, args: List[str], action_label: str) -> bool:
        """Execute an `adb shell input ...` command with one automatic reconnect retry on failure."""
        cmd = [self.adb_path, "-s", self.address, "shell", "input", *args]
        for attempt in range(2):
            try:
                res = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self.config.command_timeout,
                )
                if res.returncode == 0:
                    return True
                if attempt == 0 and self._reconnect_cli():
                    continue
                return False
            except subprocess.TimeoutExpired:
                logger.warning("ADB %s timed out (%s)", action_label, " ".join(args))
                if attempt == 0 and self._reconnect_cli():
                    continue
                return False
        return False

    def screencap(self) -> np.ndarray:
        if not self.is_connected:
            raise ControllerExecutionError("AdbController is not connected")

        if self._mock_delegate:
            return self._mock_delegate.screencap()

        # 1. Try MaaController screencap
        if self._use_maa and self.maa_controller:
            try:
                job = self.maa_controller.post_screencap()
                job.wait()
                if job.succeeded:
                    arr = self.maa_controller.cached_image
                    if arr is not None and arr.size > 0:
                        if arr.ndim == 3 and arr.shape[2] == 3:
                            arr = arr[:, :, ::-1].copy()
                        self._last_frame_size = (int(arr.shape[1]), int(arr.shape[0]))
                        return arr
            except Exception as e:
                logger.warning("Maa screencap failed, falling back to CLI: %s", e)

        # 2. Direct CLI screencap via exec-out (with one reconnect + retry on transient adb drop)
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                res = subprocess.run(
                    [self.adb_path, "-s", self.address, "exec-out", "screencap", "-p"],
                    capture_output=True,
                    timeout=self.config.command_timeout,
                )
                if res.returncode == 0 and len(res.stdout) > 0:
                    img = Image.open(io.BytesIO(res.stdout)).convert("RGB")
                    arr = np.array(img, dtype=np.uint8)
                    self._last_frame_size = (int(arr.shape[1]), int(arr.shape[0]))
                    return arr
                if attempt == 0 and self._reconnect_cli():
                    continue
            except Exception as e:
                last_err = e
                if attempt == 0 and self._reconnect_cli():
                    continue
                break

        if self.config.fallback_to_mock:
            logger.warning("ADB screencap unavailable after retry (%s); delegating to MockController.", last_err or "empty/error output")
            return self._get_or_init_mock_delegate().screencap()

        if last_err is not None:
            raise ControllerExecutionError(f"ADB screencap failed: {last_err}") from last_err
        raise ControllerExecutionError("Failed to obtain screencap from ADB device")

    def click(self, x: int, y: int) -> bool:
        if not self.is_connected:
            raise ControllerExecutionError("AdbController is not connected")

        x = int(x)
        y = int(y)
        if self._last_frame_size is not None:
            fw, fh = self._last_frame_size
            if (fw, fh) != (1080, 1920) and (x > fw or y > fh):
                x = int(round(x * fw / 1080.0))
                y = int(round(y * fh / 1920.0))

        if self._mock_delegate:
            return self._mock_delegate.click(x, y)

        if self._use_maa and self.maa_controller:
            try:
                job = self.maa_controller.post_click(x, y)
                job.wait()
                if job.succeeded:
                    return True
            except Exception:
                pass

        # Fallback to adb shell input tap x y
        return self._run_cli_input(["tap", str(x), str(y)], f"click({x},{y})")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> bool:
        if not self.is_connected:
            raise ControllerExecutionError("AdbController is not connected")

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        duration_ms = int(duration_ms)
        if self._last_frame_size is not None:
            fw, fh = self._last_frame_size
            if (fw, fh) != (1080, 1920) and (x1 > fw or y1 > fh or x2 > fw or y2 > fh):
                x1 = int(round(x1 * fw / 1080.0))
                y1 = int(round(y1 * fh / 1920.0))
                x2 = int(round(x2 * fw / 1080.0))
                y2 = int(round(y2 * fh / 1920.0))

        if self._mock_delegate:
            return self._mock_delegate.swipe(x1, y1, x2, y2, duration_ms)

        if self._use_maa and self.maa_controller:
            try:
                job = self.maa_controller.post_swipe(x1, y1, x2, y2, duration_ms)
                job.wait()
                if job.succeeded:
                    return True
            except Exception:
                pass

        # Fallback to adb shell input swipe x1 y1 x2 y2 duration_ms
        return self._run_cli_input(
            ["swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            f"swipe({x1},{y1}->{x2},{y2})",
        )
