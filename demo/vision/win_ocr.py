"""
Windows Runtime (WinRT) Japanese / English / Chinese OCR helper for real ADB frames.

Keeps a single warm PowerShell worker process running so repeated OCR calls on live
Android screenshots complete in ~150-300ms with zero extra Python packages.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PS_WORKER_SCRIPT = r'''
$ErrorActionPreference = "SilentlyContinue"
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Add-Type -AssemblyName System.Runtime.WindowsRuntime | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function AwaitOp($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

$targetLang = if ($env:GAKUMAS_OCR_LANG) { $env:GAKUMAS_OCR_LANG } else { "ja" }
$langs = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages
$lang = $langs | Where-Object { $_.LanguageTag -eq $targetLang } | Select-Object -First 1
if (-not $lang) { $lang = $langs | Where-Object { $_.LanguageTag -like "$targetLang*" } | Select-Object -First 1 }
if (-not $lang) { $lang = $langs | Where-Object { $_.LanguageTag -like "en*" } | Select-Object -First 1 }
$engine = if ($lang) { [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang) } else { $null }

Write-Output "READY"
[Console]::Out.Flush()

while (($lineIn = [Console]::In.ReadLine()) -ne $null) {
    $imgPath = $lineIn.Trim()
    if (-not $imgPath) { continue }
    if ($imgPath -eq "__EXIT__") { break }
    if (-not $engine -or -not (Test-Path -LiteralPath $imgPath)) {
        Write-Output '{"text":"","lines":[]}'
        [Console]::Out.Flush()
        continue
    }
    try {
        $file = AwaitOp ([Windows.Storage.StorageFile]::GetFileFromPathAsync($imgPath)) ([Windows.Storage.StorageFile])
        $stream = AwaitOp ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $decoder = AwaitOp ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = AwaitOp ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $res = AwaitOp ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $linesOut = @()
        foreach ($ol in $res.Lines) {
            $wordsOut = @()
            $minX = 999999; $minY = 999999; $maxX = 0; $maxY = 0
            foreach ($w in $ol.Words) {
                $r = $w.BoundingRect
                $x1 = [int][Math]::Floor($r.X)
                $y1 = [int][Math]::Floor($r.Y)
                $x2 = [int][Math]::Ceiling($r.X + $r.Width)
                $y2 = [int][Math]::Ceiling($r.Y + $r.Height)
                if ($x1 -lt $minX) { $minX = $x1 }
                if ($y1 -lt $minY) { $minY = $y1 }
                if ($x2 -gt $maxX) { $maxX = $x2 }
                if ($y2 -gt $maxY) { $maxY = $y2 }
                $wordsOut += @{ text = $w.Text; bbox = @($x1, $y1, $x2, $y2) }
            }
            if ($minX -eq 999999) { $minX = 0; $minY = 0 }
            $linesOut += @{
                text = ($ol.Text -replace '\s+', '')
                raw_text = $ol.Text
                bbox = @($minX, $minY, $maxX, $maxY)
                words = $wordsOut
            }
        }
        $stream.Dispose()
        $jsonOut = @{ text = $res.Text; lines = $linesOut } | ConvertTo-Json -Depth 5 -Compress
        Write-Output $jsonOut
    } catch {
        Write-Output '{"text":"","lines":[]}'
    }
    [Console]::Out.Flush()
}
'''


@dataclass
class OcrLine:
    text: str
    raw_text: str
    bbox: List[int]  # [xmin, ymin, xmax, ymax]
    words: List[Dict[str, Any]]


class PersistentWinOCR:
    """Thread-safe persistent Windows WinRT OCR worker."""

    def __init__(self, language: str = "ja") -> None:
        self.language = language or "ja"
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._script_path: Optional[str] = None
        self._available: bool = os.name == "nt"

    def _ensure_worker(self) -> bool:
        if not self._available:
            return False
        if self._proc is not None and self._proc.poll() is None:
            return True

        fd, script_path = tempfile.mkstemp(prefix="gakumas_ocr_", suffix=".ps1")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(_PS_WORKER_SCRIPT)
        self._script_path = script_path

        try:
            env = os.environ.copy()
            env["GAKUMAS_OCR_LANG"] = self.language
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            assert self._proc.stdout is not None
            ready_line = self._proc.stdout.readline().strip()
            if ready_line != "READY":
                self.close()
                self._available = False
                return False
            return True
        except Exception as exc:
            logger.warning("Failed to start WinRT OCR worker: %s", exc)
            self.close()
            self._available = False
            return False

    def recognize_image_path(self, image_path: str) -> Dict[str, Any]:
        with self._lock:
            if not self._ensure_worker():
                return {"text": "", "lines": []}
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            try:
                self._proc.stdin.write(f"{image_path}\n")
                self._proc.stdin.flush()
                raw = self._proc.stdout.readline()
                if not raw:
                    return {"text": "", "lines": []}
                return json.loads(raw)
            except Exception as exc:
                logger.warning("WinRT OCR read failed: %s", exc)
                self.close()
                return {"text": "", "lines": []}

    def recognize_array(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        fd, img_path = tempfile.mkstemp(prefix="gakumas_frame_", suffix=".png")
        os.close(fd)
        try:
            Image.fromarray(frame_rgb[:, :, :3].astype(np.uint8), mode="RGB").save(img_path)
            return self.recognize_image_path(img_path)
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin and not self._proc.stdin.closed:
                    self._proc.stdin.write("__EXIT__\n")
                    self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        if self._script_path:
            try:
                os.remove(self._script_path)
            except OSError:
                pass
            self._script_path = None


_GLOBAL_WIN_OCRS: Dict[str, PersistentWinOCR] = {}


def get_win_ocr(lang: str = "ja") -> PersistentWinOCR:
    key = (lang or "ja").lower()
    if key not in _GLOBAL_WIN_OCRS:
        inst = PersistentWinOCR(language=key)
        _GLOBAL_WIN_OCRS[key] = inst
        atexit.register(inst.close)
    return _GLOBAL_WIN_OCRS[key]
