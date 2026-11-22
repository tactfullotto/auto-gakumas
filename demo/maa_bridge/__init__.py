"""
MaaFramework Bridge and Controller Layer for Gakumas System.
"""

from demo.maa_bridge.base_controller import (
    BaseController,
    ControllerError,
    ControllerConnectionError,
    ControllerExecutionError,
)
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.adb_controller import AdbController, AdbConfig
from demo.maa_bridge.win32_controller import Win32Controller, Win32Config
from demo.maa_bridge.simulator import MaaPipelineSimulator, PipelineStepResult
from demo.maa_bridge.custom_recognition import (
    BaseGakumasRecognition,
    GakumasSceneRoutingRecognition,
    GakumasBattleRecognition,
    GakumasProduceRecognition,
    GakumasRewardRecognition,
    GakumasShopRecognition,
    GakumasPopupRecognition,
    GakumasSkipRecognition,
)
from demo.maa_bridge.custom_action import GakumasDecisionAction
from demo.maa_bridge.runner import GakumasRunner, RunnerState, StepResult

__all__ = [
    "BaseController",
    "ControllerError",
    "ControllerConnectionError",
    "ControllerExecutionError",
    "MockController",
    "AdbController",
    "AdbConfig",
    "Win32Controller",
    "Win32Config",
    "MaaPipelineSimulator",
    "PipelineStepResult",
    "BaseGakumasRecognition",
    "GakumasSceneRoutingRecognition",
    "GakumasBattleRecognition",
    "GakumasProduceRecognition",
    "GakumasRewardRecognition",
    "GakumasShopRecognition",
    "GakumasPopupRecognition",
    "GakumasSkipRecognition",
    "GakumasDecisionAction",
    "GakumasRunner",
    "RunnerState",
    "StepResult",
]
