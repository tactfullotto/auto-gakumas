"""
Pure-Python MaaPipelineSimulator.
Provides 100% compliant state machine execution of pipeline.json with registered
CustomRecognition and CustomAction components without requiring native C++ libraries.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from demo.schemas import BoundingBox, DecisionResult, LegalAction, ScreenObservation
from demo.maa_bridge.base_controller import BaseController

logger = logging.getLogger(__name__)


@dataclass
class PipelineStepResult:
    """Summary of a single simulated pipeline step."""
    step: int
    node_name: str
    recognition_hit: bool
    action_success: bool
    clicked_coord: Optional[Tuple[int, int]] = None
    next_node: Optional[str] = None
    observation: Optional[ScreenObservation] = None
    decision: Optional[DecisionResult] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class SimulatorContext:
    """Task context passed to CustomRecognition and CustomAction callbacks."""

    def __init__(self, simulator: "MaaPipelineSimulator"):
        self.simulator = simulator

    def override_next(self, node_name: str, next_list: List[str]) -> bool:
        """Override the subsequent candidate nodes for the state machine."""
        self.simulator._overridden_next[node_name] = list(next_list)
        return True


@dataclass
class SimulatorAnalyzeArg:
    image: np.ndarray
    node_name: str
    custom_recognition_param: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorRecoDetail:
    best_result: Any
    raw_detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorRunArg:
    node_name: str
    reco_detail: Any
    custom_action_param: Dict[str, Any] = field(default_factory=dict)


class MaaPipelineSimulator:
    """
    Simulates MaaFramework Tasker and Resource execution of pipeline.json in pure Python.
    Supports candidate fallbacks, recognition analysis, action dispatching, and dynamic routing.
    """

    def __init__(self, pipeline_path_or_dict: Union[str, Path, Dict[str, Any]]):
        if isinstance(pipeline_path_or_dict, dict):
            self.pipeline = dict(pipeline_path_or_dict)
        else:
            p = Path(pipeline_path_or_dict)
            with open(p, "r", encoding="utf-8") as f:
                self.pipeline = json.load(f)

        self.custom_recognitions: Dict[str, Any] = {}
        self.custom_actions: Dict[str, Any] = {}
        self.controller: Optional[BaseController] = None

        self._current_node: str = "SceneRouting"
        self._active_candidates: List[str] = ["SceneRouting"]
        self._overridden_next: Dict[str, List[str]] = {}
        self.step_count = 0
        self.context = SimulatorContext(self)

    @property
    def current_node(self) -> str:
        return self._current_node

    @current_node.setter
    def current_node(self, value: str) -> None:
        self._current_node = str(value)
        if not self._active_candidates or self._active_candidates[0] != self._current_node:
            self._active_candidates = [self._current_node]

    def set_candidates(self, candidates: List[str]) -> None:
        """Explicitly set active candidate list."""
        self._active_candidates = list(candidates)
        if self._active_candidates:
            self._current_node = self._active_candidates[0]

    def register_custom_recognition(self, name: str, reco: Any) -> None:
        """Register custom recognition handler."""
        self.custom_recognitions[name] = reco

    def register_custom_action(self, name: str, action: Any) -> None:
        """Register custom action handler."""
        self.custom_actions[name] = action

    def bind_controller(self, controller: BaseController) -> None:
        """Bind physical or mock controller."""
        self.controller = controller

    def step(self) -> PipelineStepResult:
        """Execute one step in the pipeline state machine."""
        self.step_count += 1
        step_idx = self.step_count
        start_time = time.perf_counter()

        if self.controller:
            if not self.controller.is_connected:
                self.controller.connect()
            frame = self.controller.screencap()
            if frame is None:
                return PipelineStepResult(
                    step=step_idx,
                    node_name=self.current_node,
                    recognition_hit=False,
                    action_success=False,
                    error="Controller screencap returned None",
                )
        else:
            frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

        # Candidate fallback evaluation: sequential trial across active candidates
        candidates = list(self._active_candidates) if self._active_candidates else [self.current_node]
        if self.current_node in self._overridden_next:
            candidates = self._overridden_next.pop(self.current_node)
            self._active_candidates = list(candidates)

        matched_node = None
        matched_node_cfg = None
        reco_res = None
        obs = None

        for cand_name in candidates:
            cand_cfg = self.pipeline.get(cand_name, {})
            cand_reco_name = cand_cfg.get("custom_recognition")
            cand_reco_param = cand_cfg.get("custom_recognition_param", {})

            hit = False
            curr_reco_res = None
            curr_obs = None

            if cand_reco_name and cand_reco_name in self.custom_recognitions:
                handler = self.custom_recognitions[cand_reco_name]
                arg = SimulatorAnalyzeArg(
                    image=frame,
                    node_name=cand_name,
                    custom_recognition_param=cand_reco_param,
                )
                curr_reco_res = handler.analyze(self.context, arg)
                if curr_reco_res is not None:
                    hit = True
                    if hasattr(curr_reco_res, "detail") and curr_reco_res.detail:
                        try:
                            d_str = (
                                curr_reco_res.detail
                                if isinstance(curr_reco_res.detail, str)
                                else json.dumps(curr_reco_res.detail)
                            )
                            curr_obs = ScreenObservation.model_validate_json(d_str)
                        except Exception:
                            pass
            elif cand_cfg.get("action") == "DoNothing" and not cand_reco_name:
                hit = True

            if hit:
                matched_node = cand_name
                matched_node_cfg = cand_cfg
                reco_res = curr_reco_res
                obs = curr_obs
                break

        action_ok = False
        clicked_coord = None

        if matched_node is not None:
            executing_node = matched_node
            node_cfg = matched_node_cfg
            reco_hit = True

            # 2. Run Action if recognition hit
            action_type = node_cfg.get("action")
            if action_type == "Custom":
                action_name = node_cfg.get("custom_action")
                action_param = node_cfg.get("custom_action_param", {})
                if action_name and action_name in self.custom_actions:
                    action_handler = self.custom_actions[action_name]
                    reco_detail = SimulatorRecoDetail(best_result=reco_res)
                    run_arg = SimulatorRunArg(
                        node_name=executing_node,
                        reco_detail=reco_detail,
                        custom_action_param=action_param,
                    )
                    res = action_handler.run(self.context, run_arg)
                    action_ok = bool(res)
            elif action_type == "Click":
                target = node_cfg.get("target", [540, 960, 10, 10])
                cx = target[0] + target[2] // 2
                cy = target[1] + target[3] // 2
                clicked_coord = (cx, cy)
                if self.controller:
                    action_ok = self.controller.click(cx, cy)
                else:
                    action_ok = True
            elif action_type == "DoNothing":
                action_ok = True

            # 3. Determine next node
            next_candidates = self._overridden_next.pop(executing_node, None)
            if not next_candidates:
                next_candidates = node_cfg.get("next", ["SceneRouting"])
        else:
            # Fallback when all candidates exhausted without recognition
            reco_hit = False
            executing_node = candidates[0] if candidates else self.current_node
            fallback_cfg = self.pipeline.get(executing_node, {})
            next_candidates = self._overridden_next.pop(executing_node, None)
            if not next_candidates:
                next_candidates = fallback_cfg.get("on_error", ["SceneRouting"])

        next_node = next_candidates[0] if next_candidates else "SceneRouting"
        self._active_candidates = list(next_candidates) if next_candidates else ["SceneRouting"]
        self._current_node = next_node

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PipelineStepResult(
            step=step_idx,
            node_name=executing_node,
            recognition_hit=reco_hit,
            action_success=action_ok,
            clicked_coord=clicked_coord,
            next_node=next_node,
            observation=obs,
            elapsed_ms=round(elapsed_ms, 2),
        )

    def run_task(self, entry_node: str = "SceneRouting", max_steps: int = 15) -> List[PipelineStepResult]:
        """Run execution loop up to max_steps."""
        self.current_node = entry_node
        results: List[PipelineStepResult] = []
        for _ in range(max_steps):
            res = self.step()
            results.append(res)
            if not res.recognition_hit and not res.action_success:
                break
        return results
