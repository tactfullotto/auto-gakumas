"""
Mock Controller for offline simulation, unit testing, and autonomous demo mode.
Integrates with MockFrameGenerator to produce synthetic frames and simulate turn progression.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import time
import numpy as np

from demo.schemas import BoundingBox, ScreenObservation
from demo.core.db import GakumasDB
from demo.vision.mock_generator import MockFrameGenerator
from demo.vision.roi import BASE_WIDTH, BASE_HEIGHT, END_TURN_BBOX
from demo.maa_bridge.base_controller import BaseController, ControllerExecutionError


class MockController(BaseController):
    """
    Simulated controller with built-in scenario management, frame injection,
    and auto-progression.
    """

    def __init__(
        self,
        generator: Optional[MockFrameGenerator] = None,
        db: Optional[GakumasDB] = None,
        plan: str = "sense",
        auto_progress: bool = True,
        **kwargs,
    ):
        self.db = db
        self.plan = plan.lower()
        self.generator = generator or MockFrameGenerator(db=db, plan=self.plan)
        self.auto_progress = auto_progress
        self._connected = False

        # State & frame storage
        self.current_frame: Optional[np.ndarray] = None
        self.current_obs: Optional[ScreenObservation] = None
        self._screencap_read: bool = False
        self.scenario_id: str = str(kwargs.get("scenario_id", "hajime_regular"))
        self.current_week: int = int(kwargs.get("week", 1))
        self.week_mode: bool = bool(kwargs.get("week_mode", False))

        # Queues for injected frames and scenario scripts
        self._frame_queue: List[Tuple[np.ndarray, Optional[ScreenObservation]]] = []
        self.scenario_queue: List[Dict[str, Any]] = []

        # Action history tracking
        self.click_history: List[Dict[str, Any]] = []
        self.swipe_history: List[Dict[str, Any]] = []
        self._action_history: List[Tuple] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def screencap(self) -> np.ndarray:
        if not self._connected:
            raise ControllerExecutionError("MockController is not connected")
        if self.current_frame is None:
            if self._frame_queue:
                self.current_frame, self.current_obs = self._frame_queue.pop(0)
            else:
                self.set_scenario("battle_turn")
        self._screencap_read = True
        return self.current_frame

    def click(self, x: int, y: int) -> bool:
        if not self._connected:
            raise ControllerExecutionError("MockController is not connected")

        timestamp = time.time()
        hit_target = self._identify_hit_target(x, y)
        self.click_history.append({
            "x": x,
            "y": y,
            "timestamp": timestamp,
            "hit_target": hit_target,
        })
        self._action_history.append(("click", x, y, timestamp))

        if self.auto_progress:
            self._handle_auto_progress(x, y, hit_target)

        return True

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> bool:
        if not self._connected:
            raise ControllerExecutionError("MockController is not connected")

        timestamp = time.time()
        self.swipe_history.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "duration_ms": duration_ms,
            "timestamp": timestamp,
        })
        self._action_history.append(("swipe", x1, y1, x2, y2, duration_ms, timestamp))
        return True

    def get_click_history(self) -> List[Tuple]:
        """Return unified action history formatted as tuples for test verification."""
        return list(self._action_history)

    def inject_frame(self, frame: np.ndarray, obs: Optional[ScreenObservation] = None) -> None:
        """Inject a synthetic frame into the controller. Replaces if read, queues if pending."""
        if self.current_frame is None or self._screencap_read:
            self.current_frame = frame
            self.current_obs = obs
            self._screencap_read = False
        else:
            self._frame_queue.append((frame, obs))

    def set_scenario(self, scene_type: str = "battle_turn", **kwargs) -> Tuple[np.ndarray, ScreenObservation]:
        """Manually inject a synthetic scenario by scene type."""
        frame, obs = self.generator.generate_frame(scene_type=scene_type, plan=self.plan, **kwargs)
        self.current_frame = frame
        self.current_obs = obs
        return frame, obs

    def set_week(
        self,
        week: int = 1,
        scenario_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """Set controller to a specific week in a produce scenario (e.g. Week 1..13 of hajime_regular)."""
        self.week_mode = True
        if scenario_id:
            self.scenario_id = scenario_id
        self.current_week = int(week)
        frame, obs = self.generator.generate_week_frame(
            week=self.current_week,
            scenario_id=self.scenario_id,
            plan=self.plan,
            **kwargs,
        )
        self.current_frame = frame
        self.current_obs = obs
        return frame, obs

    def set_custom_frame(self, frame: np.ndarray, obs: ScreenObservation) -> None:
        """Inject explicit custom frame and observation directly."""
        self.current_frame = frame
        self.current_obs = obs

    def queue_scenarios(self, scenarios: List[Union[str, Dict[str, Any]]]) -> None:
        """Queue a sequence of scenarios to step through on actions."""
        for s in scenarios:
            if isinstance(s, str):
                self.scenario_queue.append({"scene_type": s})
            else:
                self.scenario_queue.append(s)

    def last_click_hit_bbox(self, bbox: BoundingBox) -> bool:
        """Check if the most recent click fell inside the specified bbox."""
        if not self.click_history:
            return False
        last = self.click_history[-1]
        x, y = last["x"], last["y"]
        return bbox.xmin <= x <= bbox.xmax and bbox.ymin <= y <= bbox.ymax

    def _identify_hit_target(self, x: int, y: int) -> Optional[str]:
        if not self.current_obs:
            return None
        obs = self.current_obs

        # Check end turn button
        if END_TURN_BBOX.xmin <= x <= END_TURN_BBOX.xmax and END_TURN_BBOX.ymin <= y <= END_TURN_BBOX.ymax:
            return "end_turn"

        # Check hand cards
        for i, card in enumerate(obs.cards):
            b = card.bbox
            if b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax:
                return f"card_{i}_{card.card_id}"

        # Check drinks
        for i, drink in enumerate(obs.drinks):
            b = drink.bbox
            if b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax:
                return f"drink_{i}_{drink.drink_id}"

        # Check options
        for i, opt in enumerate(obs.options):
            b = opt.get("bbox")
            if isinstance(b, BoundingBox):
                if b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax:
                    return f"option_{i}_{opt.get('name', '')}"
            elif isinstance(b, dict):
                if b.get("xmin", 0) <= x <= b.get("xmax", 0) and b.get("ymin", 0) <= y <= b.get("ymax", 0):
                    return f"option_{i}_{opt.get('name', '')}"
            elif isinstance(b, (tuple, list)) and len(b) == 4:
                if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                    return f"option_{i}_{opt.get('name', '')}"

        return None

    def _handle_auto_progress(self, x: int, y: int, hit_target: Optional[str]) -> None:
        """Simulate game turn and scenario progression upon click events."""
        # 1. Advance injected frame queue if present
        if self._frame_queue:
            self.current_frame, self.current_obs = self._frame_queue.pop(0)
            self._screencap_read = False
            return

        # 2. Advance scenario queue if present
        if self.scenario_queue:
            next_scene = dict(self.scenario_queue.pop(0))
            if "week" in next_scene:
                w = int(next_scene.pop("week"))
                sid = next_scene.pop("scenario_id", self.scenario_id)
                self.set_week(week=w, scenario_id=sid, **next_scene)
            else:
                self.set_scenario(**next_scene)
            self._screencap_read = False
            return

        if not self.current_obs:
            return

        obs = self.current_obs
        if self.week_mode and obs.scene_type != "battle_turn":
            next_w = (self.current_week % max(1, obs.total_weeks)) + 1
            self.set_week(week=next_w, scenario_id=self.scenario_id)
            return

        if obs.scene_type == "battle_turn":
            carry = {
                "max_stamina": obs.max_stamina,
                "target_score": obs.target_score,
                "buffs": dict(obs.buffs),
            }
            if hit_target == "end_turn":
                new_turn = obs.turn + 1
                new_remaining = max(0, obs.turns_remaining - 1)
                if self.week_mode and new_remaining == 0:
                    next_w = (self.current_week % max(1, obs.total_weeks)) + 1
                    self.set_week(week=next_w, scenario_id=self.scenario_id)
                    return
                self.set_scenario(
                    scene_type="battle_turn",
                    turn=new_turn,
                    turns_remaining=new_remaining,
                    stamina=obs.stamina,
                    genki=obs.genki,
                    score=obs.score + 500,
                    drinks=list(obs.drinks),
                    **carry,
                )
            elif hit_target and hit_target.startswith("card_"):
                parts = hit_target.split("_")
                card_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                remaining_cards = [c for i, c in enumerate(obs.cards) if i != card_idx]
                new_score = obs.score + 800
                if self.week_mode and (not remaining_cards or (obs.target_score > 0 and new_score >= obs.target_score)):
                    next_w = (self.current_week % max(1, obs.total_weeks)) + 1
                    self.set_week(week=next_w, scenario_id=self.scenario_id)
                    return
                self.set_scenario(
                    scene_type="battle_turn",
                    turn=obs.turn,
                    turns_remaining=obs.turns_remaining,
                    stamina=max(0, obs.stamina - 4),
                    genki=obs.genki,
                    score=new_score,
                    cards=remaining_cards,
                    drinks=list(obs.drinks),
                    **carry,
                )
            elif hit_target and hit_target.startswith("drink_"):
                parts = hit_target.split("_")
                drink_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                remaining_drinks = [d for i, d in enumerate(obs.drinks) if i != drink_idx]
                self.set_scenario(
                    scene_type="battle_turn",
                    turn=obs.turn,
                    turns_remaining=obs.turns_remaining,
                    stamina=obs.stamina,
                    genki=obs.genki,
                    score=obs.score + 300,
                    cards=list(obs.cards),
                    drinks=remaining_drinks,
                    **carry,
                )
            if self.current_obs is not None:
                self.current_obs.scenario_id = obs.scenario_id
                self.current_obs.week = obs.week
                self.current_obs.total_weeks = obs.total_weeks
                self.current_obs.week_activity = obs.week_activity
                self.current_obs.idol_stats = dict(obs.idol_stats)
                self.current_obs.p_items = list(obs.p_items)
                self.current_obs.stage_bonuses = dict(obs.stage_bonuses)
        elif obs.scene_type in ("produce_schedule", "card_reward"):
            self.set_scenario(scene_type="battle_turn", turn=1, turns_remaining=10)
        elif obs.scene_type in ("popup", "shop"):
            self.set_scenario(scene_type="battle_turn")
