"""
High-Level Automation Runner and Orchestrator for Gakumas Dual-Mode Autonomous System.
Coordinates perception, state tracking, decision-making, action dispatching,
and Rich TUI event notification.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from demo.schemas import BoundingBox, DecisionResult, LegalAction, ScreenObservation
from demo.core.db import GakumasDB
from demo.core.idol_guides import resolve_p_idol_id
from demo.core.tracker import GameStateTracker
from demo.core.prompt import PromptBuilder
from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.maa_bridge.base_controller import BaseController
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.adb_controller import AdbController
from demo.maa_bridge.win32_controller import Win32Controller
from demo.maa_bridge.custom_recognition import GakumasSceneRoutingRecognition

logger = logging.getLogger(__name__)


def _compute_hand_card_core_motion(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    cards: Sequence[Any],
) -> Tuple[float, bool]:
    """Compute motion MAD inside opaque card cores, ignoring animated 3D background gaps."""
    if frame_a is None or frame_b is None or frame_a.size == 0 or frame_a.shape != frame_b.shape:
        return (0.0, False)
    if len(frame_a.shape) < 2:
        return (0.0, False)
    fh, fw = frame_a.shape[:2]
    card_mads: List[float] = []
    for c in cards or ():
        bbox = getattr(c, "bbox", None)
        if bbox is None:
            continue
        if isinstance(bbox, dict):
            xmin = int(bbox.get("xmin", 0) or 0)
            ymin = int(bbox.get("ymin", 0) or 0)
            xmax = int(bbox.get("xmax", 0) or 0)
            ymax = int(bbox.get("ymax", 0) or 0)
        else:
            xmin = int(getattr(bbox, "xmin", 0) or 0)
            ymin = int(getattr(bbox, "ymin", 0) or 0)
            xmax = int(getattr(bbox, "xmax", 0) or 0)
            ymax = int(getattr(bbox, "ymax", 0) or 0)
        cw = xmax - xmin
        ch = ymax - ymin
        if cw < 24 or ch < 24:
            continue
        x1 = max(0, min(fw - 1, int(xmin + cw * 0.16)))
        x2 = max(x1 + 1, min(fw, int(xmax - cw * 0.16)))
        y1 = max(0, min(fh - 1, int(ymin + ch * 0.32)))
        y2 = max(y1 + 1, min(fh, int(ymax - ch * 0.08)))
        crop_a = frame_a[y1:y2, x1:x2].astype(np.float32)
        crop_b = frame_b[y1:y2, x1:x2].astype(np.float32)
        card_mads.append(float(np.mean(np.abs(crop_b - crop_a))))
    if card_mads:
        return (max(card_mads), True)

    y1 = max(0, min(fh - 1, int(fh * 0.84)))
    y2 = max(y1 + 1, min(fh, int(fh * 0.89)))
    x1 = max(0, min(fw - 1, int(fw * 0.18)))
    x2 = max(x1 + 1, min(fw, int(fw * 0.82)))
    fallback_mad = float(
        np.mean(np.abs(frame_b[y1:y2, x1:x2].astype(np.float32) - frame_a[y1:y2, x1:x2].astype(np.float32)))
    )
    return (fallback_mad, False)


class RunnerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StepResult:
    """Summary of a single perception-decision-action cycle."""
    step_idx: int
    success: bool
    scene_type: str = "unknown"
    observation: Optional[ScreenObservation] = None
    legal_actions: List[LegalAction] = field(default_factory=list)
    decision: Optional[DecisionResult] = None
    action_taken: Optional[LegalAction] = None
    click_coords: Optional[Tuple[int, int]] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class GakumasRunner:
    """
    High-level orchestrator class for Gakumas automation.
    Integrates controllers, perception, agent decision-making, and event hooks.
    """

    def __init__(
        self,
        mode: str = "mock",  # 'mock' | 'adb' | 'win32'
        controller: Optional[BaseController] = None,
        controller_config: Optional[Dict[str, Any]] = None,
        agent: Optional[AgentDecisionMaker] = None,
        db: Optional[GakumasDB] = None,
        tracker: Optional[GameStateTracker] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        pipeline_path: Optional[Union[str, Path]] = None,
        plan: str = "sense",
        scenario_id: str = "hajime_regular",
        p_idol_id: Optional[int] = None,
        session_mode: Optional[bool] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        auto_resume: bool = True,
        idle_blank_tap_seconds: float = 30.0,
        loadout_scanner: Optional[Any] = None,
        loadout_library_path: Optional[Union[str, Path]] = None,
        manual_memory_hooks: Optional[Dict[Any, Dict[str, Any]]] = None,
        manual_card_customizations: Optional[Dict[Any, Dict[str, Any]]] = None,
        manual_memory_slots: Optional[List[Dict[str, Any]]] = None,
        manual_memory_passives: Optional[List[Dict[str, Any]]] = None,
        retry_home_on_empty_loadout: Optional[bool] = None,
        max_loadout_home_retries: int = 3,
        # Rich TUI Event Hooks
        on_observation: Optional[Callable[[ScreenObservation], None]] = None,
        on_prompt_built: Optional[Callable[[str, List[LegalAction]], None]] = None,
        on_decision: Optional[Callable[[DecisionResult, Optional[LegalAction]], None]] = None,
        on_action_executed: Optional[Callable[[Optional[LegalAction], Tuple[int, int]], None]] = None,
        on_step_completed: Optional[Callable[[StepResult], None]] = None,
        on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None,
    ):
        self.mode = mode.lower()
        if self.mode not in ("mock", "adb", "win32"):
            raise ValueError(f"Unsupported mode: '{mode}'. Must be 'mock', 'adb', or 'win32'.")

        self.plan = plan.lower()
        self.scenario_id = scenario_id
        self.p_idol_id: Optional[int] = p_idol_id
        self.session_mode: bool = (self.mode != "mock") if session_mode is None else bool(session_mode)
        self.db = db or GakumasDB()
        self.controller_config = controller_config or {}
        self.controller = controller or self._create_controller(self.mode, self.controller_config)
        self.agent = agent or AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
        if tracker is not None:
            self.tracker = tracker
            if p_idol_id is not None:
                self.tracker.p_idol_id = int(p_idol_id)
        else:
            default_pid = (
                p_idol_id
                if p_idol_id is not None
                else (None if self.scenario_id in ("hif_honsen", "hif") else (126 if self.plan == "logic" else None))
            )
            self.tracker = GameStateTracker(plan=self.plan, p_idol_id=default_pid, scenario_id=self.scenario_id)
        self.prompt_builder = prompt_builder or PromptBuilder(db=self.db)
        if hasattr(self.agent, "bind_tracker"):
            self.agent.bind_tracker(self.tracker)
        self.loadout_scanner = loadout_scanner
        self.loadout_library_path: Optional[Path] = Path(loadout_library_path) if loadout_library_path else None
        self.manual_memory_hooks: Dict[int, Dict[str, Any]] = {
            int(k): dict(v) for k, v in (manual_memory_hooks or {}).items() if isinstance(v, dict)
        }
        self.manual_card_customizations: Dict[int, Dict[str, Any]] = {
            int(k): dict(v) for k, v in (manual_card_customizations or {}).items() if isinstance(v, dict)
        }
        self.manual_memory_slots: List[Dict[str, Any]] = [
            dict(s) for s in (manual_memory_slots or []) if isinstance(s, dict)
        ]
        self.manual_memory_passives: List[Dict[str, Any]] = [
            dict(p) for p in (manual_memory_passives or []) if isinstance(p, dict)
        ]
        self._has_entered_produce: bool = False
        self._has_seen_produce_setup: bool = False
        self._produce_event_seen_count: int = 0
        self._loadout_scan_attempts: int = 0
        self._loadout_lp_attempts: int = 0
        self._loadout_detail_attempts: int = 0
        self._in_loadout_scan: bool = False
        self.retry_home_on_empty_loadout: bool = (
            bool(retry_home_on_empty_loadout)
            if retry_home_on_empty_loadout is not None
            else bool(self.mode != "mock" and not isinstance(self.controller, MockController))
        )
        self.max_loadout_home_retries: int = max(0, int(max_loadout_home_retries))
        self._loadout_home_retries: int = 0
        self._retrying_loadout_via_home: bool = False
        self.scene_routing_reco = GakumasSceneRoutingRecognition(
            db=self.db,
            plan=self.plan,
            scenario_id=self.scenario_id,
            p_idol_id=self.tracker.p_idol_id,
        )
        agent_prefs = getattr(self.agent, "preferences", None)
        if agent_prefs is not None and hasattr(self.scene_routing_reco, "set_banned_drink_ids"):
            self.scene_routing_reco.set_banned_drink_ids(agent_prefs.banned_drink_ids)

        default_pipeline = Path(__file__).parent / "pipeline.json"
        self.pipeline_path = Path(pipeline_path) if pipeline_path else default_pipeline
        self.checkpoint_path: Optional[Path] = Path(checkpoint_path) if checkpoint_path else None
        self.auto_resume: bool = bool(auto_resume)
        self._checkpoint_loaded: bool = False
        self._claimed_card_in_reward_scene: bool = False

        # Event hooks
        self.on_observation = on_observation
        self.on_prompt_built = on_prompt_built
        self.on_decision = on_decision
        self.on_action_executed = on_action_executed
        self.on_step_completed = on_step_completed
        self.on_error = on_error

        # Runtime status
        self.state = RunnerState.IDLE
        self.current_step = 0
        self.history: List[StepResult] = []
        self.max_retries = 3
        self.reconnect_retries = 3
        self.idle_blank_tap_seconds: float = float(idle_blank_tap_seconds)
        self._ctrl_lock = threading.Lock()
        self._last_frame_size: Tuple[int, int] = (720, 1600)
        now_ts = time.monotonic()
        self._last_progress_ts: float = now_ts
        self._last_heartbeat_ts: float = now_ts
        self._last_blank_tap_ts: float = now_ts
        self._last_frame_thumb: Optional[np.ndarray] = None
        self._last_progress_sig: Optional[Tuple[Any, ...]] = None
        self._watchdog_stop_evt = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self.watchdog_tap_count: int = 0
        self._shop_step_count: int = 0
        self._reward_step_count: int = 0
        self._reward_forced_cancel_count: int = 0
        self._reward_grid_scrolled_bottom: bool = False
        self._reward_grid_flipped_scroll: bool = False
        self._reward_grid_started_at_top: bool = False
        self._mid_battle_skip_streak: int = 0
        self._popup_fallback_streak: int = 0
        self._popup_fallback_key: Optional[Tuple[int, str, Tuple[int, int]]] = None
        self._in_battle_decision: bool = False
        self._pending_change_pickup_cid: Optional[int] = None
        self._battle_toast_settle_sleep: float = 0.9
        self._last_stable_shop_p_points: Optional[int] = None
        self._last_shop_expected_p: Optional[int] = None
        self._shop_settle_sleep: float = 0.90
        self._schedule_settle_sleep: float = 1.0
        self._last_scene_type: Optional[str] = None

    def tap_blank_area(self, reason: str = "") -> Tuple[int, int]:
        """Tap a safe HUD-free blank area on screen (e.g. when stalled >30s)."""
        fw, fh = self._last_frame_size
        tx, ty = int(round(fw * 0.93)), int(round(fh * 0.26))
        with self._ctrl_lock:
            if self.controller.is_connected:
                self.controller.click(tx, ty)
        self.watchdog_tap_count += 1
        self._last_blank_tap_ts = time.monotonic()
        logger.warning(
            "Watchdog blank-area tap #%d at (%d, %d)%s",
            self.watchdog_tap_count,
            tx,
            ty,
            f" ({reason})" if reason else "",
        )
        return (tx, ty)

    def check_idle_watchdog(self, now: Optional[float] = None) -> bool:
        """Fire a blank-area tap once every `idle_blank_tap_seconds` whenever stuck with no progress."""
        if (
            self.state != RunnerState.RUNNING
            or self.idle_blank_tap_seconds <= 0
            or self._in_loadout_scan
            or self._in_battle_decision
        ):
            return False
        cur = float(now) if now is not None else time.monotonic()
        if (cur - self._last_blank_tap_ts) < self.idle_blank_tap_seconds:
            return False
        stalled_progress = (cur - self._last_progress_ts) >= self.idle_blank_tap_seconds
        stalled_heartbeat = (cur - self._last_heartbeat_ts) >= self.idle_blank_tap_seconds
        if not (stalled_progress or stalled_heartbeat):
            return False
        why = (
            f"no progress for {cur - self._last_progress_ts:.1f}s"
            if stalled_progress
            else f"no step heartbeat for {cur - self._last_heartbeat_ts:.1f}s"
        )
        self.tap_blank_area(why)
        if now is not None:
            self._last_blank_tap_ts = float(now)
        return True

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop_evt.wait(2.0):
            try:
                self.check_idle_watchdog()
            except Exception as e:
                logger.warning("Idle watchdog tick error: %s", e)

    def _create_controller(self, mode: str, config: Dict[str, Any]) -> BaseController:
        if mode == "mock":
            return MockController(db=self.db, plan=self.plan, **config)
        elif mode == "adb":
            host = config.get("host", "192.168.0.115")
            port = config.get("port", 5555)
            return AdbController(host=host, port=port, **config)
        elif mode == "win32":
            window_title = config.get("window_title", "学园偶像大师")
            return Win32Controller(window_title=window_title, **config)
        raise ValueError(f"Unknown controller mode: {mode}")

    def start(self) -> bool:
        """Connect controller, restore saved checkpoint if requested, and initialize runner state."""
        if self.auto_resume and not self._checkpoint_loaded and self.checkpoint_path and self.checkpoint_path.exists():
            self.load_checkpoint()
        if not self.controller.is_connected:
            ok = self.controller.connect()
            if not ok:
                logger.error("Failed to connect controller in mode '%s'", self.mode)
                self.state = RunnerState.ERROR
                return False

        self.state = RunnerState.RUNNING
        now_ts = time.monotonic()
        self._last_progress_ts = now_ts
        self._last_heartbeat_ts = now_ts
        self._last_blank_tap_ts = now_ts
        self._watchdog_stop_evt.clear()
        if (
            self.mode != "mock"
            and not isinstance(self.controller, MockController)
            and self.idle_blank_tap_seconds > 0
            and (self._watchdog_thread is None or not self._watchdog_thread.is_alive())
        ):
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="gakumas-idle-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()
        logger.info("GakumasRunner started successfully in mode '%s'", self.mode)
        return True

    def stop(self) -> None:
        """Stop runner, persist checkpoint if configured, and disconnect controller cleanly."""
        self.state = RunnerState.STOPPED
        self._watchdog_stop_evt.set()
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=3.0)
        self.save_checkpoint()
        try:
            if self.controller.is_connected:
                self.controller.disconnect()
        except Exception as e:
            logger.warning("Error during controller disconnect: %s", e)
        logger.info("GakumasRunner stopped.")

    def save_checkpoint(self, path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Atomically persist tracker, recognizer fallback state, and LLM session to JSON."""
        target = Path(path) if path else self.checkpoint_path
        if target is None:
            return None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "mode": self.mode,
                "plan": self.plan,
                "scenario_id": self.scenario_id,
                "current_step": self.current_step,
                "saved_at": time.time(),
                "tracker": self.tracker.export_checkpoint(),
                "recognizer": self.scene_routing_reco.export_state(),
                "agent_session": self.agent.export_session(),
            }
            tmp_path = target.with_suffix(target.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, target)
            return target
        except Exception as e:
            logger.warning("Failed to save runner checkpoint %s: %s", target, e)
            return None

    def _flush_produce_library(self) -> None:
        """Persist updated produce deck/loadout library JSON when configured."""
        if self.loadout_library_path is None:
            return
        try:
            self.loadout_library_path.parent.mkdir(parents=True, exist_ok=True)
            lib_payload = self.tracker.export_produce_library(db=self.db)
            self.loadout_library_path.write_text(
                json.dumps(lib_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Failed to write produce library %s: %s", self.loadout_library_path, e)

    def load_checkpoint(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Restore tracker, recognizer state, and LLM session from JSON checkpoint."""
        target = Path(path) if path else self.checkpoint_path
        if target is None or not target.exists():
            return False
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse checkpoint %s: %s", target, e)
            return False
        if not isinstance(data, dict):
            return False
        saved_plan = str(data.get("plan") or self.plan).lower()
        saved_sc = str(data.get("scenario_id") or self.scenario_id)
        if saved_plan != self.plan or saved_sc != self.scenario_id:
            logger.info(
                "Skipping checkpoint %s due to config mismatch (%s/%s != %s/%s).",
                target, saved_plan, saved_sc, self.plan, self.scenario_id,
            )
            return False

        if isinstance(data.get("tracker"), dict):
            self.tracker.load_checkpoint(data["tracker"], db=self.db)
        if self.tracker.p_idol_id is not None:
            self.p_idol_id = self.tracker.p_idol_id
            if hasattr(self.scene_routing_reco, "set_p_idol_id"):
                self.scene_routing_reco.set_p_idol_id(self.tracker.p_idol_id)
        if isinstance(data.get("recognizer"), dict):
            self.scene_routing_reco.load_state(data["recognizer"])
        restored_sess = False
        if isinstance(data.get("agent_session"), dict):
            restored_sess = self.agent.load_session(data["agent_session"])
        self.current_step = int(data.get("current_step", self.current_step) or 0)

        # Keep tracker sync flags strictly consistent with loaded agent.session_messages
        if self.session_mode and not restored_sess:
            self.tracker.reset_llm_session()
        if self.session_mode:
            rec_state = data.get("recognizer") if isinstance(data.get("recognizer"), dict) else {}
            w_hint = rec_state.get("last_week") or self.tracker.llm_last_week or "?"
            self.tracker.record_session_event(
                f"【程序中断恢复】已从检查点恢复第 {w_hint} 周卡库({len(self.tracker.produce_deck)}张)/饮料/面板上下文，请结合当前实机画面无缝衔接决策。"
            )
        self._checkpoint_loaded = True
        logger.info("Restored GakumasRunner checkpoint from %s (step=%d, deckN=%d).", target, self.current_step, len(self.tracker.produce_deck))
        return True

    def _get_inner_recognizer(self):
        return (
            getattr(self.scene_routing_reco, "_live_recognizer", None)
            or getattr(self.scene_routing_reco, "recognizer", None)
        )

    def reset_produce_session(self, clear_checkpoint: bool = True) -> None:
        """Reset deck/LLM memory (and optional saved checkpoint) when starting a brand-new Produce run."""
        self.tracker.reset(keep_deck=False)
        self.agent.reset_session()
        self._checkpoint_loaded = False
        self._claimed_card_in_reward_scene = False
        self._loadout_scan_attempts = 0
        self._loadout_lp_attempts = 0
        self._loadout_detail_attempts = 0
        self._loadout_home_retries = 0
        self._retrying_loadout_via_home = False
        self._reward_step_count = 0
        self._reward_forced_cancel_count = 0
        self._reward_grid_scrolled_bottom = False
        self._mid_battle_skip_streak = 0
        self._popup_fallback_streak = 0
        self._popup_fallback_key = None
        self._last_scene_type = None
        if clear_checkpoint and self.checkpoint_path and self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
            except Exception as e:
                logger.warning("Failed to remove checkpoint %s: %s", self.checkpoint_path, e)

    def run_once(self) -> StepResult:
        """
        Execute one perception -> decision -> action step.
        Guarantees complete exception containment and triggers callbacks in order:
        obs -> prompt -> dec -> act -> step
        """
        self.current_step += 1
        step_idx = self.current_step
        start_time = time.perf_counter()

        try:
            # 1. Connection check with auto-reconnect
            if not self.controller.is_connected:
                reconnected = False
                for attempt in range(1, self.reconnect_retries + 1):
                    logger.warning(
                        "Controller disconnected. Reconnecting attempt %d/%d...",
                        attempt,
                        self.reconnect_retries,
                    )
                    if self.controller.connect():
                        reconnected = True
                        break
                    time.sleep(0.1)
                if not reconnected:
                    raise ConnectionError(f"Controller {self.mode} disconnected and reconnect failed.")

            # 2 & 3. Screencap retrieval + Vision perception (wait out NOW LOADING frames cleanly)
            frame: Optional[np.ndarray] = None
            obs: Optional[ScreenObservation] = None
            precaptured_frame: Optional[np.ndarray] = None
            precaptured_ts: float = 0.0
            max_loading_waits = 6 if (self.mode != "mock" and not isinstance(self.controller, MockController)) else 1
            schedule_entry_settled = False
            for load_try in range(max_loading_waits):
                frame = None
                cap_ts = 0.0
                if precaptured_frame is not None and precaptured_frame.size > 0:
                    frame = precaptured_frame
                    cap_ts = precaptured_ts
                    precaptured_frame = None
                else:
                    for attempt in range(self.max_retries):
                        with self._ctrl_lock:
                            frame = self.controller.screencap()
                        if frame is not None and frame.size > 0:
                            cap_ts = time.perf_counter()
                            break
                        time.sleep(0.1)

                if frame is None or frame.size == 0:
                    raise RuntimeError("Failed to capture valid screencap from controller.")

                obs = self.scene_routing_reco.recognize(frame)
                if obs is None:
                    obs = ScreenObservation(scene_type="battle_turn", turn=1, turns_remaining=10)
                    break
                if obs.scene_type == "loading" and load_try + 1 < max_loading_waits:
                    logger.info("Detected loading transition frame (%d/%d); waiting 1.5s...", load_try + 1, max_loading_waits)
                    time.sleep(1.5)
                    continue
                if (
                    obs.scene_type == "battle_turn"
                    and max_loading_waits > 1
                    and (
                        (not obs.cards and load_try == 0)
                        or (isinstance(self.controller, (AdbController, Win32Controller)) and (not obs.options or not obs.cards) and load_try < 4)
                    )
                ):
                    logger.info(
                        "Detected battle transition (cards=%d, skip_ready=%s); settling 1.1s...",
                        len(obs.cards),
                        bool(obs.options),
                    )
                    time.sleep(1.1)
                    continue
                if (
                    obs.scene_type == "battle_turn"
                    and isinstance(self.controller, (AdbController, Win32Controller))
                    and max_loading_waits > 1
                    and load_try < 3
                    and obs.cards
                    and all(not bool(getattr(c, "playable", True)) for c in obs.cards)
                ):
                    logger.info(
                        "Detected all %d battle hand cards dimmed mid-animation (try %d/3); settling 1.1s...",
                        len(obs.cards),
                        load_try + 1,
                    )
                    time.sleep(1.1)
                    continue
                if (
                    obs.scene_type == "battle_turn"
                    and isinstance(self.controller, (AdbController, Win32Controller))
                    and max_loading_waits > 1
                    and load_try < 2
                    and obs.cards
                    and frame is not None
                    and frame.size > 0
                ):
                    rem_wait = 0.30 - (time.perf_counter() - cap_ts)
                    if rem_wait > 0.01:
                        time.sleep(rem_wait)
                    with self._ctrl_lock:
                        probe_frame = self.controller.screencap()
                    if probe_frame is not None and probe_frame.shape == frame.shape:
                        hand_motion, used_card_cores = _compute_hand_card_core_motion(
                            frame, probe_frame, obs.cards
                        )
                        motion_thresh = 14.0 if used_card_cores else 16.0
                        if hand_motion > motion_thresh:
                            logger.info(
                                "Detected active battle hand animation (card_motion=%.1f, cores=%s, try %d/2); settling...",
                                hand_motion,
                                used_card_cores,
                                load_try + 1,
                            )
                            time.sleep(0.45)
                            with self._ctrl_lock:
                                settled_frame = self.controller.screencap()
                            settled_ts = time.perf_counter()
                            if settled_frame is not None and settled_frame.shape == frame.shape:
                                settle_motion, _ = _compute_hand_card_core_motion(
                                    probe_frame, settled_frame, obs.cards
                                )
                                if settle_motion > motion_thresh:
                                    time.sleep(0.45)
                                    with self._ctrl_lock:
                                        settled_frame2 = self.controller.screencap()
                                    if settled_frame2 is not None and settled_frame2.shape == frame.shape:
                                        settled_frame = settled_frame2
                                        settled_ts = time.perf_counter()
                                precaptured_frame = settled_frame
                                precaptured_ts = settled_ts
                            continue
                if (
                    obs.scene_type == "battle_turn"
                    and bool(getattr(obs, "buff_hud_occluded", False))
                    and max_loading_waits > 1
                    and load_try < 4
                ):
                    logger.info(
                        "Detected floating buff toast occluding left HUD (try %d/%d); settling %.2fs before re-capture...",
                        load_try + 1,
                        max_loading_waits,
                        self._battle_toast_settle_sleep,
                    )
                    if self._battle_toast_settle_sleep > 0:
                        time.sleep(self._battle_toast_settle_sleep)
                    continue
                if (
                    obs.scene_type == "shop"
                    and isinstance(self.controller, (AdbController, Win32Controller))
                    and max_loading_waits > 1
                    and load_try < 2
                    and not any(
                        k in str((o or {}).get("name") or "") or k in str((o or {}).get("shop_slot_key") or "")
                        for o in (obs.options or [])
                        for k in ("スキルカード", "Pドリンク", "交換する")
                    )
                ):
                    logger.info(
                        "Detected shop shelf transition with 0 product options (try %d/2); settling %.2fs before re-capture...",
                        load_try + 1,
                        max(0.85, float(self._shop_settle_sleep)),
                    )
                    time.sleep(max(0.85, float(self._shop_settle_sleep)))
                    continue
                if (
                    obs.scene_type == "shop"
                    and max_loading_waits > 1
                    and load_try < 2
                    and self._last_stable_shop_p_points is not None
                ):
                    cur_shop_p = int(getattr(obs, "p_points", 0) or 0)
                    prev_shop_p = int(
                        self._last_shop_expected_p
                        if self._last_shop_expected_p is not None
                        else self._last_stable_shop_p_points
                    )
                    if (prev_shop_p >= 50 and cur_shop_p < 20) or (prev_shop_p >= 120 and cur_shop_p < prev_shop_p - 140):
                        logger.info(
                            "Detected abrupt shop P-point drop (%d -> %d, try %d/%d); settling %.2fs before re-capture...",
                            prev_shop_p,
                            cur_shop_p,
                            load_try + 1,
                            max_loading_waits,
                            self._shop_settle_sleep,
                        )
                        reco_inner = self._get_inner_recognizer()
                        if reco_inner is not None and hasattr(reco_inner, "_last_p_points"):
                            reco_inner._last_p_points = prev_shop_p
                        if self._shop_settle_sleep > 0:
                            time.sleep(self._shop_settle_sleep)
                        continue
                if obs.scene_type == "produce_schedule" and max_loading_waits > 1:
                    if (
                        not schedule_entry_settled
                        and self._last_scene_type is not None
                        and self._last_scene_type != "produce_schedule"
                        and load_try + 1 < max_loading_waits
                    ):
                        schedule_entry_settled = True
                        logger.info(
                            "Entered produce_schedule from %s; waiting %.2fs before re-capturing...",
                            self._last_scene_type,
                            self._schedule_settle_sleep,
                        )
                        if self._schedule_settle_sleep > 0:
                            time.sleep(self._schedule_settle_sleep)
                        continue
                    if load_try < 3:
                        has_non_rest_sched = any(
                            tok in f"{(o or {}).get('name') or ''} {(o or {}).get('title') or ''} {(o or {}).get('description') or ''}"
                            for o in (obs.options or ())
                            for tok in (
                                "授業",
                                "公開レッスン",
                                "レッスン",
                                "追い込み",
                                "差し入れ",
                                "活動支給",
                                "相談",
                                "特別指導",
                                "インターバル",
                                "おでかけ",
                                "破棄",
                                "捨てる",
                            )
                        )
                        sched_stam = int(getattr(obs, "stamina", 0) or 0)
                        sched_max = max(1, int(getattr(obs, "max_stamina", 44) or 44))
                        if sched_stam <= 0 and int(getattr(self.tracker, "stamina", 0) or 0) > 0:
                            sched_stam = int(self.tracker.stamina)
                            sched_max = max(1, int(getattr(self.tracker, "max_stamina", 44) or 44))
                        sched_low_hp = sched_stam > 0 and sched_stam < 15
                        if (not has_non_rest_sched) and sched_stam > 0 and (not sched_low_hp):
                            logger.info(
                                "Detected produce_schedule with rest-only/empty options at healthy HP (%d/%d, try %d/3); waiting %.2fs before re-capturing...",
                                sched_stam,
                                sched_max,
                                load_try + 1,
                                self._schedule_settle_sleep,
                            )
                            if self._schedule_settle_sleep > 0:
                                time.sleep(self._schedule_settle_sleep)
                            continue
                break

            assert obs is not None
            if obs.scene_type == "shop":
                cur_shop_p = int(getattr(obs, "p_points", 0) or 0)
                ref_shop_p = (
                    int(self._last_shop_expected_p)
                    if self._last_shop_expected_p is not None
                    else (
                        int(self._last_stable_shop_p_points)
                        if self._last_stable_shop_p_points is not None
                        else None
                    )
                )
                if ref_shop_p is not None and ref_shop_p >= 180 and cur_shop_p < ref_shop_p - 180:
                    logger.warning(
                        "Preserving expected shop P-points %d over persistent implausible OCR drop (%d).",
                        ref_shop_p,
                        cur_shop_p,
                    )
                    reco_inner = self._get_inner_recognizer()
                    if reco_inner is not None and hasattr(reco_inner, "_last_p_points"):
                        reco_inner._last_p_points = ref_shop_p
                    if frame is not None:
                        re_obs = self.scene_routing_reco.recognize(frame)
                        if re_obs is not None:
                            obs = re_obs
                            cur_shop_p = int(getattr(obs, "p_points", 0) or 0)
                    if cur_shop_p < ref_shop_p - 180:
                        obs.p_points = ref_shop_p
                        cur_shop_p = ref_shop_p
                if cur_shop_p > 0:
                    self._last_stable_shop_p_points = cur_shop_p
                self._last_shop_expected_p = None
            elif obs.scene_type not in ("shop", "popup", "card_reward"):
                self._last_stable_shop_p_points = None
                self._last_shop_expected_p = None
            if (
                self.loadout_scanner is not None
                and not self.tracker.initial_loadout_captured
                and obs.scene_type == "produce_setup"
            ):
                scan_opt = next((o for o in (obs.options or []) if o.get("is_loadout_target")), None)
                if scan_opt is not None and scan_opt.get("bbox"):
                    via_lp = bool(scan_opt.get("open_via_long_press", True))
                    max_budget = 1 if via_lp else 2
                    used_budget = self._loadout_lp_attempts if via_lp else self._loadout_detail_attempts
                    if used_budget < max_budget:
                        sb = scan_opt["bbox"]
                        open_xy = ((int(sb["xmin"]) + int(sb["xmax"])) // 2, (int(sb["ymin"]) + int(sb["ymax"])) // 2)
                        while not self.tracker.initial_loadout_captured:
                            used_budget = self._loadout_lp_attempts if via_lp else self._loadout_detail_attempts
                            if used_budget >= max_budget:
                                break
                            if via_lp:
                                self._loadout_lp_attempts += 1
                            else:
                                self._loadout_detail_attempts += 1
                            self._loadout_scan_attempts += 1
                            cur_try = self._loadout_lp_attempts if via_lp else self._loadout_detail_attempts
                            logger.info(
                                "Running H.I.F initial loadout scan (%s attempt %d/%d, long_press=%s at %s)...",
                                "LP" if via_lp else "detail",
                                cur_try,
                                max_budget,
                                via_lp,
                                open_xy,
                            )
                            now_pre = time.monotonic()
                            self._last_progress_ts = now_pre
                            self._last_heartbeat_ts = now_pre
                            self._last_blank_tap_ts = now_pre
                            self._in_loadout_scan = True
                            try:
                                with self._ctrl_lock:
                                    scan_res = self.loadout_scanner.scan_via_controller(
                                        self.controller,
                                        open_xy=open_xy,
                                        open_via_long_press=via_lp,
                                        plan=self.plan,
                                    )
                            finally:
                                self._in_loadout_scan = False
                                now_post = time.monotonic()
                                self._last_progress_ts = now_post
                                self._last_heartbeat_ts = now_post
                                self._last_blank_tap_ts = now_post

                            if scan_res.deck_card_ids:
                                self._retrying_loadout_via_home = False
                                self.tracker.set_initial_loadout(
                                    p_idol_id=scan_res.p_idol_id,
                                    plan=scan_res.plan,
                                    recommended_effect=scan_res.recommended_effect,
                                    deck_ids=scan_res.deck_card_ids,
                                    p_items=scan_res.p_items,
                                    card_memory_hooks=scan_res.card_memory_hooks,
                                    card_customizations=scan_res.card_customizations,
                                    memory_slots=self.manual_memory_slots or None,
                                    memory_passives=self.manual_memory_passives or None,
                                    db=self.db,
                                )
                                if scan_res.card_customizations:
                                    logger.info(
                                        "H.I.F loadout scan inferred customizations for %d cards: %s",
                                        len(scan_res.card_customizations),
                                        scan_res.card_customizations,
                                    )
                                if self.manual_memory_hooks or self.manual_card_customizations:
                                    self.tracker.merge_manual_memory_hooks(
                                        hooks=self.manual_memory_hooks,
                                        memory_slots=self.manual_memory_slots or None,
                                        memory_passives=self.manual_memory_passives or None,
                                        card_customizations=self.manual_card_customizations,
                                        db=self.db,
                                    )
                                if scan_res.plan:
                                    self.plan = scan_res.plan
                                    reco_inner = (
                                        getattr(self.scene_routing_reco, "_live_recognizer", None)
                                        or getattr(self.scene_routing_reco, "recognizer", None)
                                    )
                                    if reco_inner is not None and hasattr(reco_inner, "plan"):
                                        reco_inner.plan = scan_res.plan
                                if scan_res.p_idol_id is not None:
                                    self.p_idol_id = scan_res.p_idol_id
                                    if hasattr(self.scene_routing_reco, "set_p_idol_id"):
                                        self.scene_routing_reco.set_p_idol_id(scan_res.p_idol_id)
                                self._flush_produce_library()
                                break
                            logger.warning(
                                "H.I.F loadout scan (%s attempt %d/%d) yielded empty deck (p_items=%d).",
                                "LP" if via_lp else "detail",
                                cur_try,
                                max_budget,
                                len(scan_res.p_items or []),
                            )
                            if (not via_lp) and self._loadout_detail_attempts < max_budget:
                                time.sleep(0.8)
                                continue
                            break

                        # Re-capture setup frame cleanly after modal dismissal and refresh watchdog timestamps
                        now_scan_ts = time.monotonic()
                        self._last_progress_ts = now_scan_ts
                        self._last_heartbeat_ts = now_scan_ts
                        self._last_blank_tap_ts = now_scan_ts
                        with self._ctrl_lock:
                            refreshed = self.controller.screencap()
                        if refreshed is not None and refreshed.size > 0:
                            refreshed_obs = self.scene_routing_reco.recognize(refreshed)
                            if refreshed_obs is not None:
                                frame = refreshed
                                obs = refreshed_obs

                        if (
                            not self.tracker.initial_loadout_captured
                            and self.retry_home_on_empty_loadout
                            and self._loadout_home_retries < self.max_loadout_home_retries
                        ):
                            self._loadout_home_retries += 1
                            self._loadout_lp_attempts = 0
                            self._loadout_detail_attempts = 0
                            self._retrying_loadout_via_home = True
                            self._has_entered_produce = False
                            self._has_seen_produce_setup = False
                            if hasattr(self.agent, "_has_entered_produce"):
                                self.agent._has_entered_produce = False
                            if hasattr(self.agent, "_has_seen_produce_setup"):
                                self.agent._has_seen_produce_setup = False

                            fh_r, fw_r = (frame.shape[:2] if frame is not None and frame.size > 0 else (1600, 720))
                            home_x = int(round(fw_r * (84.0 / 720.0)))
                            home_y = int(round(fh_r * (1532.0 / 1600.0)))
                            logger.warning(
                                "H.I.F loadout scan found 0 cards; returning to Home (%d/%d) at (%d, %d) to re-enter cleanly.",
                                self._loadout_home_retries,
                                self.max_loadout_home_retries,
                                home_x,
                                home_y,
                            )
                            with self._ctrl_lock:
                                self.controller.click(home_x, home_y)
                            if self.mode != "mock" and not isinstance(self.controller, MockController):
                                time.sleep(1.0)
                            home_act = LegalAction(
                                choice_id=0,
                                action_type="pick_option",
                                description="返回主页重新进入以重试读取卡牌",
                                bbox=BoundingBox(
                                    xmin=max(0, home_x - 45),
                                    ymin=max(0, home_y - 32),
                                    xmax=home_x + 45,
                                    ymax=home_y + 32,
                                ),
                                payload={"name": "ホーム", "retry_empty_loadout": True},
                            )
                            home_dec = DecisionResult(
                                choice_id=0,
                                reason=(
                                    f"Runner: Loadout scan found 0 cards; returning to Home "
                                    f"({self._loadout_home_retries}/{self.max_loadout_home_retries}) to re-enter"
                                ),
                                source="runner_retry",
                            )
                            if self.on_decision:
                                try:
                                    self.on_decision(home_dec, home_act)
                                except Exception as cb_err:
                                    logger.error("Error in on_decision callback: %s", cb_err)
                            if self.on_action_executed:
                                try:
                                    self.on_action_executed(home_act, (home_x, home_y))
                                except Exception as cb_err:
                                    logger.error("Error in on_action_executed callback: %s", cb_err)
                            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                            self._last_scene_type = obs.scene_type
                            retry_result = StepResult(
                                step_idx=step_idx,
                                success=True,
                                scene_type=obs.scene_type,
                                observation=obs,
                                legal_actions=[home_act],
                                decision=home_dec,
                                action_taken=home_act,
                                click_coords=(home_x, home_y),
                                elapsed_ms=round(elapsed_ms, 2),
                            )
                            self.history.append(retry_result)
                            if self.on_step_completed:
                                try:
                                    self.on_step_completed(retry_result)
                                except Exception as cb_err:
                                    logger.error("Error in on_step_completed callback: %s", cb_err)
                            return retry_result

            is_live_ctrl = self.mode != "mock" and not isinstance(self.controller, MockController)
            raw_r = obs.raw_text or ""
            raw_r_clean = raw_r
            for _preview_btn_tok in ("強化確認", "カスタマイズ確認", "カスタマイス確認", "強化認"):
                raw_r_clean = raw_r_clean.replace(_preview_btn_tok, "")
            has_explicit_trade_prompt = any(
                k in raw_r_clean for k in ("チェンジするスキルカード", "削除するスキルカード", "手放すスキルカード")
            )
            is_trade_or_delete_grid = (
                any(tok in raw_r_clean for tok in ("チェンジ", "削除", "手放す"))
                and "チェンジで獲得" not in raw_r_clean
                and "カスタマイズ" not in raw_r_clean
                and "強化" not in raw_r_clean
                and (has_explicit_trade_prompt or ("移動" not in raw_r_clean and "手札に移動" not in raw_r_clean))
            )
            if (
                is_live_ctrl
                and obs.scene_type == "card_reward"
                and not self._reward_grid_scrolled_bottom
                and is_trade_or_delete_grid
                and any(
                    tok in str((o or {}).get("name") or "")
                    for o in (obs.options or [])
                    for tok in ("待选卡", "换出候选卡", "チェンジする", "削除する")
                )
            ):
                is_delete_flow = any(tok in raw_r for tok in ("削除", "手放す"))
                deck_ids_cur = list(self.tracker.produce_deck or [])
                deck_has_basic_n = (not deck_ids_cur) or any(
                    ((self.db.get_card_by_id(int(cid)) or {}).get("rarity") == "N")
                    for cid in deck_ids_cur
                )
                if is_delete_flow or deck_has_basic_n:
                    fh, fw = frame.shape[:2]
                    sx = int(round(fw * 0.50))
                    sy1, sy2 = int(round(fh * 0.72)), int(round(fh * 0.34))
                    logger.info("Scrolling deck-grid reward to bottom so weak/trouble cards appear first...")
                    with self._ctrl_lock:
                        for _sw in range(2):
                            self.controller.swipe(sx, sy1, sx, sy2, duration_ms=650)
                            time.sleep(0.35)
                    time.sleep(0.5)
                    reco_inner = self._get_inner_recognizer()
                    if reco_inner is not None and hasattr(reco_inner, "_reward_visited_slots"):
                        reco_inner._reward_visited_slots.clear()
                    with self._ctrl_lock:
                        refreshed_grid = self.controller.screencap()
                    if refreshed_grid is not None and refreshed_grid.size > 0:
                        refreshed_grid_obs = self.scene_routing_reco.recognize(refreshed_grid)
                        if refreshed_grid_obs is not None:
                            frame = refreshed_grid
                            obs = refreshed_grid_obs
                    self._reward_grid_started_at_top = False
                else:
                    logger.info("Staying at top of deck grid for H.I.F チェンジ (no N-rarity basic cards in deck)...")
                    self._reward_grid_started_at_top = True
                self._reward_step_count = 0
                self._reward_grid_scrolled_bottom = True

            if obs.scene_type == "home":
                self._retrying_loadout_via_home = False
            elif obs.scene_type == "produce_setup":
                self._has_seen_produce_setup = True
            elif obs.scene_type in ("produce_schedule", "battle_turn", "card_reward", "shop"):
                self._has_entered_produce = True
            elif obs.scene_type == "produce_event":
                self._produce_event_seen_count += 1
                if (
                    self._has_seen_produce_setup
                    or int(obs.week or 1) >= 2
                    or self._produce_event_seen_count >= 2
                    or bool(self.tracker.produce_deck)
                ):
                    self._has_entered_produce = True

            frame_h, frame_w = frame.shape[:2]
            self._last_frame_size = (frame_w, frame_h)
            thumb = frame[::max(1, frame_h // 32), ::max(1, frame_w // 32), :3].astype(np.float32)
            prog_sig = (
                obs.scene_type,
                obs.week,
                obs.turn,
                obs.turns_remaining,
                obs.score,
                obs.stamina,
                obs.genki,
                tuple(c.card_id for c in obs.cards),
                tuple(o.get("name") for o in obs.options),
            )
            frame_changed = (
                self._last_frame_thumb is None
                or self._last_frame_thumb.shape != thumb.shape
                or float(np.mean(np.abs(thumb - self._last_frame_thumb))) > 3.5
            )
            if frame_changed or prog_sig != self._last_progress_sig:
                self._last_progress_ts = time.monotonic()
                self._last_frame_thumb = thumb
                self._last_progress_sig = prog_sig
            if obs.scene_type == "shop":
                self._shop_step_count += 1
            elif obs.scene_type == "popup" and self._shop_step_count > 0:
                self._shop_step_count += 1
            else:
                self._shop_step_count = 0
            is_mid_reward_drink_popup = (
                obs.scene_type == "popup"
                and self._reward_step_count > 0
                and any(
                    k in (str((o or {}).get("name") or "") + str((o or {}).get("title") or ""))
                    for o in (obs.options or [])
                    for k in ("捨てる", "破棄", "ドリンク")
                )
            )
            if obs.scene_type == "card_reward":
                self._reward_step_count += 1
            elif not is_mid_reward_drink_popup:
                self._reward_step_count = 0
                self._reward_forced_cancel_count = 0
                self._reward_grid_scrolled_bottom = False
                self._reward_grid_flipped_scroll = False
                self._reward_grid_started_at_top = False
            if obs.scene_type != "card_reward":
                self._claimed_card_in_reward_scene = False

            if self.on_observation:
                try:
                    self.on_observation(obs)
                except Exception as cb_err:
                    logger.error("Error in on_observation callback: %s", cb_err)

            # Maintain produce_deck, backend shield/HP ledger & per-battle zones without breaking stateless mock tests
            if is_live_ctrl or self.session_mode:
                if self.p_idol_id is None:
                    detected_pid = resolve_p_idol_id(
                        plan=self.plan,
                        p_idol_id=None,
                        deck_ids=self.tracker.produce_deck,
                        hand_cards=obs.cards,
                        default_logic_to_sekirara=False,
                    )
                    if detected_pid is not None and detected_pid != self.tracker.p_idol_id:
                        self.tracker.p_idol_id = detected_pid
                        if self.p_idol_id is None:
                            self.p_idol_id = detected_pid
                        if hasattr(self.scene_routing_reco, "set_p_idol_id"):
                            self.scene_routing_reco.set_p_idol_id(detected_pid)
                if not self.tracker.produce_deck:
                    pid = self.tracker.p_idol_id if self.tracker.p_idol_id is not None else (
                        None if self.scenario_id in ("hif_honsen", "hif") else (126 if self.plan.lower() == "logic" else None)
                    )
                    self.tracker.init_produce_deck(
                        db=self.db,
                        plan=self.plan,
                        p_idol_id=pid,
                        scenario_id=self.scenario_id,
                    )
                if obs.scene_type == "battle_turn":
                    self._mid_battle_skip_streak = 0
                    if not self.tracker.in_battle:
                        self.tracker.start_new_battle(db=self.db)
                    if is_live_ctrl:
                        self.tracker.reconcile_live_battle_state(obs)
                    self.tracker.sync_observed_hand(
                        obs.cards,
                        db=self.db,
                        turn=obs.turn,
                        turns_remaining=obs.turns_remaining,
                    )
                else:
                    keep_mid_battle_popup = (
                        obs.scene_type == "popup"
                        and self.tracker.in_battle
                        and not any(
                            k in (obs.raw_text or "")
                            for k in ("CLEAR", "PERFECT", "結果", "順位", "獲得", "再挑戦", "完了")
                        )
                        and any(
                            o.get("name") in ("使う", "使用する", "閉じる", "キャンセル", "はい", "決定")
                            or (
                                o.get("name") == "終了"
                                and (
                                    "ターン終了" in (obs.raw_text or "")
                                    or "ターン終了" in (str(o.get("title") or "") + str(o.get("description") or ""))
                                )
                            )
                            for o in (obs.options or [])
                        )
                    )
                    _opt_blob = " ".join(
                        f"{o.get('name') or ''} {o.get('description') or ''}"
                        for o in (obs.options or [])
                    )
                    _post_battle_reward_toks = (
                        "受け取る",
                        "受け取らない",
                        "再抽選",
                        "獲得",
                        "報酬",
                        "CLEAR",
                        "PERFECT",
                        "削除",
                        "強化",
                        "カスタマイズ",
                        "セレクトチェンジ",
                        "特別指導",
                    )
                    has_post_battle_reward_cue = any(
                        k in (obs.raw_text or "") or k in _opt_blob
                        for k in _post_battle_reward_toks
                    )
                    keep_mid_battle_move_modal = (
                        obs.scene_type == "card_reward"
                        and self.tracker.in_battle
                        and (
                            any(
                                str(o.get("name") or "") == "移動する"
                                or "手札に移動" in str(o.get("description") or "")
                                or "加入手牌" in str(o.get("description") or "")
                                for o in (obs.options or [])
                            )
                            or (
                                ("移動" in (obs.raw_text or "") or "手札" in (obs.raw_text or ""))
                                and not any(
                                    k in (obs.raw_text or "")
                                    for k in ("受け取る", "チェンジ", "削除", "強化", "カスタマイズ", "獲得")
                                )
                            )
                            or (not has_post_battle_reward_cue)
                        )
                    )
                    keep_mid_battle_skip = False
                    if (
                        self.tracker.in_battle
                        and obs.scene_type == "skip"
                        and not any(
                            k in (obs.raw_text or "")
                            for k in ("CLEAR", "PERFECT", "結果", "順位", "獲得", "次へ", "完了", "インターバル")
                        )
                    ):
                        self._mid_battle_skip_streak += 1
                        if self._mid_battle_skip_streak <= 4:
                            keep_mid_battle_skip = True
                        else:
                            self._mid_battle_skip_streak = 0
                    elif not (keep_mid_battle_popup or keep_mid_battle_move_modal):
                        self._mid_battle_skip_streak = 0

                    if not (keep_mid_battle_popup or keep_mid_battle_move_modal or keep_mid_battle_skip):
                        if self.tracker.in_battle:
                            self.tracker.sanitize_produce_deck(self.db)
                            if hasattr(self.agent, "reset_battle"):
                                self.agent.reset_battle()
                        self.tracker.in_battle = False
                    if is_live_ctrl:
                        self.tracker.reconcile_live_battle_state(obs)
            self.tracker.update_observation(obs)

            # 4. Prompt & Legal Action Building
            sync_meta: Optional[Dict[str, Any]] = None
            if self.session_mode:
                prompt_text, legal_actions, sync_meta = self.prompt_builder.build_session_prompt(
                    obs,
                    self.tracker,
                    commit=False,
                )
            else:
                prompt_text, legal_actions = self.prompt_builder.build_prompt(obs, self.tracker)
            if self.on_prompt_built:
                try:
                    self.on_prompt_built(prompt_text, legal_actions)
                except Exception as cb_err:
                    logger.error("Error in on_prompt_built callback: %s", cb_err)

            # 5. Agent Decision
            if hasattr(self.agent, "_reward_grid_can_flip_scroll"):
                self.agent._reward_grid_can_flip_scroll = bool(
                    is_live_ctrl
                    and obs.scene_type == "card_reward"
                    and is_trade_or_delete_grid
                    and not self._reward_grid_flipped_scroll
                )
            is_battle_decide = (obs.scene_type == "battle_turn")
            if is_battle_decide:
                self._in_battle_decision = True
            try:
                decision = self.agent.decide(
                    prompt=prompt_text,
                    legal_actions=legal_actions,
                    obs=obs,
                    use_session=self.session_mode,
                    scenario_id=self.scenario_id,
                    p_idol_id=self.tracker.p_idol_id,
                )
            finally:
                if is_battle_decide:
                    self._in_battle_decision = False
                    now_post = time.monotonic()
                    self._last_progress_ts = now_post
                    self._last_heartbeat_ts = now_post
                    self._last_blank_tap_ts = now_post
            if (
                is_live_ctrl
                and getattr(self.agent, "_reward_grid_request_scroll", None) == "flip"
                and not self._reward_grid_flipped_scroll
            ):
                self._reward_grid_flipped_scroll = True
                self._reward_step_count = min(self._reward_step_count, 10)
                if hasattr(self.agent, "_reward_grid_request_scroll"):
                    self.agent._reward_grid_request_scroll = None
                if hasattr(self.agent, "_reward_grid_can_flip_scroll"):
                    self.agent._reward_grid_can_flip_scroll = False
                fh, fw = frame.shape[:2]
                sx = int(round(fw * 0.50))
                if self._reward_grid_started_at_top:
                    sy1, sy2 = int(round(fh * 0.72)), int(round(fh * 0.34))
                    logger.info("No positive swap candidate at top; scrolling deck grid to bottom...")
                else:
                    sy1, sy2 = int(round(fh * 0.42)), int(round(fh * 0.78))
                    logger.info("No positive swap candidate at bottom; scrolling deck grid back to top...")
                with self._ctrl_lock:
                    for _sw in range(2):
                        self.controller.swipe(sx, sy1, sx, sy2, duration_ms=650)
                        time.sleep(0.35)
                time.sleep(0.5)
                reco_inner = getattr(self.scene_routing_reco, "recognizer", None)
                if reco_inner is not None and hasattr(reco_inner, "_reward_visited_slots"):
                    reco_inner._reward_visited_slots.clear()
                if hasattr(self.agent, "_clear_reward_scan_state"):
                    self.agent._clear_reward_scan_state()
                with self._ctrl_lock:
                    flipped_grid = self.controller.screencap()
                if flipped_grid is not None and flipped_grid.size > 0:
                    flipped_obs = self.scene_routing_reco.recognize(flipped_grid)
                    if flipped_obs is not None:
                        frame = flipped_grid
                        obs = flipped_obs
                        if self.session_mode:
                            prompt_text, legal_actions, sync_meta = self.prompt_builder.build_session_prompt(
                                obs,
                                self.tracker,
                                commit=False,
                            )
                        else:
                            prompt_text, legal_actions = self.prompt_builder.build_prompt(obs, self.tracker)
                        decision = self.agent.decide(
                            prompt=prompt_text,
                            legal_actions=legal_actions,
                            obs=obs,
                            use_session=self.session_mode,
                            scenario_id=self.scenario_id,
                            p_idol_id=self.tracker.p_idol_id,
                        )
            self._last_heartbeat_ts = time.monotonic()

            chosen_action: Optional[LegalAction] = None
            for act in legal_actions:
                if act.choice_id == decision.choice_id:
                    chosen_action = act
                    break

            if chosen_action is None and legal_actions:
                chosen_action = legal_actions[0]

            shop_cap = 24 if int(getattr(obs, "week", 0) or 0) >= 8 else 14
            if obs.scene_type == "shop" and self._shop_step_count >= shop_cap and legal_actions:
                exit_act = next(
                    (
                        a for a in legal_actions
                        if any(k in str((a.payload or {}).get("name") or a.description) for k in ("終了", "次へ", "スキップ", "閉じる", "离开"))
                    ),
                    None,
                )
                if exit_act is not None and chosen_action is not exit_act:
                    logger.warning("Shop anti-stall cap (%d steps) forcing exit: %s", self._shop_step_count, exit_act.description)
                    chosen_action = exit_act
                    decision = DecisionResult(
                        choice_id=exit_act.choice_id,
                        reason=f"Shop anti-stall cap after {self._shop_step_count} steps -> exit",
                        source="rule_skip",
                    )
            elif obs.scene_type == "card_reward" and self._reward_step_count >= 20 and legal_actions:
                triggered_reward_steps = self._reward_step_count
                change_confirm_act = next(
                    (a for a in legal_actions if str((a.payload or {}).get("name") or "") == "チェンジする"),
                    None,
                )
                delete_confirm_act = next(
                    (a for a in legal_actions if str((a.payload or {}).get("name") or "") == "削除する"),
                    None,
                )
                is_active_removal_screen = (change_confirm_act is not None) or (delete_confirm_act is not None)
                if is_active_removal_screen and self._reward_step_count < 30:
                    safe_reward_act = None
                elif change_confirm_act is not None:
                    chg_cid = int((change_confirm_act.payload or {}).get("card_id") or 0)
                    chg_row = self.db.get_card_by_id(chg_cid) if (self.db and chg_cid > 0) else None
                    is_sig_change = bool(chg_row and chg_row.get("is_signature"))
                    is_custom_change = bool((change_confirm_act.payload or {}).get("is_customized"))
                    is_protected_change = is_sig_change or is_custom_change
                    if (not is_protected_change) and hasattr(self.agent, "is_protected_from_removal"):
                        is_protected_change = bool(
                            self.agent.is_protected_from_removal(
                                chg_cid,
                                is_customized=is_custom_change,
                                desc=str(change_confirm_act.description or ""),
                                p_idol_id=self.tracker.p_idol_id,
                                plan=self.plan,
                            )
                        )
                    came_from_pickup = bool(getattr(self.agent, "_came_from_change_pickup", False))
                    cancel_cand_chg = next(
                        (
                            a for a in legal_actions
                            if str((a.payload or {}).get("name") or "") in ("キャンセル", "戻る", "もどる")
                        ),
                        None,
                    )
                    if is_protected_change and (not came_from_pickup) and self._reward_forced_cancel_count < 1 and cancel_cand_chg is not None:
                        safe_reward_act = cancel_cand_chg
                        self._reward_forced_cancel_count += 1
                    elif not is_protected_change:
                        # H.I.F Page-2 mandatory チェンジ has no skip on Page 1; confirm unprotected slot cleanly.
                        safe_reward_act = change_confirm_act
                        self._reward_step_count = 0
                    else:
                        def _preview_slot_idx(act: LegalAction) -> Optional[int]:
                            p = act.payload or {}
                            if p.get("slot_idx") is not None:
                                return int(p["slot_idx"])
                            m_s = re.search(
                                r"(?:第|#|卡)\s*(\d+)",
                                str(p.get("name") or act.description or ""),
                            )
                            return (int(m_s.group(1)) - 1) if m_s else None

                        cur_chg_slot = (change_confirm_act.payload or {}).get("slot_idx")
                        if cur_chg_slot is None:
                            cur_chg_slot = getattr(self.agent, "_last_clicked_grid_slot", None)
                        cur_chg_slot_i = int(cur_chg_slot) if cur_chg_slot is not None else None

                        nc_prevs = [
                            a for a in legal_actions
                            if any(
                                tok in str((a.payload or {}).get("name") or "")
                                for tok in ("换出候选卡", "待选卡", "预览")
                            )
                            and not (a.payload or {}).get("is_customized")
                            and (_preview_slot_idx(a) is None or _preview_slot_idx(a) != cur_chg_slot_i)
                        ]
                        grid_scores = dict(getattr(self.agent, "_reward_grid_scores", None) or {})
                        visited_slots = set(getattr(self.agent, "_reward_visited_grid_slots", None) or ())

                        safe_reward_act = None
                        if self._reward_step_count < 42 and nc_prevs:
                            pos_scored = [
                                (int(grid_scores[s]), a)
                                for a in nc_prevs
                                if (s := _preview_slot_idx(a)) is not None and int(grid_scores.get(s, -99_999)) > 0
                            ]
                            if pos_scored:
                                pos_scored.sort(key=lambda item: item[0], reverse=True)
                                safe_reward_act = pos_scored[0][1]
                            else:
                                unvisited_nc = [
                                    a for a in nc_prevs
                                    if (s := _preview_slot_idx(a)) is not None and s not in visited_slots
                                ]
                                if unvisited_nc:
                                    safe_reward_act = unvisited_nc[self._reward_step_count % len(unvisited_nc)]
                                else:
                                    cur_slot_score = (
                                        int(grid_scores[cur_chg_slot_i])
                                        if cur_chg_slot_i is not None and cur_chg_slot_i in grid_scores
                                        else -99_999
                                    )
                                    better_scored = [
                                        (int(grid_scores[s]), a)
                                        for a in nc_prevs
                                        if (s := _preview_slot_idx(a)) is not None
                                        and s in grid_scores
                                        and int(grid_scores[s]) > cur_slot_score
                                    ]
                                    if better_scored:
                                        better_scored.sort(key=lambda item: item[0], reverse=True)
                                        safe_reward_act = better_scored[0][1]
                                    else:
                                        safe_reward_act = nc_prevs[self._reward_step_count % len(nc_prevs)]

                        if safe_reward_act is None:
                            if cancel_cand_chg is not None and (not came_from_pickup) and self._reward_forced_cancel_count < 2:
                                safe_reward_act = cancel_cand_chg
                                self._reward_forced_cancel_count += 1
                            else:
                                safe_reward_act = change_confirm_act
                                self._reward_step_count = 0
                elif delete_confirm_act is not None:
                    safe_reward_act = next(
                        (
                            a for a in legal_actions
                            if str((a.payload or {}).get("name") or "") in ("キャンセル", "戻る", "もどる")
                        ),
                        None,
                    )
                else:
                    safe_reward_act = next(
                        (
                            a for a in legal_actions
                            if str((a.payload or {}).get("name") or "") in (
                                "決定",
                                "受け取らない",
                                "スキップ",
                                "受け取る",
                                "次へ",
                                "強化する",
                                "カスタマイズする",
                            )
                            or any(k in a.description for k in ("已勾选『受け取らない』", "直接不拿新饮料", "スキップ"))
                        ),
                        None,
                    )
                if safe_reward_act is None and not is_active_removal_screen:
                    if self._reward_step_count >= 24 and self._reward_forced_cancel_count < 2:
                        cancel_cand = next(
                            (
                                a for a in legal_actions
                                if str((a.payload or {}).get("name") or "") in ("キャンセル", "戻る", "もどる")
                            ),
                            None,
                        )
                        if cancel_cand is not None:
                            safe_reward_act = cancel_cand
                            self._reward_forced_cancel_count += 1
                    if safe_reward_act is None:
                        prev_candidates = [
                            a for a in legal_actions
                            if any(
                                tok in str((a.payload or {}).get("name") or "")
                                for tok in ("换出候选卡", "待选卡", "候选卡", "预览")
                            )
                        ]
                        if prev_candidates:
                            safe_reward_act = prev_candidates[(self._reward_step_count - 20) % len(prev_candidates)]
                if safe_reward_act is None and not is_active_removal_screen and self._reward_forced_cancel_count < 2:
                    safe_reward_act = next(
                        (
                            a for a in legal_actions
                            if str((a.payload or {}).get("name") or "") in ("キャンセル", "戻る", "もどる")
                        ),
                        None,
                    )
                    if safe_reward_act is not None:
                        self._reward_forced_cancel_count += 1
                if safe_reward_act is not None and chosen_action is not safe_reward_act:
                    logger.warning("Reward anti-stall cap (%d steps) forcing: %s", triggered_reward_steps, safe_reward_act.description)
                    chosen_action = safe_reward_act
                    decision = DecisionResult(
                        choice_id=safe_reward_act.choice_id,
                        reason=f"Reward anti-stall cap after {triggered_reward_steps} steps",
                        source="rule_skip",
                    )

            if self.on_decision:
                try:
                    self.on_decision(decision, chosen_action)
                except Exception as cb_err:
                    logger.error("Error in on_decision callback: %s", cb_err)

            # 6. Action Execution (Click Mapping + Double-Tap for cards & early produce selection)
            click_coords: Optional[Tuple[int, int]] = None
            returning_home_hold = obs.scene_type in ("home", "produce_setup") and (
                self._has_entered_produce or bool(getattr(self.agent, "_has_entered_produce", False))
            )
            if returning_home_hold:
                click_coords = None
            elif chosen_action is not None and chosen_action.action_type != "wait":
                click_coords = chosen_action.bbox.center()
            elif obs.scene_type != "loading" and chosen_action is None:
                click_coords = (540, 960)

            is_popup_generic_fallback = False
            if (
                obs.scene_type == "popup"
                and chosen_action is not None
                and click_coords is not None
            ):
                _p_act_name = str((chosen_action.payload or {}).get("name") or "")
                _p_act_desc = str(chosen_action.description or "")
                is_popup_generic_fallback = (
                    _p_act_name == "点击屏幕继续"
                    or (_p_act_name == "閉じる" and "关闭弹窗或教程提示" in _p_act_desc)
                )
            if is_popup_generic_fallback and click_coords is not None:
                _p_fb_name = str((chosen_action.payload or {}).get("name") or "")
                _p_fb_key = (int(obs.week or 0), _p_fb_name, click_coords)
                if self._popup_fallback_key == _p_fb_key:
                    self._popup_fallback_streak += 1
                else:
                    self._popup_fallback_key = _p_fb_key
                    self._popup_fallback_streak = 1
                if self._popup_fallback_streak >= 4:
                    fw, fh = self._last_frame_size
                    blank_xy = (int(round(fw * 0.93)), int(round(fh * 0.26)))
                    logger.warning(
                        "Popup fallback tap stalled (%d steps at %s -> %s); tapping blank area (%d, %d).",
                        self._popup_fallback_streak,
                        _p_fb_name,
                        click_coords,
                        blank_xy[0],
                        blank_xy[1],
                    )
                    click_coords = blank_xy
                    self._popup_fallback_streak = 0
                    self._popup_fallback_key = None
            else:
                self._popup_fallback_streak = 0
                self._popup_fallback_key = None

            click_ok = True
            card_play_confirmed = True
            if click_coords is not None:
                already_selected = bool(
                    chosen_action is not None
                    and chosen_action.action_type == "play_card"
                    and any(
                        getattr(c, "selected", False) and c.bbox.center() == click_coords
                        for c in (obs.cards or [])
                    )
                ) or bool(
                    obs.scene_type == "produce_event"
                    and chosen_action is not None
                    and bool((chosen_action.payload or {}).get("is_selected"))
                )
                confirm_click_coords = click_coords
                if (
                    is_live_ctrl
                    and isinstance(self.controller, (AdbController, Win32Controller))
                    and chosen_action is not None
                    and chosen_action.action_type == "play_card"
                    and frame is not None
                    and frame.size > 0
                ):
                    fh_c, fw_c = frame.shape[:2]
                    n_hand_slots = max(len(obs.cards or []), int(getattr(obs, "hand_slot_count", 0) or 0))
                    tap_x = click_coords[0]
                    if n_hand_slots >= 5:
                        # Clamp 5-card fanned outer slots inward away from screen borders / neighbor overlap
                        tap_x = max(int(round(fw_c * 0.135)), min(int(round(fw_c * 0.865)), tap_x))
                    lifted_y = max(int(round(fh_c * 0.75)), click_coords[1] - int(round(fh_c * 0.022)))
                    click_coords = (tap_x, lifted_y if already_selected else click_coords[1])
                    confirm_click_coords = (tap_x, lifted_y)

                def _slot_still_selected_in_frame(f_chk: Optional[np.ndarray], cx_chk: int) -> bool:
                    if f_chk is None or f_chk.size == 0 or len(f_chk.shape) < 3:
                        return False
                    hp_s, wp_s = f_chk.shape[:2]
                    sx1 = max(0, cx_chk - int(round(wp_s * 0.075)))
                    sx2 = min(wp_s, cx_chk + int(round(wp_s * 0.075)))
                    if sx2 <= sx1 + 4:
                        return False
                    rib = f_chk[int(hp_s * 0.906) : int(hp_s * 0.930), sx1:sx2]
                    rib_orange = int(
                        ((rib[:, :, 0] > 225) & (rib[:, :, 1] > 140) & (rib[:, :, 1] < 215) & (rib[:, :, 2] < 95)).sum()
                    ) if rib.size else 0
                    sel_band = f_chk[int(hp_s * 0.902) : int(hp_s * 0.934), sx1:sx2]
                    badge_cnt = int(
                        ((sel_band[:, :, 0] > 215) & (sel_band[:, :, 1] > 150) & (sel_band[:, :, 2] < 95)).sum()
                    ) if sel_band.size else 0
                    return rib_orange > 450 or badge_cnt >= 220

                pre_play_hand: Optional[np.ndarray] = None
                if (
                    is_live_ctrl
                    and chosen_action is not None
                    and chosen_action.action_type == "play_card"
                    and frame is not None
                    and frame.size > 0
                ):
                    fh_p, fw_p = frame.shape[:2]
                    pre_play_hand = frame[int(fh_p * 0.73) : int(fh_p * 0.90), int(fw_p * 0.04) : int(fw_p * 0.96)].copy()
                with self._ctrl_lock:
                    click_ok = self.controller.click(click_coords[0], click_coords[1])
                chosen_name = str((chosen_action.payload or {}).get("name") or "") if chosen_action is not None else ""
                chosen_slot_key = str((chosen_action.payload or {}).get("shop_slot_key") or "") if chosen_action is not None else ""
                is_drink_discard_click = chosen_name.startswith("破棄候補_")
                is_preview_only = bool(chosen_action is not None and (chosen_action.payload or {}).get("preview_only"))
                needs_double_tap = (
                    is_live_ctrl
                    and chosen_action is not None
                    and not already_selected
                    and not is_preview_only
                    and not is_drink_discard_click
                    and (
                        chosen_action.action_type == "play_card"
                        or (
                            chosen_action.action_type == "pick_option"
                            and obs.scene_type in ("produce_schedule", "produce_event")
                        )
                    )
                )
                if click_ok and needs_double_tap:
                    time.sleep(0.75 if obs.scene_type in ("produce_schedule", "produce_event") else 0.65)
                    tap2_xy = confirm_click_coords if chosen_action.action_type == "play_card" else click_coords
                    with self._ctrl_lock:
                        self.controller.click(tap2_xy[0], tap2_xy[1])
                if click_ok and is_live_ctrl and chosen_action is not None and chosen_action.action_type in ("play_card", "use_drink", "end_turn"):
                    if chosen_action.action_type == "play_card":
                        dsl_l = (chosen_action.payload.get("dsl_actions") or "").lower()
                        is_extra_turn = (
                            "turnsremaining+=" in dsl_l
                            or chosen_action.payload.get("card_id") in (170, 171, 728)
                        )
                        is_continuous_play = (not is_extra_turn) and (
                            "cardusesremaining+=" in dsl_l
                            or chosen_action.payload.get("card_id") in (755, 756, 757, 758)
                        )
                        # Verify the card actually launched (not swallowed AND not stuck lifted in '選択中' state)
                        verif_elapsed = 0.0
                        if pre_play_hand is not None and isinstance(self.controller, (AdbController, Win32Controller)):
                            time.sleep(0.95)
                            verif_elapsed += 0.95
                            with self._ctrl_lock:
                                post_f = self.controller.screencap()
                            if post_f is not None and post_f.shape == frame.shape:
                                fh_p, fw_p = post_f.shape[:2]
                                post_hand = post_f[int(fh_p * 0.73) : int(fh_p * 0.90), int(fw_p * 0.04) : int(fw_p * 0.96)]
                                play_diff = float(np.mean(np.abs(post_hand.astype(np.float32) - pre_play_hand.astype(np.float32))))
                                stuck_selected = _slot_still_selected_in_frame(post_f, confirm_click_coords[0])
                                if play_diff < 5.5 or stuck_selected:
                                    logger.info(
                                        "Play card unlaunched (diff=%.1f, stuck_selected=%s); retrying confirm tap at %s...",
                                        play_diff,
                                        stuck_selected,
                                        confirm_click_coords,
                                    )
                                    if stuck_selected:
                                        time.sleep(0.35)
                                        with self._ctrl_lock:
                                            self.controller.click(confirm_click_coords[0], confirm_click_coords[1])
                                        time.sleep(0.75)
                                    else:
                                        time.sleep(0.90)
                                        with self._ctrl_lock:
                                            self.controller.click(click_coords[0], click_coords[1])
                                        time.sleep(0.65)
                                        with self._ctrl_lock:
                                            self.controller.click(confirm_click_coords[0], confirm_click_coords[1])
                                        time.sleep(0.75)
                                    with self._ctrl_lock:
                                        post_f2 = self.controller.screencap()
                                    if post_f2 is not None and post_f2.shape == frame.shape:
                                        post_hand2 = post_f2[int(fh_p * 0.73) : int(fh_p * 0.90), int(fw_p * 0.04) : int(fw_p * 0.96)]
                                        play_diff2 = float(np.mean(np.abs(post_hand2.astype(np.float32) - pre_play_hand.astype(np.float32))))
                                        stuck_selected2 = _slot_still_selected_in_frame(post_f2, confirm_click_coords[0])
                                        if play_diff2 < 5.5 or stuck_selected2:
                                            with self._ctrl_lock:
                                                self.controller.click(confirm_click_coords[0], confirm_click_coords[1])
                                            time.sleep(0.90)
                                            with self._ctrl_lock:
                                                post_f3 = self.controller.screencap()
                                            if post_f3 is not None and post_f3.shape == frame.shape:
                                                post_hand3 = post_f3[int(fh_p * 0.73) : int(fh_p * 0.90), int(fw_p * 0.04) : int(fw_p * 0.96)]
                                                play_diff3 = float(np.mean(np.abs(post_hand3.astype(np.float32) - pre_play_hand.astype(np.float32))))
                                                stuck_selected3 = _slot_still_selected_in_frame(post_f3, confirm_click_coords[0])
                                                if play_diff3 < 5.5 or stuck_selected3:
                                                    logger.warning(
                                                        "Card play unconfirmed after retry (diff=%.1f, stuck_selected=%s); skipping tracker/sim play record so card is not falsely marked exhausted.",
                                                        play_diff3,
                                                        stuck_selected3,
                                                    )
                                                    card_play_confirmed = False
                        total_play_wait = 4.60 if is_extra_turn else (3.70 if is_continuous_play else 4.20)
                        rem_wait = max(0.0, total_play_wait - verif_elapsed) if card_play_confirmed else 0.50
                        if rem_wait > 0:
                            time.sleep(rem_wait)
                    elif chosen_action.action_type == "use_drink":
                        time.sleep(2.60)
                    else:
                        time.sleep(2.80)
                elif (
                    click_ok
                    and is_live_ctrl
                    and chosen_action is not None
                    and chosen_action.action_type == "pick_option"
                    and (obs.scene_type in ("card_reward", "shop", "popup") or (is_drink_discard_click and obs.scene_type == "produce_schedule"))
                ):
                    if obs.scene_type == "shop":
                        if chosen_name == "リフレッシュ" or chosen_slot_key == "リフレッシュ":
                            time.sleep(1.60)
                        elif (
                            chosen_name in ("交換する", "チェンジする", "チェンジ", "強化", "削除", "カスタマイズ", "セレクトチェンジ")
                            or chosen_slot_key in ("強化", "削除", "カスタマイズ", "セレクトチェンジ")
                            or is_drink_discard_click
                        ):
                            time.sleep(1.00)
                        else:
                            time.sleep(0.65)
                    elif obs.scene_type == "popup" and chosen_name in (
                        "交換", "交換する", "チェンジ", "チェンジする", "強化", "強化する", "削除", "削除する", "捨てる", "破棄", "破棄する", "決定", "更新する"
                    ):
                        time.sleep(1.35 if chosen_name in ("交換", "交換する", "チェンジ", "チェンジする", "決定", "更新する") else 1.00)
                    elif obs.scene_type == "popup" and chosen_name in ("次へ", "閉じる"):
                        time.sleep(0.85)
                    elif is_drink_discard_click and obs.scene_type in ("card_reward", "produce_schedule"):
                        time.sleep(0.90)
                    else:
                        time.sleep(0.45)
                if not click_ok:
                    logger.warning("Controller click failed at coords: %s", click_coords)
            else:
                time.sleep(1.0)
            self._last_heartbeat_ts = time.monotonic()

            # Commit LLM sync snapshot BEFORE applying this step's deck/battle mutations,
            # so any resulting deck additions/upgrades/generated cards surface cleanly in next turn's delta.
            if self.session_mode and sync_meta is not None and decision.source == "llm":
                self.prompt_builder.commit_session_sync(self.tracker, sync_meta)

            # Track played/acquired cards & drinks across battle & produce deck
            pending_before = len(self.tracker.pending_top_deck)
            if chosen_action and chosen_action.action_type == "play_card":
                card_id = chosen_action.payload.get("card_id")
                if card_id is not None and card_play_confirmed:
                    self.tracker.record_card_play(
                        card_id,
                        limit_count=chosen_action.payload.get("limit_count", 1 if self.mode == "mock" else None),
                        base_id=chosen_action.payload.get("base_id"),
                        dsl_actions=chosen_action.payload.get("dsl_actions", ""),
                        db=self.db,
                    )
            elif chosen_action and chosen_action.action_type == "use_drink":
                drink_id = chosen_action.payload.get("drink_id")
                if drink_id is not None:
                    self.tracker.record_drink_use(
                        int(drink_id),
                        dsl_actions=chosen_action.payload.get("dsl_actions", ""),
                        db=self.db,
                    )
            elif chosen_action and chosen_action.action_type == "end_turn":
                self.tracker.record_turn_end(db=self.db)
            elif chosen_action and chosen_action.action_type == "pick_option" and obs.scene_type in ("card_reward", "shop"):
                picked_name = chosen_action.payload.get("name", "")
                picked_cid = chosen_action.payload.get("card_id")
                is_dup_reward = (
                    obs.scene_type == "card_reward"
                    and picked_name == "受け取る"
                    and self._claimed_card_in_reward_scene
                )
                is_shop_card_change = (
                    obs.scene_type == "shop"
                    and (
                        picked_name in ("チェンジする", "チェンジ")
                        or (chosen_action.payload or {}).get("shop_action_type") == "change"
                        or bool((chosen_action.payload or {}).get("is_shop_change_card"))
                    )
                    and picked_cid
                )
                if is_shop_card_change:
                    self._pending_change_pickup_cid = int(picked_cid)
                elif picked_name in ("受け取る", "交換する") and picked_cid and not is_dup_reward:
                    self._pending_change_pickup_cid = None
                    self.tracker.add_card_to_deck(int(picked_cid), db=self.db)
                    self._flush_produce_library()
                    if obs.scene_type == "card_reward" and picked_name == "受け取る":
                        self._claimed_card_in_reward_scene = True
                elif (
                    obs.scene_type == "card_reward"
                    and picked_name == "次へ"
                    and picked_cid
                    and "チェンジで獲得" in (obs.raw_text or "")
                ):
                    self._pending_change_pickup_cid = int(picked_cid)
                elif obs.scene_type == "card_reward" and picked_name in ("削除する", "チェンジする") and picked_cid:
                    cid_int = int(picked_cid)
                    removed = False
                    removed_cid: Optional[int] = None
                    if cid_int in self.tracker.produce_deck:
                        self.tracker.produce_deck.remove(cid_int)
                        removed = True
                        removed_cid = cid_int
                    elif self.db is not None:
                        crow = self.db.get_card_by_id(cid_int)
                        target_base = int(crow.get("base_id") or cid_int) if crow else cid_int
                        for idx, dcid in enumerate(self.tracker.produce_deck):
                            drow = self.db.get_card_by_id(int(dcid))
                            dbase = int(drow.get("base_id") or dcid) if drow else int(dcid)
                            if dbase == target_base:
                                removed_cid = int(dcid)
                                del self.tracker.produce_deck[idx]
                                removed = True
                                break
                    added_change = False
                    if picked_name == "チェンジする" and self._pending_change_pickup_cid:
                        gain_cid = int(self._pending_change_pickup_cid)
                        if self.db is not None:
                            rem_check_cid = removed_cid if removed_cid is not None else cid_int
                            rem_row = self.db.get_card_by_id(rem_check_cid)
                            picked_row = self.db.get_card_by_id(cid_int)
                            removed_was_upgraded = bool(
                                (rem_row and rem_row.get("upgraded"))
                                or (picked_row and picked_row.get("upgraded"))
                            )
                            if removed_was_upgraded:
                                gain_row = self.db.get_card_by_id(gain_cid)
                                if gain_row and not gain_row.get("upgraded") and gain_row.get("upgraded_id"):
                                    gain_cid = int(gain_row["upgraded_id"])
                        self.tracker.add_card_to_deck(gain_cid, db=self.db)
                        added_change = True
                    self._pending_change_pickup_cid = None
                    if removed or added_change:
                        self._flush_produce_library()
                elif picked_name in ("キャンセル", "戻る", "スキップ", "受け取らない", "終了"):
                    self._pending_change_pickup_cid = None
                elif picked_name == "強化する" and picked_cid:
                    cid_int = int(picked_cid)
                    c_row = self.db.get_card_by_id(cid_int)
                    upg_id = int(c_row["upgraded_id"]) if (c_row and c_row.get("upgraded_id")) else None
                    if upg_id and self.tracker.upgrade_card_in_deck(cid_int, upg_id):
                        self._flush_produce_library()
                elif obs.scene_type == "shop":
                    self._pending_change_pickup_cid = None
                if obs.scene_type == "card_reward" and picked_name == "再抽選":
                    reco_inner = self._get_inner_recognizer()
                    if reco_inner is not None and hasattr(reco_inner, "_reward_visited_slots"):
                        reco_inner._reward_visited_slots.clear()
                elif obs.scene_type == "card_reward" and chosen_action.payload.get("slot_idx") is not None:
                    reco_inner = self._get_inner_recognizer()
                    if reco_inner is not None and hasattr(reco_inner, "_reward_visited_slots"):
                        reco_inner._reward_visited_slots.add(int(chosen_action.payload["slot_idx"]))
                if obs.scene_type == "shop" or (obs.scene_type == "popup" and picked_name in ("更新する", "リフレッシュ")):
                    shop_slot_key = (chosen_action.payload or {}).get("shop_slot_key")
                    reco_inner = self._get_inner_recognizer()
                    if picked_name in ("リフレッシュ", "更新する") or str(shop_slot_key or "") == "リフレッシュ":
                        if reco_inner is not None:
                            if hasattr(reco_inner, "_shop_visited_slots"):
                                reco_inner._shop_visited_slots.clear()
                            if hasattr(reco_inner, "_shop_last_clicked_slot"):
                                reco_inner._shop_last_clicked_slot = None
                    elif obs.scene_type == "shop" and shop_slot_key and ":" in str(shop_slot_key):
                        if reco_inner is not None:
                            if hasattr(reco_inner, "_shop_visited_slots"):
                                reco_inner._shop_visited_slots.add(str(shop_slot_key))
                            if hasattr(reco_inner, "_shop_last_clicked_slot"):
                                reco_inner._shop_last_clicked_slot = str(shop_slot_key)
                if obs.scene_type == "shop" and (picked_name in ("交換する", "チェンジする", "チェンジ", "強化", "リフレッシュ") or str((chosen_action.payload or {}).get("shop_slot_key") or "") == "リフレッシュ"):
                    if chosen_action.payload.get("price"):
                        spent = int(chosen_action.payload["price"])
                        base_p = (
                            self._last_stable_shop_p_points
                            if self._last_stable_shop_p_points is not None
                            else self.tracker.p_points
                        )
                        exp_p = max(0, int(base_p) - spent)
                        self._last_shop_expected_p = exp_p
                        self.tracker.p_points = exp_p
                        reco_inner = self._get_inner_recognizer()
                        if reco_inner is not None and hasattr(reco_inner, "_last_p_points"):
                            reco_inner._last_p_points = exp_p
                    if (picked_name == "リフレッシュ" or str((chosen_action.payload or {}).get("shop_slot_key") or "") == "リフレッシュ") and self.mode == "adb":
                        time.sleep(0.8)
            elif chosen_action and chosen_action.action_type == "pick_option" and obs.scene_type == "popup" and self.tracker.in_battle:
                p_cancel = str((chosen_action.payload or {}).get("name") or "")
                if p_cancel in ("キャンセル", "閉じる"):
                    self.tracker.genki = int(obs.genki or 0)

            if self.session_mode and chosen_action is not None:
                if decision.source == "rule_skip":
                    self.tracker.record_session_event(f"系统自动跳过(rule_skip): {chosen_action.description}")
                elif decision.source == "fallback":
                    self.tracker.record_session_event(f"兜底策略代选: {chosen_action.description}")
                if len(self.tracker.pending_top_deck) > pending_before:
                    self.tracker.record_session_event(
                        f"上步打出《{chosen_action.payload.get('name', '')}》触发生成牌并置于牌库顶"
                    )
                if chosen_action.action_type == "pick_option":
                    p_name = str((chosen_action.payload or {}).get("name") or "")
                    if p_name.startswith("破棄候補_"):
                        self.tracker.record_session_event("已点击饮料栏旧饮料打开详情，下一步若确认删除请选择『捨てる/破棄』")
                    elif p_name in ("捨てる", "破棄", "破棄する"):
                        self.tracker.record_session_event("已确认删除旧饮料腾出空位，现在可选择并领取新饮料")
                    elif p_name == "受け取らない":
                        self.tracker.record_session_event("已勾选『受け取らない』，下一步请点击底部确认按钮跳过领取")

            if obs.scene_type == "battle_turn" and hasattr(self.agent, "notify_battle_action"):
                if not (chosen_action and chosen_action.action_type == "play_card" and not card_play_confirmed):
                    self.agent.notify_battle_action(chosen_action)

            if self.on_action_executed:
                try:
                    self.on_action_executed(chosen_action, click_coords)
                except Exception as cb_err:
                    logger.error("Error in on_action_executed callback: %s", cb_err)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_scene_type = obs.scene_type

            result = StepResult(
                step_idx=step_idx,
                success=True,
                scene_type=obs.scene_type,
                observation=obs,
                legal_actions=legal_actions,
                decision=decision,
                action_taken=chosen_action,
                click_coords=click_coords,
                elapsed_ms=round(elapsed_ms, 2),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception("Step %d execution error: %s", step_idx, e)
            if self.on_error:
                try:
                    self.on_error(e, {"step_idx": step_idx})
                except Exception as cb_err:
                    logger.error("Error in on_error callback: %s", cb_err)

            result = StepResult(
                step_idx=step_idx,
                success=False,
                error=str(e),
                elapsed_ms=round(elapsed_ms, 2),
            )

        self.history.append(result)
        if result.success and self.checkpoint_path is not None:
            self.save_checkpoint()
        if self.on_step_completed:
            try:
                self.on_step_completed(result)
            except Exception as cb_err:
                logger.error("Error in on_step_completed callback: %s", cb_err)

        return result

    def run_loop(
        self,
        max_steps: int = 10,
        stop_on_scene: Optional[Union[str, List[str]]] = None,
        delay_between_steps: float = 0.5,
        stop_on_home_return: bool = False,
    ) -> List[StepResult]:
        """Run execution loop up to max_steps or until stop_on_scene / home return is reached."""
        stop_scenes = set([stop_on_scene] if isinstance(stop_on_scene, str) else (stop_on_scene or []))
        results: List[StepResult] = []
        entered_produce = bool(self._has_entered_produce or getattr(self.agent, "_has_entered_produce", False))

        if self.state != RunnerState.RUNNING:
            if not self.start():
                return results

        for _ in range(max_steps):
            if self.state == RunnerState.STOPPED:
                break

            step_res = self.run_once()
            results.append(step_res)

            if not step_res.success:
                logger.warning("Step %d failed: %s. Stopping loop.", step_res.step_idx, step_res.error)
                break

            if self._retrying_loadout_via_home:
                entered_produce = False
            else:
                entered_produce = bool(self._has_entered_produce or getattr(self.agent, "_has_entered_produce", False))

            if (
                stop_on_home_return
                and entered_produce
                and not self._retrying_loadout_via_home
                and step_res.scene_type in ("home", "produce_setup")
            ):
                logger.info(
                    "Returned cleanly to %s after produce run at step %d. Stopping loop.",
                    step_res.scene_type,
                    step_res.step_idx,
                )
                break

            if step_res.scene_type in stop_scenes:
                logger.info("Reached stop scene '%s' at step %d.", step_res.scene_type, step_res.step_idx)
                break

            if delay_between_steps > 0:
                time.sleep(delay_between_steps)

        return results
