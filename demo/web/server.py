"""
Local HTTP/SSE Web Server for the Gakumas MFAAvalonia-style Frontend UI.

Uses Python standard library http.server.ThreadingHTTPServer (zero extra pip dependencies)
so the dual-mode Mock/ADB/Win32 card-playing workbench runs immediately out-of-the-box.
Keeps root assets (cards/, drinks/, gakumas_library.db, etc.) 100% read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import mimetypes
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image

from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.scenarios import SCENARIO_REGISTRY, get_scenario_spec
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.adb_controller import AdbConfig, AdbController
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.runner import GakumasRunner, StepResult
from demo.maa_bridge.win32_controller import Win32Config, Win32Controller
from demo.schemas import (
    BoundingBox,
    DecisionResult,
    LegalAction,
    ScreenObservation,
)
from demo.vision.mock_generator import MockFrameGenerator
from demo.vision.recognizer import recognize_screen
from demo.web.hif_session import HifWorkbenchSession

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
ALLOWED_ASSET_ROOTS = (
    (PROJECT_ROOT / "cards").resolve(),
    (PROJECT_ROOT / "drinks").resolve(),
)


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def _to_asset_url(raw_path: Optional[str]) -> Optional[str]:
    """Convert a card/drink local path into a safe read-only /assets/... URL."""
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    for root in ALLOWED_ASSET_ROOTS:
        if resolved.is_relative_to(root) and resolved.is_file():
            rel = resolved.relative_to(PROJECT_ROOT.resolve())
            return f"/assets/{rel.as_posix()}"
    return None


def _encode_png(frame_rgb: np.ndarray) -> Tuple[bytes, int, int]:
    """Encode an RGB uint8 numpy frame as PNG bytes and return (bytes, width, height)."""
    if frame_rgb.ndim == 2:
        img = Image.fromarray(frame_rgb, mode="L").convert("RGB")
    else:
        img = Image.fromarray(frame_rgb[:, :, :3].astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    w, h = img.size
    return buf.getvalue(), w, h


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) * 0.65))


@dataclass
class InstanceConfig:
    mode: str = "mock"  # 'mock' | 'adb' | 'win32'
    plan: str = "sense"  # 'sense' | 'logic' | 'anomaly'
    scenario_id: str = "hajime_regular"  # 'hajime_regular' | 'hajime_pro' | 'hajime_master' | 'nia_master'
    p_idol_id: Optional[int] = None
    current_week: int = 1
    scenario_preset: str = "hajime_regular"  # 'hajime_regular' | 'sequence' | 'battle_turn' | 'produce_schedule' | 'card_reward' | 'shop' | 'popup'
    steps: int = 5
    delay: float = 0.5
    auto_progress: bool = True
    fallback_to_mock: bool = True
    adb_host: str = "192.168.0.115"
    adb_port: int = 5555
    adb_root_pwd: str = "root/114514"
    win32_title: str = "学园偶像大师"
    llm_mode: str = "auto_xai"  # 'auto_xai' | 'custom' | 'offline'
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    tasks_enabled: Dict[str, bool] = field(
        default_factory=lambda: {
            "produce_schedule": True,
            "card_reward": True,
            "battle_turn": True,
            "shop": True,
            "popup": True,
        }
    )

    def to_public_dict(self, resolved_llm: AgentConfig) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "plan": self.plan,
            "scenario_id": self.scenario_id,
            "p_idol_id": self.p_idol_id,
            "current_week": self.current_week,
            "scenario_preset": self.scenario_preset,
            "steps": self.steps,
            "delay": self.delay,
            "auto_progress": self.auto_progress,
            "fallback_to_mock": self.fallback_to_mock,
            "adb_host": self.adb_host,
            "adb_port": self.adb_port,
            "adb_root_pwd": self.adb_root_pwd,
            "win32_title": self.win32_title,
            "llm_mode": self.llm_mode,
            "llm_base_url": self.llm_base_url or resolved_llm.base_url or "",
            "llm_model": self.llm_model or resolved_llm.model,
            "has_api_key": bool(resolved_llm.api_key),
            "api_key_masked": _mask_key(resolved_llm.api_key or ""),
            "tasks_enabled": dict(self.tasks_enabled),
        }


class InstanceSession:
    """Isolated runtime state for a single MFAAvalonia-style instance tab."""

    def __init__(self, instance_id: str, name: str, config: InstanceConfig, db: GakumasDB):
        self.id = instance_id
        self.name = name
        self.config = config
        self.db = db
        self.lock = threading.RLock()

        resolved_pid = config.p_idol_id if config.p_idol_id is not None else (
            126 if (config.plan or "").lower() == "logic" else None
        )
        self.tracker = GameStateTracker(plan=config.plan, p_idol_id=resolved_pid)
        self.prompt_builder = PromptBuilder(db=db)

        self.running: bool = False
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        self.step_count: int = 0
        self.frame_seq: int = 0
        self.frame_png_bytes: bytes = b""
        self.frame_width: int = 1080
        self.frame_height: int = 1920
        self.frame_source: str = "mock"
        self.last_raw_frame: Optional[np.ndarray] = None
        self._last_gt_obs: Optional[ScreenObservation] = None

        self.observation: Optional[ScreenObservation] = None
        self.prompt_text: str = ""
        self.est_tokens: int = 0
        self.legal_actions: List[LegalAction] = []
        self.decision: Optional[DecisionResult] = None
        self.click_coords: Optional[Tuple[int, int]] = None
        self.last_error: str = ""

        self.logs: List[Dict[str, Any]] = []
        self.snapshots: List[Dict[str, Any]] = []
        self._controller: Optional[Any] = None
        self._agent: Optional[AgentDecisionMaker] = None
        self._agent_sig: Optional[Tuple[Any, ...]] = None
        self._runner: Optional[GakumasRunner] = None
        self._custom_battle_overrides: Optional[Dict[str, Any]] = None

    def add_log(self, level: str, message: str, meta: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "id": f"{int(time.time() * 1000)}-{len(self.logs)}",
            "time": _now_hms(),
            "level": level,
            "message": message,
            "meta": meta or {},
        }
        self.logs.append(entry)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]

    def resolve_agent_config(self) -> AgentConfig:
        mode = self.config.llm_mode
        if mode == "offline":
            return AgentConfig(
                base_url=None,
                api_key=None,
                model="local-heuristic",
                enable_rule_skip=True,
                fallback_on_error=True,
            )
        if mode == "custom":
            base_cfg = AgentConfig.from_env_or_xai()
            return AgentConfig(
                base_url=self.config.llm_base_url.strip() or base_cfg.base_url,
                api_key=self.config.llm_api_key.strip() or base_cfg.api_key,
                model=self.config.llm_model.strip() or base_cfg.model,
                enable_rule_skip=True,
                fallback_on_error=True,
            )
        # auto_xai
        return AgentConfig.from_env_or_xai()

    def _checkpoint_file(self) -> Optional[Path]:
        # Web UI runs keep state in-memory so demo/state/ is never created on disk.
        return None

    def reset_controller(self, clear_checkpoint: bool = False) -> None:
        if self._controller is not None:
            try:
                self._controller.disconnect()
            except Exception:
                pass
            self._controller = None
        self._runner = None
        if self._agent is not None:
            self._agent.reset_session()
            self._agent = None
        self._agent_sig = None
        self.tracker.reset_llm_session()
        if clear_checkpoint:
            ckpt = self._checkpoint_file()
            if ckpt is not None and ckpt.exists():
                try:
                    ckpt.unlink()
                except Exception:
                    pass
        if self.config.mode != "mock":
            self.frame_source = "disconnected"

    def _build_controller(self) -> Any:
        if self.config.mode == "adb":
            return AdbController(
                config=AdbConfig(
                    host=self.config.adb_host,
                    port=int(self.config.adb_port),
                    root_pwd=self.config.adb_root_pwd,
                    connect_timeout=2.5,
                    command_timeout=3.5,
                    retry_count=1,
                    fallback_to_mock=bool(self.config.fallback_to_mock),
                )
            )
        if self.config.mode == "win32":
            return Win32Controller(
                config=Win32Config(
                    window_title=self.config.win32_title,
                    headless_fallback=bool(self.config.fallback_to_mock),
                )
            )
        ctrl = MockController(
            plan=self.config.plan,
            auto_progress=bool(self.config.auto_progress),
            db=self.db,
        )
        ctrl.connect()
        self._apply_preset_to_mock_controller(ctrl)
        return ctrl

    def _default_battle_kwargs(self) -> Dict[str, Any]:
        default_buffs = {
            "sense": {"good_condition": 4, "concentration": 2},
            "logic": {"good_impression": 5, "motivation": 3},
            "anomaly": {"strong_spirit": 2, "full_effort": 4},
        }.get(self.config.plan, {"good_condition": 3})
        return {
            "stamina": 24,
            "max_stamina": 30,
            "genki": 6,
            "turn": 3,
            "turns_remaining": 7,
            "score": 1450,
            "target_score": 5000,
            "buffs": default_buffs,
        }

    def _apply_preset_to_mock_controller(self, ctrl: MockController) -> None:
        preset = self.config.scenario_preset
        if preset == "hajime_regular":
            ctrl.set_week(week=self.config.current_week, scenario_id=self.config.scenario_id)
            return
        if preset == "sequence":
            # Full produce loop: schedule -> card reward -> shop -> popup -> battle (then natural turn progression)
            ctrl.set_scenario("produce_schedule")
            ctrl.queue_scenarios(
                [
                    "card_reward",
                    "shop",
                    "popup",
                    {"scene_type": "battle_turn", **self._default_battle_kwargs()},
                ]
            )
            return
        if preset == "battle_turn":
            kwargs: Dict[str, Any] = self._default_battle_kwargs()
            if self._custom_battle_overrides:
                kwargs.update(self._custom_battle_overrides)
            ctrl.set_scenario("battle_turn", **kwargs)
        elif preset in ("produce_schedule", "card_reward", "shop", "popup"):
            ctrl.set_scenario(preset)

    def _wrap_controller_screencap(self, ctrl: Any) -> Any:
        if getattr(ctrl, "_web_wrapped", False):
            return ctrl
        orig_screencap = ctrl.screencap

        def _tracked_screencap() -> np.ndarray:
            arr = orig_screencap()
            gt = getattr(ctrl, "current_obs", None)
            self._last_gt_obs = gt.model_copy(deep=True) if gt is not None else None
            self.last_raw_frame = arr
            png_bytes, w, h = _encode_png(arr)
            self.frame_png_bytes = png_bytes
            self.frame_width = w
            self.frame_height = h
            self.frame_seq += 1
            if getattr(ctrl, "_mock_delegate", None) is not None:
                self.frame_source = f"{self.config.mode}_fallback"
            else:
                self.frame_source = self.config.mode
            return arr

        ctrl.screencap = _tracked_screencap  # type: ignore[method-assign]
        ctrl._web_wrapped = True
        return ctrl

    def ensure_runner(self) -> GakumasRunner:
        if self._controller is None:
            self._controller = self._wrap_controller_screencap(self._build_controller())
            if not self._controller.is_connected:
                if not self._controller.connect():
                    self.frame_source = "disconnected"
                    raise RuntimeError(
                        f"无法连接 {self.config.mode.upper()} 控制器 "
                        f"({self.config.adb_host}:{self.config.adb_port})"
                    )
        agent_cfg = self.resolve_agent_config()
        agent_sig = (agent_cfg.base_url, agent_cfg.api_key, agent_cfg.model, self.config.llm_mode)
        if self._agent is None or self._agent_sig != agent_sig:
            self._agent = AgentDecisionMaker(config=agent_cfg)
            self._agent_sig = agent_sig
            self.tracker.reset_llm_session()
            self._runner = None

        use_session = self.config.llm_mode != "offline"
        if (
            self._runner is not None
            and self._runner.controller is self._controller
            and self._runner.agent is self._agent
            and self._runner.plan == self.config.plan
            and self._runner.scenario_id == self.config.scenario_id
            and self._runner.session_mode == use_session
        ):
            return self._runner

        def on_obs(obs: ScreenObservation) -> None:
            gt = self._last_gt_obs
            if gt is not None and gt.scene_type == obs.scene_type:
                if gt.options:
                    obs.options = [dict(o) for o in gt.options]
                if gt.buffs and not obs.buffs:
                    obs.buffs = dict(gt.buffs)
            self.observation = obs

        def on_prompt(prompt: str, actions: List[LegalAction]) -> None:
            self.prompt_text = prompt
            self.est_tokens = _estimate_tokens(prompt)
            self.legal_actions = list(actions)

        def on_decision(dec: DecisionResult, chosen: Optional[LegalAction]) -> None:
            self.decision = dec

        def on_action(chosen: Optional[LegalAction], coords: Tuple[int, int]) -> None:
            self.click_coords = coords

        ckpt_file = self._checkpoint_file()
        self._runner = GakumasRunner(
            mode=self.config.mode,
            controller=self._controller,
            agent=self._agent,
            db=self.db,
            tracker=self.tracker,
            prompt_builder=self.prompt_builder,
            plan=self.config.plan,
            scenario_id=self.config.scenario_id,
            p_idol_id=self.tracker.p_idol_id,
            session_mode=use_session,
            checkpoint_path=ckpt_file,
            auto_resume=True,
            on_observation=on_obs,
            on_prompt_built=on_prompt,
            on_decision=on_decision,
            on_action_executed=on_action,
        )
        if ckpt_file is not None and ckpt_file.exists():
            if self._runner.load_checkpoint():
                self.step_count = max(self.step_count, self._runner.current_step)
        return self._runner

    def capture_preview(self) -> Dict[str, Any]:
        """Capture current screen and update observation + prompt without executing a click."""
        with self.lock:
            try:
                runner = self.ensure_runner()
                frame = runner.controller.screencap()
                gt_obs = getattr(runner.controller, "current_obs", None)
                if gt_obs is not None:
                    obs = gt_obs.model_copy(update={"plan": self.config.plan})
                else:
                    obs = recognize_screen(frame, plan=self.config.plan, db=self.db)

                prompt, actions = self.prompt_builder.build_prompt(obs, tracker=self.tracker)
                self.observation = obs
                self.prompt_text = prompt
                self.est_tokens = _estimate_tokens(prompt)
                self.legal_actions = actions
                self.decision = None
                self.click_coords = None
                self.last_error = ""
                self.add_log(
                    "info",
                    f"已捕获预览帧 ({self.frame_width}x{self.frame_height}, source={self.frame_source}) | 场景: {obs.scene_type} | 合法动作: {len(actions)}",
                )
                return {"ok": True, "scene_type": obs.scene_type, "frame_seq": self.frame_seq, "frame_source": self.frame_source}
            except Exception as exc:
                self.frame_source = "disconnected"
                self.last_error = str(exc)
                self.add_log("error", f"抓取预览失败: {exc}")
                raise

    def execute_step(self) -> StepResult:
        """Execute one full recognize -> prompt -> decide -> click cycle."""
        with self.lock:
            runner = self.ensure_runner()
            result = runner.run_once()
            self.step_count += 1
            if getattr(runner.controller, "week_mode", False):
                self.config.current_week = int(getattr(runner.controller, "current_week", self.config.current_week))
            if result.observation is not None:
                self.observation = result.observation
            self.legal_actions = list(result.legal_actions)
            self.decision = result.decision
            self.click_coords = result.click_coords
            self.last_error = result.error or ""

            if not result.success:
                self.add_log("error", f"Step #{self.step_count} 执行失败: {result.error or 'unknown error'}")
                return result

            dec = result.decision
            if dec is None:
                self.add_log("info", f"Step #{self.step_count} 完成 (无决策返回)")
                return result

            chosen_desc = ""
            for act in result.legal_actions:
                if act.choice_id == dec.choice_id:
                    chosen_desc = act.description
                    break

            src_tag = dec.source.upper()
            scene_name = result.observation.scene_type if result.observation else result.scene_type
            self.add_log(
                dec.source,
                f"Step #{self.step_count} [{scene_name}] -> [{dec.choice_id}] {chosen_desc} "
                f"({dec.latency_ms:.0f}ms, tok {dec.prompt_tokens}+{dec.completion_tokens}) | {dec.reason}",
                meta={
                    "source": src_tag,
                    "choice_id": dec.choice_id,
                    "latency_ms": round(dec.latency_ms, 1),
                    "click": list(result.click_coords) if result.click_coords else None,
                },
            )

            obs = result.observation
            self.snapshots.append(
                {
                    "step": self.step_count,
                    "time": _now_hms(),
                    "scene_type": scene_name,
                    "turn": obs.turn if obs else 0,
                    "stamina": obs.stamina if obs else 0,
                    "max_stamina": obs.max_stamina if obs else 0,
                    "genki": obs.genki if obs else 0,
                    "score": obs.score if obs else 0,
                    "choice_id": dec.choice_id,
                    "source": dec.source,
                    "reason": dec.reason,
                    "latency_ms": round(dec.latency_ms, 1),
                }
            )
            if len(self.snapshots) > 24:
                self.snapshots = self.snapshots[-24:]
            return result

    def _enrich_cards_and_drinks(self, obs: Optional[ScreenObservation]) -> Optional[Dict[str, Any]]:
        if obs is None:
            return None
        obs_dict = obs.model_dump()
        for card in obs_dict.get("cards", []):
            row = self.db.get_card_by_id(card["card_id"])
            card["icon_url"] = _to_asset_url(row.get("icon_path")) if row else None
            card["details_url"] = _to_asset_url(row.get("details_path")) if row else None
            card["limit_count"] = row.get("limit_count", 0) if row else 0
            card["description"] = row.get("description", card.get("dsl_summary", "")) if row else ""
        for drink in obs_dict.get("drinks", []):
            row = self.db.get_drink_by_id(drink["drink_id"])
            drink["icon_url"] = _to_asset_url(row.get("icon_path")) if row else None
            drink["details_url"] = _to_asset_url(row.get("details_path")) if row else None
            drink["description"] = row.get("description", drink.get("dsl_summary", "")) if row else ""
        for opt in obs_dict.get("options", []):
            if "card_id" in opt and opt["card_id"]:
                row = self.db.get_card_by_id(int(opt["card_id"]))
                opt["icon_url"] = _to_asset_url(row.get("icon_path")) if row else None
                opt["details_url"] = _to_asset_url(row.get("details_path")) if row else None
            elif "drink_id" in opt and opt["drink_id"]:
                row = self.db.get_drink_by_id(int(opt["drink_id"]))
                opt["icon_url"] = _to_asset_url(row.get("icon_path")) if row else None
                opt["details_url"] = _to_asset_url(row.get("details_path")) if row else None
        return obs_dict

    def _enrich_actions(self, actions: List[LegalAction]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for act in actions:
            item = act.model_dump()
            payload = item.get("payload") or {}
            icon_url = None
            details_url = None
            card_id = payload.get("card_id")
            drink_id = payload.get("drink_id")
            if card_id:
                row = self.db.get_card_by_id(int(card_id))
                if row:
                    icon_url = _to_asset_url(row.get("icon_path"))
                    details_url = _to_asset_url(row.get("details_path"))
            elif drink_id:
                row = self.db.get_drink_by_id(int(drink_id))
                if row:
                    icon_url = _to_asset_url(row.get("icon_path"))
                    details_url = _to_asset_url(row.get("details_path"))
            item["icon_url"] = icon_url
            item["details_url"] = details_url
            enriched.append(item)
        return enriched

    def to_summary_dict(self) -> Dict[str, Any]:
        with self.lock:
            resolved_llm = self.resolve_agent_config()
            spec = get_scenario_spec(self.config.scenario_id)
            roadmap = [
                {
                    "week": wn.week,
                    "phase_key": wn.phase_key,
                    "title": wn.title,
                    "scene_type": wn.scene_type,
                    "activities": list(wn.activities),
                    "recommended_action": wn.recommended_action,
                    "operation_guide": wn.operation_guide,
                    "target_score": wn.target_score,
                }
                for wn in spec.weeks.values()
            ]
            return {
                "id": self.id,
                "name": self.name,
                "running": self.running,
                "step_count": self.step_count,
                "frame_seq": self.frame_seq,
                "frame_source": self.frame_source,
                "frame_meta": {"width": self.frame_width, "height": self.frame_height},
                "config": self.config.to_public_dict(resolved_llm),
                "scenario_roadmap": {
                    "scenario_id": spec.scenario_id,
                    "name": spec.name,
                    "short_name": spec.short_name,
                    "total_weeks": spec.total_weeks,
                    "stat_cap": spec.stat_cap,
                    "description": spec.description,
                    "weeks": roadmap,
                },
                "observation": self._enrich_cards_and_drinks(self.observation),
                "prompt_text": self.prompt_text,
                "est_tokens": self.est_tokens,
                "legal_actions": self._enrich_actions(self.legal_actions),
                "decision": self.decision.model_dump() if self.decision else None,
                "click_coords": list(self.click_coords) if self.click_coords else None,
                "tracker": {
                    "played_limit_cards": sorted(self.tracker.used_limit_cards),
                    "turn_history_count": len(self.tracker.cards_played_history),
                    "session_bootstrapped": self.tracker.session_bootstrapped,
                    "session_turns": self._agent.get_session_stats()["turns"] if self._agent else 0,
                    "synced_cards": len(self.tracker.llm_synced_card_ids),
                },
                "last_error": self.last_error,
                "logs": list(self.logs[-80:]),
                "snapshots": list(self.snapshots[-12:]),
            }


class WebSessionManager:
    """Manages multiple MFAAvalonia-style instances, SSE notifications, and DB library queries."""

    def __init__(self, db_path: Optional[str] = None, auto_preview: bool = True):
        self.db = GakumasDB(db_path=db_path or str(PROJECT_ROOT / "gakumas_library.db"))
        self.lock = threading.RLock()
        self._subscribers: List[queue.Queue] = []

        self.instances: Dict[str, InstanceSession] = {
            "default": InstanceSession(
                "default",
                "Default · Sense 感性流",
                InstanceConfig(mode="mock", plan="sense", scenario_id="hajime_regular", current_week=1, scenario_preset="hajime_regular"),
                self.db,
            ),
            "logic": InstanceSession(
                "logic",
                "Logic · 理性好印象",
                InstanceConfig(mode="mock", plan="logic", scenario_id="hajime_regular", current_week=6, scenario_preset="hajime_regular"),
                self.db,
            ),
            "anomaly": InstanceSession(
                "anomaly",
                "Anomaly · 异常爆发",
                InstanceConfig(mode="mock", plan="anomaly", scenario_id="hajime_regular", current_week=13, scenario_preset="hajime_regular"),
                self.db,
            ),
        }
        self.active_id: str = "default"
        self._next_custom_idx: int = 1
        self.hif = HifWorkbenchSession(self.db, broadcast_fn=self.broadcast)

        if auto_preview:
            for inst in self.instances.values():
                try:
                    inst.capture_preview()
                except Exception as exc:
                    logger.warning("Initial preview capture failed for %s: %s", inst.id, exc)

    def get_active(self) -> InstanceSession:
        with self.lock:
            return self.instances[self.active_id]

    def get_instance(self, instance_id: Optional[str] = None) -> InstanceSession:
        with self.lock:
            target = instance_id or self.active_id
            if target not in self.instances:
                raise KeyError(f"Unknown instance: {target}")
            return self.instances[target]

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=32)
        with self.lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, event_name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        data = payload if payload is not None else {"active_id": self.active_id, "ts": time.time()}
        with self.lock:
            dead: List[queue.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait((event_name, data))
                except queue.Full:
                    dead.append(q)
            for d in dead:
                if d in self._subscribers:
                    self._subscribers.remove(d)

    def get_full_state(self) -> Dict[str, Any]:
        with self.lock:
            active = self.instances[self.active_id]
            tabs = [
                {
                    "id": inst.id,
                    "name": inst.name,
                    "plan": inst.config.plan,
                    "mode": inst.config.mode,
                    "running": inst.running,
                    "step_count": inst.step_count,
                    "scene_type": inst.observation.scene_type if inst.observation else "idle",
                }
                for inst in self.instances.values()
            ]
            monitors = []
            for inst in self.instances.values():
                obs = inst.observation
                dec = inst.decision
                monitors.append(
                    {
                        "id": inst.id,
                        "name": inst.name,
                        "plan": inst.config.plan,
                        "mode": inst.config.mode,
                        "running": inst.running,
                        "step_count": inst.step_count,
                        "frame_seq": inst.frame_seq,
                        "frame_meta": {"width": inst.frame_width, "height": inst.frame_height},
                        "scene_type": obs.scene_type if obs else "idle",
                        "turn": obs.turn if obs else 0,
                        "stamina": obs.stamina if obs else 0,
                        "max_stamina": obs.max_stamina if obs else 0,
                        "score": obs.score if obs else 0,
                        "last_choice": dec.choice_id if dec else None,
                        "last_source": dec.source if dec else None,
                        "last_reason": dec.reason if dec else "等待执行...",
                        "latency_ms": round(dec.latency_ms, 1) if dec else 0.0,
                    }
                )
        return {
            "active_id": self.active_id,
            "instances": tabs,
            "monitors": monitors,
            "active": active.to_summary_dict(),
        }

    def select_instance(self, instance_id: str) -> Dict[str, Any]:
        with self.lock:
            if instance_id not in self.instances:
                raise KeyError(f"Unknown instance: {instance_id}")
            self.active_id = instance_id
        self.broadcast("state")
        return {"ok": True, "active_id": self.active_id}

    def create_instance(self, name: Optional[str] = None, plan: str = "sense") -> Dict[str, Any]:
        with self.lock:
            idx = self._next_custom_idx
            self._next_custom_idx += 1
            inst_id = f"custom-{idx}"
            label = (name or f"配置 #{idx} ({plan})").strip()
            inst = InstanceSession(
                inst_id,
                label,
                InstanceConfig(mode="mock", plan=plan if plan in ("sense", "logic", "anomaly") else "sense"),
                self.db,
            )
            inst.capture_preview()
            self.instances[inst_id] = inst
            self.active_id = inst_id
        self.broadcast("state")
        return {"ok": True, "id": inst_id}

    def update_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inst = self.get_instance(payload.get("instance_id"))
        valid_enums = {
            "mode": ("mock", "adb", "win32"),
            "plan": ("sense", "logic", "anomaly"),
            "scenario_id": ("hajime_regular", "hajime_pro", "hajime_master", "nia_master", "hif_honsen"),
            "scenario_preset": ("hajime_regular", "sequence", "battle_turn", "produce_schedule", "card_reward", "shop", "popup"),
            "llm_mode": ("auto_xai", "custom", "offline"),
        }
        for ek, allowed in valid_enums.items():
            if ek in payload and payload[ek] is not None:
                val = str(payload[ek]).strip()
                if val not in allowed:
                    raise ValueError(f"Invalid {ek}: {val!r} (expected one of {allowed})")

        port_val: Optional[int] = None
        if "adb_port" in payload and payload["adb_port"] is not None:
            port_val = max(1, min(65535, int(payload["adb_port"])))
        steps_val: Optional[int] = None
        if "steps" in payload and payload["steps"] is not None:
            steps_val = max(1, min(50, int(payload["steps"])))
        delay_val: Optional[float] = None
        if "delay" in payload and payload["delay"] is not None:
            delay_val = max(0.0, min(10.0, float(payload["delay"])))
        week_val: Optional[int] = None
        if "current_week" in payload and payload["current_week"] is not None:
            week_val = max(1, min(25, int(payload["current_week"])))

        rebuild_needed = False
        with inst.lock:
            cfg = inst.config
            for key in ("mode", "plan", "scenario_id", "scenario_preset", "adb_host", "adb_root_pwd", "win32_title", "llm_mode", "llm_base_url", "llm_model"):
                if key in payload and payload[key] is not None:
                    val = str(payload[key]).strip() if key != "win32_title" else str(payload[key])
                    if getattr(cfg, key) != val:
                        setattr(cfg, key, val)
                        if key in ("mode", "plan", "scenario_id", "scenario_preset", "adb_host", "adb_root_pwd", "win32_title"):
                            rebuild_needed = True
            if week_val is not None:
                spec = get_scenario_spec(cfg.scenario_id)
                clamped_w = max(1, min(spec.total_weeks, week_val))
                if cfg.current_week != clamped_w:
                    cfg.current_week = clamped_w
                    rebuild_needed = True
            if "llm_api_key" in payload and payload["llm_api_key"] is not None:
                key_val = str(payload["llm_api_key"]).strip()
                if key_val:
                    cfg.llm_api_key = key_val
            if port_val is not None and cfg.adb_port != port_val:
                cfg.adb_port = port_val
                rebuild_needed = True
            if steps_val is not None:
                cfg.steps = steps_val
            if delay_val is not None:
                cfg.delay = delay_val
            if "auto_progress" in payload and payload["auto_progress"] is not None:
                new_ap = bool(payload["auto_progress"])
                if cfg.auto_progress != new_ap:
                    cfg.auto_progress = new_ap
                    rebuild_needed = True
            if "fallback_to_mock" in payload and payload["fallback_to_mock"] is not None:
                new_fb = bool(payload["fallback_to_mock"])
                if cfg.fallback_to_mock != new_fb:
                    cfg.fallback_to_mock = new_fb
                    rebuild_needed = True
            if isinstance(payload.get("tasks_enabled"), dict):
                for tk, tv in payload["tasks_enabled"].items():
                    if tk in cfg.tasks_enabled:
                        cfg.tasks_enabled[tk] = bool(tv)

            if "p_idol_id" in payload:
                raw_pid = payload.get("p_idol_id")
                new_pid = int(raw_pid) if raw_pid not in (None, "", 0) else None
                if cfg.p_idol_id != new_pid:
                    cfg.p_idol_id = new_pid
                    rebuild_needed = True

            if rebuild_needed:
                resolved_pid = cfg.p_idol_id if cfg.p_idol_id is not None else (
                    126 if (cfg.plan or "").lower() == "logic" else None
                )
                if inst.tracker.plan != cfg.plan or inst.tracker.p_idol_id != resolved_pid:
                    inst.tracker = GameStateTracker(plan=cfg.plan, p_idol_id=resolved_pid)
                inst.reset_controller()
                inst.add_log("info", f"配置已更新 -> mode={cfg.mode}, plan={cfg.plan}, preset={cfg.scenario_preset}")
                if cfg.mode == "mock":
                    inst.capture_preview()

        self.broadcast("state")
        return {"ok": True}

    def inject_scenario(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inst = self.get_instance(payload.get("instance_id"))
        preset = str(payload.get("scenario_preset") or "hajime_regular")
        if preset not in ("hajime_regular", "sequence", "battle_turn", "produce_schedule", "card_reward", "shop", "popup"):
            raise ValueError(f"Unsupported scenario_preset: {preset}")

        with inst.lock:
            if payload.get("plan") in ("sense", "logic", "anomaly"):
                inst.config.plan = payload["plan"]
                inst.tracker = GameStateTracker(plan=inst.config.plan)
            if payload.get("scenario_id") in SCENARIO_REGISTRY:
                inst.config.scenario_id = str(payload["scenario_id"])
            if payload.get("week") is not None:
                spec = get_scenario_spec(inst.config.scenario_id)
                inst.config.current_week = max(1, min(spec.total_weeks, int(payload["week"])))
            inst.config.mode = "mock"
            inst.config.scenario_preset = preset
            overrides = payload.get("battle_overrides")
            if isinstance(overrides, dict):
                cleaned: Dict[str, Any] = {}
                for num_key in ("stamina", "max_stamina", "genki", "turn", "turns_remaining", "score", "target_score"):
                    if num_key in overrides and overrides[num_key] is not None:
                        cleaned[num_key] = int(overrides[num_key])
                if isinstance(overrides.get("buffs"), dict):
                    cleaned["buffs"] = {str(k): int(v) for k, v in overrides["buffs"].items()}
                inst._custom_battle_overrides = cleaned
            else:
                inst._custom_battle_overrides = None
            inst.reset_controller()
            res = inst.capture_preview()
        self.broadcast("state")
        return res

    def run_step(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        inst = self.get_instance(instance_id)
        with inst.lock:
            if inst.running:
                return {"ok": False, "error": "自动任务队列正在运行中，请先停止再单步执行"}
        try:
            result = inst.execute_step()
            self.broadcast("state")
            dec = result.decision
            return {
                "ok": bool(result.success),
                "step": inst.step_count,
                "choice_id": dec.choice_id if dec else None,
                "source": dec.source if dec else None,
                "reason": dec.reason if dec else (result.error or ""),
                "error": result.error,
            }
        except Exception as exc:
            with inst.lock:
                inst.last_error = str(exc)
                inst.add_log("error", f"执行单步失败: {exc}")
            self.broadcast("state")
            return {"ok": False, "error": str(exc)}

    def start_loop(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        inst = self.get_instance(instance_id)
        prev_thread = inst.worker_thread
        if prev_thread is not None and prev_thread.is_alive() and not inst.running:
            prev_thread.join(timeout=2.0)
        with inst.lock:
            if inst.running:
                return {"ok": True, "already_running": True}
            inst.running = True
            inst.stop_event.clear()
            inst.last_error = ""
            max_steps = inst.config.steps
            delay = inst.config.delay
            inst.add_log("info", f"开始自动任务队列 (plan={inst.config.plan}, mode={inst.config.mode}, steps={max_steps})")

        def _worker() -> None:
            try:
                for i in range(max_steps):
                    if inst.stop_event.is_set():
                        inst.add_log("info", "任务队列已手动停止")
                        break
                    try:
                        inst.execute_step()
                    except Exception as exc:
                        with inst.lock:
                            inst.last_error = str(exc)
                            inst.add_log("error", f"任务循环异常中止: {exc}")
                        break
                    self.broadcast("state")
                    if i < max_steps - 1 and delay > 0:
                        if inst.stop_event.wait(delay):
                            inst.add_log("info", "任务队列已手动停止")
                            break
            finally:
                with inst.lock:
                    inst.running = False
                self.broadcast("state")

        thread = threading.Thread(target=_worker, name=f"gakumas-loop-{inst.id}", daemon=True)
        inst.worker_thread = thread
        thread.start()
        self.broadcast("state")
        return {"ok": True}

    def stop_loop(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        inst = self.get_instance(instance_id)
        with inst.lock:
            inst.stop_event.set()
            inst.running = False
        self.broadcast("state")
        return {"ok": True}

    def stop_all(self) -> None:
        self.hif.stop()
        with self.lock:
            for inst in self.instances.values():
                inst.stop_event.set()
                inst.running = False
                inst.reset_controller()

    def test_adb_connection(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        host = str(payload.get("adb_host") or "192.168.0.115").strip()
        port = int(payload.get("adb_port") or 5555)
        root_pwd = str(payload.get("adb_root_pwd") or "root/114514").strip()
        t0 = time.perf_counter()
        ctrl = AdbController(
            config=AdbConfig(
                host=host,
                port=port,
                root_pwd=root_pwd,
                connect_timeout=2.5,
                command_timeout=3.5,
                retry_count=1,
                fallback_to_mock=False,
            )
        )
        try:
            connected = ctrl.connect()
            if not connected:
                return {
                    "ok": False,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "message": f"无法连接到 ADB 设备 {host}:{port}",
                }
            frame = ctrl.screencap()
            h, w = frame.shape[:2]
            return {
                "ok": True,
                "width": w,
                "height": h,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "message": f"ADB 连接成功 ({host}:{port})，截图分辨率 {w}x{h}",
            }
        except Exception as exc:
            return {
                "ok": False,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "message": f"ADB 连接探测失败: {exc}",
            }
        finally:
            try:
                ctrl.disconnect()
            except Exception:
                pass

    def test_llm_connection(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        inst = self.get_instance(instance_id)
        cfg = inst.resolve_agent_config()
        if not cfg.api_key:
            return {
                "ok": False,
                "mode": cfg.base_url,
                "model": cfg.model,
                "message": "当前处于纯本地规则模式（未检测到 API Key）。开启自动决策不会阻断流程。",
            }
        t0 = time.perf_counter()
        maker = AgentDecisionMaker(config=cfg)
        probe_actions = [
            LegalAction(
                choice_id=0,
                action_type="pick_option",
                description="Vocal Lesson +30",
                bbox=BoundingBox(xmin=100, ymin=400, xmax=500, ymax=600),
            ),
            LegalAction(
                choice_id=1,
                action_type="pick_option",
                description="Rest +20 Stamina",
                bbox=BoundingBox(xmin=580, ymin=400, xmax=980, ymax=600),
            ),
        ]
        probe_obs = ScreenObservation(scene_type="produce_schedule", stamina=18, max_stamina=30, plan=inst.config.plan)
        dec = maker.decide("Scene: produce_schedule | Stamina: 18/30\nLegal Actions:\n[0] Vocal Lesson\n[1] Rest", probe_actions, obs=probe_obs)
        return {
            "ok": dec.source == "llm",
            "source": dec.source,
            "model": cfg.model,
            "base_url": cfg.base_url,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "message": f"决策测试完成: source={dec.source}, choice={dec.choice_id}, reason={dec.reason}",
        }

    def query_library_cards(
        self,
        plan: str = "all",
        category: str = "all",
        query: str = "",
        limit: int = 80,
    ) -> Dict[str, Any]:
        plans = ("sense", "logic", "anomaly") if plan == "all" else (plan,)
        seen_ids = set()
        results: List[Dict[str, Any]] = []
        q_lower = query.strip().lower()
        for p in plans:
            for row in self.db.get_cards_for_plan(p):
                cid = row["id"]
                if cid in seen_ids:
                    continue
                if category != "all" and row.get("category") != category:
                    continue
                name = row.get("name") or ""
                desc = row.get("description") or ""
                if q_lower and q_lower not in name.lower() and q_lower not in desc.lower() and q_lower not in str(cid):
                    continue
                seen_ids.add(cid)
                item = dict(row)
                raw_avail = row.get("available_customization_ids_json")
                try:
                    item["available_customization_ids"] = [int(x) for x in json.loads(raw_avail or "[]")]
                except Exception:
                    item["available_customization_ids"] = []
                item["icon_url"] = _to_asset_url(row.get("icon_path"))
                item["details_url"] = _to_asset_url(row.get("details_path"))
                results.append(item)

        results.sort(key=lambda x: (x.get("id", 0), x.get("upgraded", 0)))
        total = len(results)
        return {"total": total, "items": results[:limit]}

    def query_library_drinks(self, plan: str = "all", query: str = "", limit: int = 80) -> Dict[str, Any]:
        plans = ("sense", "logic", "anomaly") if plan == "all" else (plan,)
        seen_ids = set()
        results: List[Dict[str, Any]] = []
        q_lower = query.strip().lower()
        for p in plans:
            for row in self.db.get_drinks_for_plan(p):
                did = row["id"]
                if did in seen_ids:
                    continue
                name = row.get("name") or ""
                desc = row.get("description") or ""
                if q_lower and q_lower not in name.lower() and q_lower not in desc.lower() and q_lower not in str(did):
                    continue
                seen_ids.add(did)
                item = dict(row)
                raw_avail = row.get("available_customization_ids_json")
                try:
                    item["available_customization_ids"] = [int(x) for x in json.loads(raw_avail or "[]")]
                except Exception:
                    item["available_customization_ids"] = []
                item["icon_url"] = _to_asset_url(row.get("icon_path"))
                item["details_url"] = _to_asset_url(row.get("details_path"))
                results.append(item)
        results.sort(key=lambda x: x.get("id", 0))
        return {"total": len(results), "items": results[:limit]}


class GakumasWebHandler(BaseHTTPRequestHandler):
    """HTTP Request handler serving MFAAvalonia UI static assets, read-only game art, and REST/SSE APIs."""

    server_version = "GakumasWebUI/1.0"

    def __init__(self, *args: Any, session_manager: WebSessionManager, **kwargs: Any):
        self.manager = session_manager
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("HTTP %s - %s", self.address_string(), format % args)

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, data: bytes, content_type: str, cache_control: str = "no-store") -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _serve_safe_file(
        self,
        rel_path: str,
        base_dir: Path,
        allowed_roots: Tuple[Path, ...],
        cache_control: str,
    ) -> None:
        clean_parts = [p for p in unquote(rel_path).split("/") if p]
        if not clean_parts or any(p in (".", "..") for p in clean_parts):
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid path")
            return
        candidate = (base_dir / Path(*clean_parts)).resolve()
        if not any(candidate.is_relative_to(root.resolve()) for root in allowed_roots):
            self.send_error(HTTPStatus.FORBIDDEN, "Access denied")
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        ctype, _ = mimetypes.guess_type(str(candidate))
        self._send_bytes(candidate.read_bytes(), ctype or "application/octet-stream", cache_control=cache_control)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            index_file = STATIC_DIR / "index.html"
            if not index_file.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "index.html missing")
                return
            self._send_bytes(index_file.read_bytes(), "text/html; charset=utf-8", cache_control="no-store")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            static_root = STATIC_DIR.resolve()
            self._serve_safe_file(rel, static_root, (static_root,), cache_control="no-store")
            return

        if path.startswith("/assets/"):
            rel = path[len("/assets/"):]
            self._serve_safe_file(rel, PROJECT_ROOT.resolve(), ALLOWED_ASSET_ROOTS, cache_control="public, max-age=3600")
            return

        if path == "/api/state":
            self._send_json(self.manager.get_full_state())
            return

        if path == "/api/scenarios":
            catalog = {}
            for sid, spec in SCENARIO_REGISTRY.items():
                catalog[sid] = {
                    "scenario_id": spec.scenario_id,
                    "name": spec.name,
                    "short_name": spec.short_name,
                    "difficulty": spec.difficulty,
                    "total_weeks": spec.total_weeks,
                    "stat_cap": spec.stat_cap,
                    "midterm_week": spec.midterm_week,
                    "final_week": spec.final_week,
                    "description": spec.description,
                    "weeks": [
                        {
                            "week": wn.week,
                            "phase_key": wn.phase_key,
                            "title": wn.title,
                            "scene_type": wn.scene_type,
                            "activities": list(wn.activities),
                            "recommended_action": wn.recommended_action,
                            "operation_guide": wn.operation_guide,
                            "target_score": wn.target_score,
                        }
                        for wn in spec.weeks.values()
                    ],
                }
            self._send_json(catalog)
            return

        if path == "/api/frame":
            inst_id = qs.get("instance_id", [None])[0]
            try:
                inst = self.manager.get_instance(inst_id)
            except KeyError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            with inst.lock:
                png = inst.frame_png_bytes
            if not png:
                inst.capture_preview()
                with inst.lock:
                    png = inst.frame_png_bytes
            self._send_bytes(png, "image/png", cache_control="no-store")
            return

        if path == "/api/events":
            self._handle_sse()
            return

        if path == "/api/library/cards":
            plan = qs.get("plan", ["all"])[0]
            category = qs.get("category", ["all"])[0]
            q = qs.get("q", [""])[0]
            try:
                limit = max(1, min(1500, int(qs.get("limit", ["120"])[0])))
            except ValueError:
                self._send_json({"ok": False, "error": "Invalid limit parameter"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.manager.query_library_cards(plan=plan, category=category, query=q, limit=limit))
            return

        if path == "/api/library/drinks":
            plan = qs.get("plan", ["all"])[0]
            q = qs.get("q", [""])[0]
            try:
                limit = max(1, min(500, int(qs.get("limit", ["120"])[0])))
            except ValueError:
                self._send_json({"ok": False, "error": "Invalid limit parameter"}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(self.manager.query_library_drinks(plan=plan, query=q, limit=limit))
            return

        if path == "/api/hif/catalog":
            self._send_json(self.manager.hif.build_catalog())
            return

        if path == "/api/hif/state":
            self._send_json(self.manager.hif.get_state())
            return

        if path == "/api/hif/frame":
            self._send_bytes(self.manager.hif.get_frame_png(), "image/png", cache_control="no-store")
            return

        if path == "/api/hif/export-library":
            self._send_json(self.manager.hif.build_library_payload())
            return

        if path.startswith("/api/"):
            self._send_json({"ok": False, "error": f"Endpoint not found: {path}"}, status=HTTPStatus.NOT_FOUND)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json_body()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if path == "/api/instance/select":
                self._send_json(self.manager.select_instance(str(body.get("instance_id") or "")))
                return
            if path == "/api/instance/create":
                self._send_json(self.manager.create_instance(name=body.get("name"), plan=str(body.get("plan") or "sense")))
                return
            if path == "/api/config":
                self._send_json(self.manager.update_config(body))
                return
            if path == "/api/scenario":
                self._send_json(self.manager.inject_scenario(body))
                return
            if path == "/api/preview":
                if body.get("all"):
                    refreshed = []
                    for inst_obj in list(self.manager.instances.values()):
                        if inst_obj.config.mode == "mock" or inst_obj.id == self.manager.active_id:
                            inst_obj.capture_preview()
                            refreshed.append(inst_obj.id)
                    self.manager.broadcast("state")
                    self._send_json({"ok": True, "refreshed": refreshed})
                    return
                inst = self.manager.get_instance(body.get("instance_id"))
                res = inst.capture_preview()
                self.manager.broadcast("state")
                self._send_json(res)
                return
            if path == "/api/step":
                self._send_json(self.manager.run_step(body.get("instance_id")))
                return
            if path == "/api/start":
                self._send_json(self.manager.start_loop(body.get("instance_id")))
                return
            if path == "/api/stop":
                self._send_json(self.manager.stop_loop(body.get("instance_id")))
                return
            if path == "/api/reset":
                inst = self.manager.get_instance(body.get("instance_id"))
                with inst.lock:
                    inst.stop_event.set()
                    inst.running = False
                    inst.tracker.reset()
                    inst.step_count = 0
                    inst.snapshots.clear()
                    inst.decision = None
                    inst.click_coords = None
                    inst.reset_controller(clear_checkpoint=True)
                    inst.add_log("info", "状态追踪器与控制器已重设")
                    if inst.config.mode == "mock":
                        inst.capture_preview()
                self.manager.broadcast("state")
                self._send_json({"ok": True})
                return
            if path == "/api/logs/clear":
                inst = self.manager.get_instance(body.get("instance_id"))
                with inst.lock:
                    inst.logs.clear()
                self.manager.broadcast("state")
                self._send_json({"ok": True})
                return
            if path == "/api/test-adb":
                self._send_json(self.manager.test_adb_connection(body))
                return
            if path == "/api/test-llm":
                self._send_json(self.manager.test_llm_connection(body.get("instance_id")))
                return
            if path == "/api/hif/select-idol":
                self._send_json(self.manager.hif.select_idol(body))
                return
            if path == "/api/hif/config":
                self._send_json(self.manager.hif.update_config(body))
                return
            if path == "/api/hif/loadout":
                self._send_json(self.manager.hif.update_loadout(body))
                return
            if path == "/api/hif/memories":
                self._send_json(self.manager.hif.update_memories(body))
                return
            if path == "/api/hif/preferences":
                self._send_json(self.manager.hif.update_preferences(body))
                return
            if path == "/api/hif/import-library":
                self._send_json(self.manager.hif.import_library(body))
                return
            if path == "/api/hif/preview":
                self._send_json(self.manager.hif.capture_preview())
                return
            if path == "/api/hif/start":
                self._send_json(self.manager.hif.start())
                return
            if path == "/api/hif/stop":
                self._send_json(self.manager.hif.stop())
                return
            if path == "/api/hif/logs/clear":
                self._send_json(self.manager.hif.clear_logs())
                return
        except PermissionError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.FORBIDDEN)
            return
        except KeyError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        except ValueError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            logger.exception("API error on %s", path)
            self.manager.broadcast("state")
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json({"ok": False, "error": f"Endpoint not found: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _handle_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = self.manager.subscribe()
        try:
            init_payload = json.dumps({"active_id": self.manager.active_id, "ts": time.time()})
            self.wfile.write(f"event: connected\ndata: {init_payload}\n\n".encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    event_name, data = q.get(timeout=15.0)
                    msg = f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.manager.unsubscribe(q)


def create_web_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: Optional[str] = None,
    auto_preview: bool = True,
) -> Tuple[ThreadingHTTPServer, WebSessionManager]:
    manager = WebSessionManager(db_path=db_path, auto_preview=auto_preview)
    handler_cls = partial(GakumasWebHandler, session_manager=manager)
    httpd = ThreadingHTTPServer((host, port), handler_cls)
    httpd.daemon_threads = True
    return httpd, manager


def run_web_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    db_path: Optional[str] = None,
    open_browser: bool = False,
) -> None:
    httpd, manager = create_web_server(host=host, port=port, db_path=db_path, auto_preview=True)
    actual_port = httpd.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print(f"[Gakumas Web UI] Running MFAAvalonia-style workbench at {url}")
    print("[Gakumas Web UI] Press Ctrl+C to stop.")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Gakumas MFAAvalonia-style Web UI Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    run_web_server(host=args.host, port=args.port, open_browser=args.open)
