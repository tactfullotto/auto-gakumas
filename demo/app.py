"""
Gakumas Dual-Mode Autonomous Decision & Automation Demo Entrypoint.

Supports:
- --mode mock : Synthesizes frames from local cards/ & drinks/ assets and runs
                the full closed loop (vision -> DB -> prompt -> LLM/fallback -> click -> state transition).
- --mode adb  : Connects to an Android emulator (default 192.168.0.115:5555) via MaaFw / ADB.
- --mode win32: Attaches to a Windows desktop client window via Win32 GDI/PostMessage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Ensure workspace root (parent of demo/) is importable when running `python demo/app.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.adb_controller import AdbConfig, AdbController
from demo.maa_bridge.base_controller import BaseController
from demo.maa_bridge.custom_action import GakumasDecisionAction
from demo.maa_bridge.custom_recognition import (
    GakumasBattleRecognition,
    GakumasPopupRecognition,
    GakumasProduceRecognition,
    GakumasRewardRecognition,
    GakumasSceneRoutingRecognition,
    GakumasShopRecognition,
    GakumasSkipRecognition,
)
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.runner import GakumasRunner, StepResult
from demo.maa_bridge.simulator import MaaPipelineSimulator
from demo.maa_bridge.win32_controller import Win32Config, Win32Controller
from demo.schemas import DecisionResult, LegalAction, ScreenObservation

ALL_PLANS = ("sense", "logic", "anomaly")


def _configure_stdio_utf8() -> None:
    """Safely configure UTF-8 stdout/stderr on Windows terminals."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def estimate_token_count(text: str) -> int:
    """Estimate token count conservatively for mixed CJK/ASCII prompts."""
    if not text:
        return 0
    return max(1, (len(text) + 1) // 2)


def build_demo_scenarios(plan: str = "sense") -> List[Dict[str, Any]]:
    """
    Build a representative multi-scene scenario sequence for offline mock mode:
    produce_schedule -> card_reward -> battle_turn (turn 1) -> battle_turn (turn 2) -> popup.
    """
    plan_lower = plan.lower()
    if plan_lower == "sense":
        buffs_t1 = {"goodConditionTurns": 3, "concentration": 2}
        buffs_t2 = {"goodConditionTurns": 4, "concentration": 5}
    elif plan_lower == "logic":
        buffs_t1 = {"goodImpressionTurns": 4, "motivation": 3}
        buffs_t2 = {"goodImpressionTurns": 6, "motivation": 5}
    else:
        buffs_t1 = {"fullPowerCharge": 4}
        buffs_t2 = {"fullPowerCharge": 8}

    return [
        {"scene_type": "produce_schedule"},
        {"scene_type": "card_reward"},
        {
            "scene_type": "battle_turn",
            "turn": 1,
            "turns_remaining": 8,
            "stamina": 28,
            "max_stamina": 30,
            "genki": 5,
            "score": 120,
            "target_score": 5000,
            "buffs": buffs_t1,
        },
        {
            "scene_type": "battle_turn",
            "turn": 2,
            "turns_remaining": 7,
            "stamina": 24,
            "max_stamina": 30,
            "genki": 9,
            "score": 680,
            "target_score": 5000,
            "buffs": buffs_t2,
        },
        {"scene_type": "popup", "message": "Confirm Stage Progress?"},
    ]


def create_configured_simulator(
    db: GakumasDB,
    plan: str,
    controller: BaseController,
    agent: AgentDecisionMaker,
    tracker: GameStateTracker,
    prompt_builder: PromptBuilder,
    pipeline_path: Optional[Path] = None,
) -> MaaPipelineSimulator:
    """Load pipeline.json into MaaPipelineSimulator and register all custom handlers."""
    resolved_pipeline = pipeline_path or (Path(__file__).resolve().parent / "maa_bridge" / "pipeline.json")
    sim = MaaPipelineSimulator(resolved_pipeline)
    sim.bind_controller(controller)

    sim.register_custom_recognition("GakumasSceneRoutingRecognition", GakumasSceneRoutingRecognition(db=db, plan=plan))
    sim.register_custom_recognition("GakumasBattleRecognition", GakumasBattleRecognition(db=db, plan=plan))
    sim.register_custom_recognition("GakumasProduceRecognition", GakumasProduceRecognition(db=db, plan=plan))
    sim.register_custom_recognition("GakumasRewardRecognition", GakumasRewardRecognition(db=db, plan=plan))
    sim.register_custom_recognition("GakumasShopRecognition", GakumasShopRecognition(db=db, plan=plan))
    sim.register_custom_recognition("GakumasPopupRecognition", GakumasPopupRecognition(db=db, plan=plan))
    sim.register_custom_recognition("GakumasSkipRecognition", GakumasSkipRecognition(db=db, plan=plan))

    decision_action = GakumasDecisionAction(
        controller=controller,
        agent=agent,
        prompt_builder=prompt_builder,
        tracker=tracker,
        db=db,
    )
    sim.register_custom_action("GakumasDecisionAction", decision_action)
    return sim


class GakumasTUIDashboard:
    """Rich terminal visualizer displaying vision, prompt, and decision panels per step."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def render_banner(self, mode: str, plans: Sequence[str], endpoint_info: str, pipeline_nodes: int) -> None:
        header = Text()
        header.append("GAKUMAS DUAL-MODE DECISION & AUTOMATION DEMO\n", style="bold cyan")
        header.append(f"Mode: {mode.upper()}  |  Plans: {', '.join(plans)}  |  Target: {endpoint_info}  |  Pipeline Nodes: {pipeline_nodes}", style="white")
        self.console.print(Panel(header, border_style="bright_blue", box=box.ROUNDED))

    def _build_observation_panel(
        self,
        obs: Optional[ScreenObservation],
        mode: str,
        plan: str,
        step_idx: int,
    ) -> Panel:
        if obs is None:
            return Panel("No observation captured.", title=f"[1] Vision & Scene (Step {step_idx})", border_style="red")

        buff_str = ", ".join(f"{k}:{v}" for k, v in obs.buffs.items() if v) or "无"
        meta = (
            f"[bold]Mode:[/bold] {mode}  |  [bold]Plan:[/bold] {plan}  |  [bold]Scene:[/bold] {obs.scene_type}\n"
            f"[bold]Turn:[/bold] {obs.turn} (Left {obs.turns_remaining})  |  "
            f"[bold]Stamina:[/bold] {obs.stamina}/{obs.max_stamina}  |  "
            f"[bold]Genki:[/bold] {obs.genki}  |  [bold]Score:[/bold] {obs.score}/{obs.target_score}\n"
            f"[bold]Buffs:[/bold] {buff_str}"
        )

        renderables: List[Any] = [Text.from_markup(meta)]

        if obs.cards:
            card_table = Table(title="Recognized Hand Cards", box=box.SIMPLE, show_lines=False)
            card_table.add_column("ID", style="cyan", no_wrap=True)
            card_table.add_column("Name", style="white")
            card_table.add_column("+", justify="center")
            card_table.add_column("Playable", justify="center")
            card_table.add_column("HP Cost", justify="right")
            card_table.add_column("BBox", style="dim")
            for c in obs.cards:
                b = c.bbox
                card_table.add_row(
                    str(c.card_id),
                    c.name,
                    "+" if c.upgraded else "-",
                    "YES" if c.playable else f"NO ({c.reason})",
                    str(c.hp_loss),
                    f"({b.xmin},{b.ymin},{b.xmax},{b.ymax})",
                )
            renderables.append(card_table)

        if obs.drinks:
            drink_table = Table(title="Recognized P-Drinks", box=box.SIMPLE, show_lines=False)
            drink_table.add_column("ID", style="green", no_wrap=True)
            drink_table.add_column("Name", style="white")
            drink_table.add_column("Playable", justify="center")
            drink_table.add_column("BBox", style="dim")
            for d in obs.drinks:
                b = d.bbox
                drink_table.add_row(
                    str(d.drink_id),
                    d.name,
                    "YES" if d.playable else "NO",
                    f"({b.xmin},{b.ymin},{b.xmax},{b.ymax})",
                )
            renderables.append(drink_table)

        if obs.options:
            opt_table = Table(title="Recognized Scene Options", box=box.SIMPLE, show_lines=False)
            opt_table.add_column("Opt#", style="yellow", no_wrap=True)
            opt_table.add_column("Title / Name", style="white")
            opt_table.add_column("BBox", style="dim")
            for idx, opt in enumerate(obs.options):
                title = opt.get("title") or opt.get("name") or f"Option {idx}"
                raw_bbox = opt.get("bbox")
                if hasattr(raw_bbox, "xmin"):
                    bbox_str = f"({raw_bbox.xmin},{raw_bbox.ymin},{raw_bbox.xmax},{raw_bbox.ymax})"
                elif isinstance(raw_bbox, dict):
                    bbox_str = f"({raw_bbox.get('xmin')},{raw_bbox.get('ymin')},{raw_bbox.get('xmax')},{raw_bbox.get('ymax')})"
                else:
                    bbox_str = str(raw_bbox)
                opt_table.add_row(str(opt.get("choice_id", idx)), str(title), bbox_str)
            renderables.append(opt_table)

        return Panel(
            Group(*renderables),
            title=f"[1] Scene & Vision Observation (Step {step_idx})",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def _build_prompt_panel(self, prompt_text: str, legal_actions: Sequence[LegalAction]) -> Panel:
        est_tokens = estimate_token_count(prompt_text)
        actions_lines = [
            f"[{act.choice_id}] {act.action_type}: {act.description} @ ({act.bbox.xmin},{act.bbox.ymin},{act.bbox.xmax},{act.bbox.ymax})"
            for act in legal_actions
        ]
        actions_block = "\n".join(actions_lines) if actions_lines else "(none)"
        body = Group(
            Text(f"Estimated Prompt Tokens: ~{est_tokens} (< 800 limit)\n", style="bold"),
            Text("Prompt Payload", style="bold underline"),
            Text(prompt_text + "\n"),
            Text(f"Legal Actions ({len(legal_actions)})", style="bold underline"),
            Text(actions_block),
        )
        return Panel(
            body,
            title="[2] Compact Text Prompt & Legal Actions",
            border_style="magenta",
            box=box.ROUNDED,
        )

    def _build_decision_panel(self, step_result: StepResult) -> Panel:
        dec: Optional[DecisionResult] = step_result.decision
        act: Optional[LegalAction] = step_result.action_taken
        click_xy = step_result.click_coords
        if dec is None:
            content = f"Status: {'OK' if step_result.success else 'ERROR'} | Error: {step_result.error}"
        else:
            action_desc = act.description if act else "N/A"
            content = (
                f"Choice ID      : {dec.choice_id} -> {action_desc}\n"
                f"Decision Source: {dec.source}\n"
                f"Reason         : {dec.reason}\n"
                f"Click Target   : {click_xy}\n"
                f"Decision Delay : {dec.latency_ms:.2f} ms (Total Step: {step_result.elapsed_ms:.2f} ms)\n"
                f"Tokens (In/Out): {dec.prompt_tokens} / {dec.completion_tokens}"
            )
        return Panel(
            Text(content),
            title=f"[3] Decision & Simulated Click (Step {step_result.step_idx})",
            border_style="green" if step_result.success else "red",
            box=box.ROUNDED,
        )

    def render_step_panel(
        self,
        step_result: StepResult,
        prompt_text: str,
        mode: str = "mock",
        plan: str = "sense",
    ) -> Panel:
        """Render a composite Rich Panel containing all 3 live decision-loop panes."""
        p1 = self._build_observation_panel(step_result.observation, mode=mode, plan=plan, step_idx=step_result.step_idx)
        p2 = self._build_prompt_panel(prompt_text, step_result.legal_actions)
        p3 = self._build_decision_panel(step_result)
        return Panel(
            Group(p1, p2, p3),
            title=f"Step {step_result.step_idx} | {plan.upper()} | {step_result.scene_type}",
            border_style="bright_white",
            box=box.DOUBLE,
        )

    def print_step(
        self,
        step_result: StepResult,
        prompt_text: str,
        mode: str = "mock",
        plan: str = "sense",
    ) -> None:
        self.console.print(self.render_step_panel(step_result, prompt_text, mode=mode, plan=plan))

    def print_summary(self, records: Sequence[Tuple[str, StepResult]]) -> None:
        table = Table(title="Execution Summary", box=box.ROUNDED)
        table.add_column("Plan", style="cyan")
        table.add_column("Step", justify="right")
        table.add_column("Scene", style="yellow")
        table.add_column("Choice", justify="right")
        table.add_column("Source", style="green")
        table.add_column("Click (x,y)")
        table.add_column("Time (ms)", justify="right")
        table.add_column("Status")

        for plan, res in records:
            choice_str = str(res.decision.choice_id) if res.decision else "-"
            source_str = res.decision.source if res.decision else "-"
            click_str = str(res.click_coords) if res.click_coords else "-"
            status_str = "OK" if res.success else f"ERR: {res.error}"
            table.add_row(
                plan,
                str(res.step_idx),
                res.scene_type,
                choice_str,
                source_str,
                click_str,
                f"{res.elapsed_ms:.1f}",
                status_str,
            )
        self.console.print(table)


def _create_controller(
    mode: str,
    db: GakumasDB,
    plan: str,
    host: str,
    port: int,
    adb_path: Optional[str],
    root_pwd: str,
    window_title: str,
    fallback_to_mock: bool,
) -> Tuple[BaseController, str]:
    mode_lower = mode.lower()
    if mode_lower == "mock":
        ctrl = MockController(db=db, plan=plan, auto_progress=True)
        ctrl.queue_scenarios(build_demo_scenarios(plan))
        return ctrl, f"mock://{plan}"
    if mode_lower == "adb":
        cfg = AdbConfig(
            host=host,
            port=port,
            adb_path=adb_path,
            root_pwd=root_pwd,
            fallback_to_mock=fallback_to_mock,
        )
        return AdbController(config=cfg), f"adb://{host}:{port}"
    if mode_lower == "win32":
        cfg = Win32Config(window_title=window_title, headless_fallback=fallback_to_mock)
        return Win32Controller(config=cfg), f"win32://{window_title}"
    raise ValueError(f"Unsupported mode: {mode}")


def run_demo(
    mode: str = "mock",
    plan: str = "sense",
    steps: int = 5,
    host: str = "192.168.0.115",
    port: int = 5555,
    adb_path: Optional[str] = None,
    root_pwd: str = "root/114514",
    window_title: str = "学园偶像大师",
    delay: float = 0.05,
    fallback_to_mock: bool = True,
    db_path: Optional[str] = None,
    console: Optional[Console] = None,
    checkpoint_path: Optional[str] = None,
    fresh: bool = False,
    p_idol_id: Optional[int] = None,
) -> List[Tuple[str, StepResult]]:
    """Run the dual-mode demo loop and return all executed (plan, StepResult) pairs."""
    resolved_db_path = db_path or str(_PROJECT_ROOT / "gakumas_library.db")
    db = GakumasDB(db_path=resolved_db_path)
    dashboard = GakumasTUIDashboard(console=console)
    agent = AgentDecisionMaker(AgentConfig.from_env_or_xai())

    plans_to_run = list(ALL_PLANS) if plan.lower() == "all" else [plan.lower()]
    all_records: List[Tuple[str, StepResult]] = []

    try:
        for idx, active_plan in enumerate(plans_to_run):
            agent.reset_session()
            resolved_pid = p_idol_id if p_idol_id is not None else (126 if active_plan == "logic" else None)
            tracker = GameStateTracker(plan=active_plan, p_idol_id=resolved_pid)
            prompt_builder = PromptBuilder(db=db)
            controller, endpoint_info = _create_controller(
                mode=mode,
                db=db,
                plan=active_plan,
                host=host,
                port=port,
                adb_path=adb_path,
                root_pwd=root_pwd,
                window_title=window_title,
                fallback_to_mock=fallback_to_mock,
            )

            # Verify MaaPipelineSimulator loads pipeline.json & registers all handlers cleanly
            simulator = create_configured_simulator(
                db=db,
                plan=active_plan,
                controller=controller,
                agent=agent,
                tracker=tracker,
                prompt_builder=prompt_builder,
            )

            if idx == 0:
                dashboard.render_banner(
                    mode=mode,
                    plans=plans_to_run,
                    endpoint_info=endpoint_info,
                    pipeline_nodes=len(simulator.pipeline),
                )

            latest_prompt = {"text": ""}

            def _capture_prompt(prompt_txt: str, _actions: List[LegalAction]) -> None:
                latest_prompt["text"] = prompt_txt

            def _render_step(step_res: StepResult) -> None:
                dashboard.print_step(
                    step_res,
                    prompt_text=latest_prompt["text"],
                    mode=mode,
                    plan=active_plan,
                )

            resolved_ckpt: Optional[Path] = None
            if checkpoint_path:
                resolved_ckpt = Path(checkpoint_path)
            elif mode.lower() != "mock":
                resolved_ckpt = _PROJECT_ROOT / "demo" / "state" / f"produce_{active_plan}.json"

            runner = GakumasRunner(
                mode=mode,
                controller=controller,
                agent=agent,
                db=db,
                tracker=tracker,
                prompt_builder=prompt_builder,
                plan=active_plan,
                p_idol_id=resolved_pid,
                session_mode=agent.is_client_available,
                checkpoint_path=resolved_ckpt,
                auto_resume=not fresh,
                on_prompt_built=_capture_prompt,
                on_step_completed=_render_step,
            )

            if fresh:
                runner.reset_produce_session(clear_checkpoint=True)

            if not runner.start():
                raise RuntimeError(f"Failed to start runner in mode={mode} for target={endpoint_info}")

            try:
                results = runner.run_loop(max_steps=max(1, steps), delay_between_steps=max(0.0, delay))
                for r in results:
                    all_records.append((active_plan, r))
            finally:
                runner.stop()

        dashboard.print_summary(all_records)
    finally:
        db.close()

    return all_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gakumas Dual-Mode Autonomous Card Decision & MaaFramework Demo"
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "adb", "win32", "ui"],
        default="mock",
        help="Execution mode: mock (offline synthetic frames), adb (Android emulator), win32 (desktop client), or ui (MFAAvalonia-style Web UI)",
    )
    parser.add_argument(
        "--plan",
        choices=["sense", "logic", "anomaly", "all"],
        default="sense",
        help="Idol archetype plan to simulate or 'all' to cycle through all 3 plans",
    )
    parser.add_argument("--steps", type=int, default=5, help="Maximum steps per plan")
    parser.add_argument("--host", default="192.168.0.115", help="ADB emulator host IP")
    parser.add_argument("--port", type=int, default=5555, help="ADB emulator port")
    parser.add_argument("--adb-path", default=None, help="Explicit path to adb executable")
    parser.add_argument("--root-pwd", default="root/114514", help="ADB root credentials/tag")
    parser.add_argument("--window-title", default="学园偶像大师", help="Win32 client window title")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay in seconds between steps")
    parser.add_argument("--ui-host", default="127.0.0.1", help="Web UI bind host (when --mode ui)")
    parser.add_argument("--ui-port", type=int, default=8765, help="Web UI bind port (when --mode ui)")
    parser.add_argument("--open-browser", action="store_true", help="Open default browser when launching Web UI")
    parser.add_argument(
        "--fallback-to-mock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fall back to mock frames if ADB/Win32 target is unreachable (default: enabled)",
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint JSON path for interrupt/resume")
    parser.add_argument("--fresh", action="store_true", help="Ignore & clear any saved produce checkpoint")
    parser.add_argument("--p-idol", type=int, default=None, help="Optional P-Idol ID (e.g. 126 for [赤裸々] 十王星南)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_stdio_utf8()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.mode == "ui":
        from demo.web.server import run_web_server

        run_web_server(host=args.ui_host, port=args.ui_port, open_browser=args.open_browser)
        return 0
    records = run_demo(
        mode=args.mode,
        plan=args.plan,
        steps=args.steps,
        host=args.host,
        port=args.port,
        adb_path=args.adb_path,
        root_pwd=args.root_pwd,
        window_title=args.window_title,
        delay=args.delay,
        fallback_to_mock=args.fallback_to_mock,
        checkpoint_path=args.checkpoint,
        fresh=args.fresh,
        p_idol_id=args.p_idol,
    )
    return 0 if records and all(r.success for _, r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
