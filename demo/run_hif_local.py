"""Local zero-LLM H.I.F 本戦 end-to-end automation CLI (ADB + Maskable PPO + SO-ISMCTS)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional

import cv2

from demo.agent.local_agent import LocalHybridDecisionAgent
from demo.core.db import GakumasDB
from demo.core.hif_catalog import ProducePreferences
from demo.maa_bridge.adb_controller import AdbConfig, AdbController
from demo.maa_bridge.runner import GakumasRunner, StepResult
from demo.sim.live_battle_planner import LiveBattlePlannerConfig
from demo.vision.loadout_scanner import HifLoadoutScanner


def _parse_host_port(addr: str) -> tuple[str, int]:
    if ":" in addr:
        host, port_str = addr.rsplit(":", 1)
        return host.strip(), int(port_str.strip())
    return addr.strip(), 5555


def build_local_hif_runner(
    *,
    adb_addr: str = "192.168.0.115:5555",
    adb_path: Optional[str] = None,
    plan: str = "logic",
    scenario_id: str = "hif_honsen",
    p_idol_id: Optional[int] = None,
    loadout_mode: str = "auto",  # 'auto' | 'manual' | 'hybrid'
    ppo_episodes: int = 2000,
    ismcts_sims: int = 64,
    initial_library: Optional[str] = None,
    initial_loadout_dict: Optional[Dict[str, Any]] = None,
    preferences: Optional[Dict[str, Any]] = None,
    library_out: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    debug_dir: Optional[str] = None,
    csv_out: Optional[str] = None,
    on_step_callback: Optional[Callable[[StepResult], None]] = None,
    on_frame_callback: Optional[Callable[[Any], None]] = None,
) -> GakumasRunner:
    """Construct a local zero-LLM GakumasRunner wired with HifLoadoutScanner and PPO+ISMCTS."""
    db = GakumasDB("gakumas_library.db")
    host, port = _parse_host_port(adb_addr)
    default_sdk_adb = Path(r"C:\Users\josha\AppData\Local\Android\Sdk\platform-tools\adb.exe")
    resolved_adb_path = adb_path or (str(default_sdk_adb) if default_sdk_adb.exists() else None)

    controller = AdbController(
        AdbConfig(
            host=host,
            port=port,
            adb_path=resolved_adb_path,
            prefer_maa=True,
        )
    )
    if on_frame_callback is not None:
        orig_screencap = controller.screencap

        def _wrapped_screencap():
            frame = orig_screencap()
            if frame is not None and frame.size > 0:
                try:
                    on_frame_callback(frame)
                except Exception:
                    pass
            return frame

        controller.screencap = _wrapped_screencap  # type: ignore[method-assign]

    seed_data: Optional[Dict[str, Any]] = None
    if initial_loadout_dict and isinstance(initial_loadout_dict, dict):
        seed_data = dict(initial_loadout_dict)
    elif initial_library and Path(initial_library).exists():
        seed_data = json.loads(Path(initial_library).read_text(encoding="utf-8"))

    mode_l = (loadout_mode or "auto").strip().lower()
    if mode_l not in ("auto", "manual", "hybrid"):
        mode_l = "auto"

    # Web-exported config JSONs have initial_loadout_captured=False (only signature cards + memories + preferences):
    # keep HifLoadoutScanner active in hybrid mode unless user explicitly forced manual mode.
    seed_is_unscanned_config = bool(
        seed_data is not None and seed_data.get("initial_loadout_captured") is False
    )
    if seed_is_unscanned_config and mode_l == "auto":
        mode_l = "hybrid"

    effective_prefs_raw = preferences
    if not effective_prefs_raw and seed_data and isinstance(seed_data.get("preferences"), dict):
        effective_prefs_raw = seed_data["preferences"]
    prefs_obj = ProducePreferences.from_dict(effective_prefs_raw)

    init_plan_hint = str(((seed_data or {}).get("plan")) or plan or "logic").lower()
    scanner = None if mode_l == "manual" else HifLoadoutScanner(db=db, plan=init_plan_hint)
    if scanner is not None and hasattr(scanner.reco, "set_banned_drink_ids"):
        scanner.reco.set_banned_drink_ids(prefs_obj.banned_drink_ids)

    agent = LocalHybridDecisionAgent(
        db=db,
        planner_cfg=LiveBattlePlannerConfig(
            ppo_episodes=ppo_episodes,
            ismcts_simulations=ismcts_sims,
        ),
        library_out_path=library_out,
        preferences=prefs_obj,
    )

    dbg_path = Path(debug_dir) if debug_dir else None
    if dbg_path is not None:
        dbg_path.mkdir(parents=True, exist_ok=True)
    saved_battle_tags: set[str] = set()
    saved_ui_dbg_count = [0]
    last_captured_frame: list[Any] = [None]

    orig_screencap_for_dbg = controller.screencap

    def _capture_tracking_screencap():
        frame = orig_screencap_for_dbg()
        if frame is not None and getattr(frame, "size", 0) > 0:
            last_captured_frame[0] = frame
        return frame

    controller.screencap = _capture_tracking_screencap  # type: ignore[method-assign]

    csv_fh = None
    csv_writer = None
    if csv_out:
        csv_path = Path(csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = open(csv_path, "w", encoding="utf-8-sig", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow([
            "step",
            "timestamp",
            "scene",
            "week",
            "turn",
            "turns_remaining",
            "stamina",
            "max_stamina",
            "genki",
            "score",
            "target_score",
            "good_impression",
            "motivation",
            "good_condition",
            "perfect_condition",
            "concentration",
            "buff_hud_occluded",
            "p_points",
            "vocal",
            "dance",
            "visual",
            "hand_cards",
            "drinks",
            "action_type",
            "action_name",
            "action_card_id",
            "action_desc",
            "decision_source",
            "decision_reason",
            "deck_count",
            "deck_ids",
            "week_activity",
            "click_coords",
        ])
        csv_fh.flush()

    def _on_step(res) -> None:
        obs = res.observation
        act = res.action_taken
        dec = res.decision
        wk_note = (
            f" | {obs.week_activity}"
            if obs is not None and obs.scene_type in ("skip", "popup") and obs.score > 0 and obs.week_activity
            else ""
        )
        print(
            f"[Step {res.step_idx:03d}] scene={res.scene_type} W{obs.week if obs else '?'} "
            f"HP={obs.stamina if obs else '?'}/{obs.max_stamina if obs else '?'} "
            f"T={obs.turn if obs else '?'}/rem={obs.turns_remaining if obs else '?'} "
            f"S={obs.score if obs else 0} | {act.description if act else 'WAIT'} | "
            f"{dec.reason if dec else ''}{wk_note}",
            flush=True,
        )
        if csv_writer is not None and csv_fh is not None:
            buffs = (obs.buffs or {}) if obs is not None else {}
            stats = (obs.idol_stats or {}) if obs is not None else {}
            act_payload = (act.payload or {}) if act is not None else {}
            hand_str = "; ".join(
                f"#{c.card_id} {c.name}(p={int(bool(c.playable))},vp={getattr(c, 'visual_playable', None)})"
                for c in ((obs.cards if obs is not None else None) or [])
            )
            drinks_str = "; ".join(
                f"#{d.drink_id} {d.name}"
                for d in ((obs.drinks if obs is not None else None) or [])
            )
            deck_list = list(getattr(runner.tracker, "produce_deck", []) or [])
            csv_writer.writerow([
                res.step_idx,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                res.scene_type,
                obs.week if obs is not None else "",
                obs.turn if obs is not None else "",
                obs.turns_remaining if obs is not None else "",
                obs.stamina if obs is not None else "",
                obs.max_stamina if obs is not None else "",
                obs.genki if obs is not None else "",
                obs.score if obs is not None else "",
                obs.target_score if obs is not None else "",
                buffs.get("goodImpressionTurns", ""),
                buffs.get("motivation", ""),
                buffs.get("goodConditionTurns", ""),
                buffs.get("perfectConditionTurns", ""),
                buffs.get("concentration", ""),
                bool(getattr(obs, "buff_hud_occluded", False)) if obs is not None else "",
                obs.p_points if obs is not None else "",
                stats.get("vocal", ""),
                stats.get("dance", ""),
                stats.get("visual", ""),
                hand_str,
                drinks_str,
                act.action_type if act is not None else "wait",
                act_payload.get("name", "") if act is not None else "",
                act_payload.get("card_id", "") if act is not None else "",
                act.description if act is not None else "WAIT",
                dec.source if dec is not None else "",
                dec.reason if dec is not None else "",
                len(deck_list),
                ",".join(str(x) for x in deck_list),
                obs.week_activity if obs is not None else "",
                f"{res.click_coords[0]},{res.click_coords[1]}" if res.click_coords else "",
            ])
            csv_fh.flush()
        if dbg_path is not None and obs is not None:
            if obs.scene_type == "battle_turn":
                tag = f"W{obs.week}_rem{obs.turns_remaining}"
                if tag not in saved_battle_tags:
                    saved_battle_tags.add(tag)
                    try:
                        raw = last_captured_frame[0] if last_captured_frame[0] is not None else controller.screencap()
                        if raw is not None and raw.size > 0:
                            cv2.imwrite(str(dbg_path / f"battle_{tag}_step{res.step_idx:03d}.png"), cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))
                    except Exception:
                        pass
            elif obs.scene_type in ("card_reward", "shop") and saved_ui_dbg_count[0] < 60:
                saved_ui_dbg_count[0] += 1
                try:
                    raw = last_captured_frame[0]
                    if raw is not None and getattr(raw, "size", 0) > 0:
                        bgr = cv2.cvtColor(raw.copy(), cv2.COLOR_RGB2BGR)
                        if res.click_coords:
                            cx, cy = int(res.click_coords[0]), int(res.click_coords[1])
                            cv2.circle(bgr, (cx, cy), 14, (0, 0, 255), 3)
                            cv2.drawMarker(bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)
                        cv2.imwrite(str(dbg_path / f"{obs.scene_type}_W{obs.week}_step{res.step_idx:03d}.png"), bgr)
                except Exception:
                    pass
        if on_step_callback is not None:
            try:
                on_step_callback(res)
            except Exception:
                pass

    hooks_raw = (seed_data.get("card_memory_hooks") or {}) if seed_data else {}
    manual_hooks_int = {
        int(k): dict(v)
        for k, v in hooks_raw.items()
        if isinstance(v, dict)
    }
    custs_raw = (seed_data.get("card_customizations") or {}) if seed_data else {}
    manual_custs_int = {
        int(k): dict(v)
        for k, v in custs_raw.items()
        if isinstance(v, dict)
    }
    manual_slots = list(seed_data.get("memory_slots") or []) if seed_data else []
    manual_passives = list(seed_data.get("memory_passives") or []) if seed_data else []

    init_pid_arg = (
        int(seed_data["p_idol_id"])
        if seed_data and seed_data.get("p_idol_id") is not None
        else p_idol_id
    )
    init_plan_arg = str((seed_data.get("plan") if seed_data else None) or plan).lower()

    runner = GakumasRunner(
        mode="adb",
        controller=controller,
        agent=agent,
        db=db,
        plan=init_plan_arg,
        scenario_id=scenario_id,
        p_idol_id=init_pid_arg,
        session_mode=False,
        checkpoint_path=checkpoint_path,
        loadout_scanner=scanner,
        loadout_library_path=library_out,
        manual_memory_hooks=manual_hooks_int,
        manual_card_customizations=manual_custs_int,
        manual_memory_slots=manual_slots,
        manual_memory_passives=manual_passives,
        idle_blank_tap_seconds=60.0,
        on_step_completed=_on_step,
    )
    runner._csv_file_handle = csv_fh
    if hasattr(runner.scene_routing_reco, "set_banned_drink_ids"):
        runner.scene_routing_reco.set_banned_drink_ids(prefs_obj.banned_drink_ids)

    if seed_data is not None:
        if mode_l == "manual" or (
            initial_library and Path(initial_library).exists() and not seed_is_unscanned_config and mode_l != "hybrid"
        ):
            runner.tracker.set_initial_loadout(
                p_idol_id=init_pid_arg,
                plan=init_plan_arg,
                recommended_effect=seed_data.get("recommended_effect"),
                deck_ids=[int(cid) for cid in (seed_data.get("deck_ids") or []) if cid],
                p_items=list(seed_data.get("p_items") or []),
                card_memory_hooks=manual_hooks_int,
                card_customizations=manual_custs_int,
                memory_slots=manual_slots,
                memory_passives=manual_passives,
                db=db,
            )
            runner.plan = init_plan_arg
            reco_inner = (
                getattr(runner.scene_routing_reco, "_live_recognizer", None)
                or getattr(runner.scene_routing_reco, "recognizer", None)
            )
            if reco_inner is not None and hasattr(reco_inner, "plan"):
                reco_inner.plan = init_plan_arg
            saved_stats = seed_data.get("idol_stats")
            if isinstance(saved_stats, dict) and max((int(v or 0) for v in saved_stats.values()), default=0) > 300:
                clean_stats = {str(k): int(v or 0) for k, v in saved_stats.items()}
                runner.tracker.idol_stats = dict(clean_stats)
                if reco_inner is not None and hasattr(reco_inner, "_last_stats"):
                    reco_inner._last_stats = dict(clean_stats)
            if init_pid_arg is not None:
                runner.p_idol_id = int(init_pid_arg)
            runner._flush_produce_library()
        elif mode_l in ("hybrid", "auto"):
            # Keep auto-scanner active at produce_setup, while ALWAYS seeding manual memories/slots/passives/customizations
            runner.tracker.memory_slots = [dict(s) for s in manual_slots if isinstance(s, dict)]
            runner.tracker.memory_passives = [dict(p) for p in manual_passives if isinstance(p, dict)]
            if manual_hooks_int or manual_custs_int:
                runner.tracker.merge_manual_memory_hooks(
                    hooks=manual_hooks_int,
                    card_customizations=manual_custs_int,
                )
            saved_stats = seed_data.get("idol_stats")
            if isinstance(saved_stats, dict) and max((int(v or 0) for v in saved_stats.values()), default=0) > 300:
                clean_stats = {str(k): int(v or 0) for k, v in saved_stats.items()}
                runner.tracker.idol_stats = dict(clean_stats)

    return runner


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run zero-LLM local H.I.F 本戦 automation via ADB.")
    parser.add_argument("--adb", default="192.168.0.115:5555", help="ADB host:port")
    parser.add_argument("--adb-path", default=None, help="Explicit adb.exe path")
    parser.add_argument("--plan", default="logic", help="Initial idol plan fallback (logic/sense/anomaly)")
    parser.add_argument("--scenario", default="hif_honsen", help="Produce scenario ID")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum loop steps")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps in seconds")
    parser.add_argument("--ppo-episodes", type=int, default=2000, help="PPO training episodes per battle")
    parser.add_argument("--ismcts-sims", type=int, default=64, help="SO-ISMCTS simulations per battle decision")
    parser.add_argument("--initial-library", default=None, help="Optional initial produce library or Web config JSON")
    parser.add_argument("--loadout-mode", default="auto", choices=("auto", "hybrid", "manual"), help="Loadout capture mode (auto/hybrid/manual)")
    parser.add_argument("--library-out", default=None, help="Optional JSON output path for scanned produce library")
    parser.add_argument("--checkpoint", default=None, help="Optional runner checkpoint JSON path")
    parser.add_argument("--debug-dir", default=None, help="Optional directory to dump battle turn screenshots")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path for per-step structured execution log (UTF-8-BOM)")
    parser.add_argument("--no-stop-home", action="store_true", help="Do not auto-stop upon returning to Home")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = build_local_hif_runner(
        adb_addr=args.adb,
        adb_path=args.adb_path,
        plan=args.plan,
        scenario_id=args.scenario,
        loadout_mode=args.loadout_mode,
        ppo_episodes=args.ppo_episodes,
        ismcts_sims=args.ismcts_sims,
        initial_library=args.initial_library,
        library_out=args.library_out,
        checkpoint_path=args.checkpoint,
        debug_dir=args.debug_dir,
        csv_out=args.csv_out,
    )
    try:
        results = runner.run_loop(
            max_steps=args.max_steps,
            delay_between_steps=args.delay,
            stop_on_home_return=(not args.no_stop_home),
        )
        print(f"Completed {len(results)} step(s). Final scene={results[-1].scene_type if results else 'none'}.")
        return 0
    finally:
        runner.stop()
        if getattr(runner, "_csv_file_handle", None) is not None:
            runner._csv_file_handle.close()
        if hasattr(runner.agent, "close"):
            runner.agent.close()
        if runner.loadout_scanner is not None and hasattr(runner.loadout_scanner, "close"):
            runner.loadout_scanner.close()
        runner.db.close()


if __name__ == "__main__":
    raise SystemExit(main())
