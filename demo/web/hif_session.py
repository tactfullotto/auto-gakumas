"""
H.I.F Workbench Backend Session Manager for the Gakumas Web UI.

Manages:
- 3-profession (sense / logic / anomaly) P-Idol selection & full catalog queries
- Auto / Manual / Hybrid loadout configuration (skills, P-items, stats)
- Manual H.I.F Memory hooks + 4-slot memory cards & passive bonuses + wiki presets
- Produce preferences (priority / avoid / keep cards, score deltas, preferred / banned drinks)
- Live ADB preview, library JSON import/export, and background H.I.F automation runner + SSE updates
"""

from __future__ import annotations

from datetime import datetime
import io
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from demo.core.db import GakumasDB
from demo.core.hif_catalog import (
    HIF_MEMORY_ABILITY_TEMPLATES,
    PLAN_METADATA,
    ProducePreferences,
    build_hif_memory_catalog,
)
from demo.core.idol_guides import (
    P_IDOL_GUIDE_REGISTRY,
    get_default_idol_stats,
    get_idol_stat_order,
    get_p_idol_guide,
)
from demo.core.scenarios import SCENARIO_REGISTRY
from demo.maa_bridge.adb_controller import AdbConfig, AdbController
from demo.maa_bridge.runner import StepResult
from demo.run_hif_local import build_local_hif_runner
from demo.vision.loadout_scanner import KNOWN_P_ITEM_DEFS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_ASSET_ROOTS = (
    (PROJECT_ROOT / "cards").resolve(),
    (PROJECT_ROOT / "drinks").resolve(),
)


def now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def to_asset_url(raw_path: Optional[str]) -> Optional[str]:
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


def encode_png(frame_rgb: np.ndarray) -> Tuple[bytes, int, int]:
    """Encode an RGB uint8 numpy frame as PNG bytes and return (bytes, width, height)."""
    if frame_rgb.ndim == 2:
        img = Image.fromarray(frame_rgb, mode="L").convert("RGB")
    else:
        img = Image.fromarray(frame_rgb[:, :, :3].astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    w, h = img.size
    return buf.getvalue(), w, h


def _default_placeholder_png() -> Tuple[bytes, int, int]:
    w, h = 540, 960
    img = Image.new("RGB", (w, h), (22, 30, 48))
    draw = ImageDraw.Draw(img)
    draw.rectangle([28, 28, w - 28, h - 28], outline=(82, 122, 204), width=2)
    draw.text((120, 420), "Gakumas H.I.F Live Preview", fill=(210, 224, 250))
    draw.text((90, 460), "Click Preview ADB Frame or Start H.I.F Run", fill=(140, 165, 208))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), w, h


def _empty_memory_slots() -> List[Dict[str, Any]]:
    return [
        {"slot": idx, "label": f"メモリー #{idx}", "owner_name": "", "passives": []}
        for idx in range(1, 5)
    ]


_DSL_VAR_LABELS = {
    "g.score": "パラメータ",
    "score": "パラメータ",
    "g.scoreTimes": "打点段数",
    "g.cost": "コスト軽減",
    "g.typedCost": "タイプ消費軽減",
    "g.goodImpressionTurns": "好印象",
    "goodImpressionTurns": "好印象",
    "g.motivation": "やる気",
    "motivation": "やる気",
    "g.genki": "元気",
    "genki": "元気",
    "fixedGenki": "固定元気",
    "g.goodConditionTurns": "好調",
    "goodConditionTurns": "好調",
    "g.perfectConditionTurns": "絶好調",
    "perfectConditionTurns": "絶好調",
    "g.concentration": "集中",
    "concentration": "集中",
    "g.fullPowerCharge": "全力値",
    "fullPowerCharge": "全力値",
    "cardUsesRemaining": "スキル使用数",
}


def _humanize_dsl_clause(clause: str) -> str:
    txt = str(clause or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"^at:[a-zA-Z0-9_]+\{(.+)\}$", r"\1", txt)
    txt = re.sub(r"^if:[^\{]+\{(.+)\}$", r"\1", txt)
    parts: List[str] = []
    for sub in [s.strip() for s in txt.split(";") if s.strip()]:
        m_cost = re.match(r"^(g\.cost|g\.typedCost)\+=(\d+)$", sub)
        if m_cost:
            lbl = "コスト" if m_cost.group(1) == "g.cost" else "固有消費"
            parts.append(f"{lbl}-{m_cost.group(2)}")
            continue
        m_op = re.match(r"^([a-zA-Z0-9_\.]+)(\+=|-=|\*=|=)([\d\.]+)$", sub)
        if m_op:
            var_k, op_s, val_s = m_op.group(1), m_op.group(2), m_op.group(3)
            lbl = _DSL_VAR_LABELS.get(var_k, var_k.replace("g.", ""))
            sign = "+" if op_s == "+=" else ("-" if op_s == "-=" else ("×" if op_s == "*=" else "="))
            parts.append(f"{lbl}{sign}{val_s}")
            continue
        if sub == "drawCard":
            parts.append("カード+1枚引く")
            continue
        m_ps = re.match(r"^preserveSwitch\((\d+)\)$", sub)
        if m_ps:
            parts.append(f"温存切替({m_ps.group(1)}段階)")
            continue
        parts.append(sub)
    return " / ".join(parts)


def _summarize_customization(crow: Dict[str, Any]) -> str:
    if int(crow.get("force_initial_hand") or 0) == 1:
        return "開始時手札に入る（开局固定上手）"
    if crow.get("limit_count") is not None and int(crow["limit_count"]) == 0:
        return "レッスン中1回の制限削除（解除本场限1）"

    raw_patches = crow.get("parsed_patches_json")
    try:
        patches = json.loads(raw_patches) if isinstance(raw_patches, str) else (raw_patches or [])
    except Exception:
        patches = []

    max_lvl = max(1, int(crow.get("max_level") or 1))
    if max_lvl > 1 and isinstance(patches, list) and patches:
        by_lvl: Dict[int, List[str]] = {}
        for p in patches:
            if not isinstance(p, dict):
                continue
            lv = int(p.get("level") or 1)
            acts = p.get("actions") or []
            for a in acts:
                if isinstance(a, dict) and a.get("lhs") and a.get("rhs"):
                    rhs_v = a["rhs"].get("value") if isinstance(a["rhs"], dict) else None
                    if rhs_v is not None:
                        clause = f"{a['lhs']}{a.get('op', '+=')}{rhs_v}"
                        by_lvl.setdefault(lv, []).append(_humanize_dsl_clause(clause))
        if by_lvl:
            segs = [f"Lv{lv}: {'・'.join(txts)}" for lv, txts in sorted(by_lvl.items())]
            return " ｜ ".join(segs)

    dsl_eff = str(crow.get("dsl_effects") or "").strip()
    dsl_act = str(crow.get("dsl_actions") or "").strip()
    dsl_cond = str(crow.get("dsl_conditions") or "").strip()
    pieces: List[str] = []
    if dsl_act:
        pieces.append(_humanize_dsl_clause(dsl_act))
    if dsl_eff:
        pieces.append(_humanize_dsl_clause(dsl_eff))
    if dsl_cond and not pieces:
        pieces.append(f"条件変更: {dsl_cond}")
    return " / ".join(p for p in pieces if p) or str(crow.get("name") or "")


class HifWorkbenchSession:
    """Thread-safe configuration, manual memory/preference state, and live runner manager."""

    def __init__(
        self,
        db: GakumasDB,
        broadcast_fn: Callable[[str, Optional[Dict[str, Any]]], None],
    ):
        self.db = db
        self._broadcast = broadcast_fn
        self.lock = threading.RLock()

        self.selected_plan: str = "logic"
        self.selected_p_idol_id: Optional[int] = None
        self.loadout_mode: str = "hybrid"  # 'auto' | 'manual' | 'hybrid'

        self.run_config: Dict[str, Any] = {
            "adb_host": "127.0.0.1",
            "adb_port": 16384,
            "scenario_id": "hif_honsen",
            "max_steps": 450,
            "delay": 0.5,
            "ppo_episodes": 2000,
            "ismcts_sims": 64,
            "stop_on_home_return": True,
            "library_out": "",
            "csv_out": "",
            "debug_dir": "",
        }

        self.deck_ids: List[int] = []
        self.p_items: List[Dict[str, Any]] = []
        self.idol_stats: Dict[str, int] = {"vocal": 2400, "dance": 1400, "visual": 1800}

        self.card_memory_hooks: Dict[int, Dict[str, Any]] = {}
        self.card_customizations: Dict[int, Dict[str, Any]] = {}
        self.memory_slots: List[Dict[str, Any]] = _empty_memory_slots()
        self.memory_passives: List[Dict[str, Any]] = []
        self.preferences: ProducePreferences = ProducePreferences()

        # Runtime state
        self.running: bool = False
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._active_runner: Optional[Any] = None
        self.step_count: int = 0
        self.frame_seq: int = 0
        self._frame_png_bytes, self.frame_width, self.frame_height = _default_placeholder_png()
        self.frame_source: str = "idle"
        self.last_error: str = ""
        self.runtime_kpi: Dict[str, Any] = {
            "scene_type": "idle",
            "week": 0,
            "turn": 0,
            "turns_remaining": 0,
            "stamina": 0,
            "max_stamina": 0,
            "genki": 0,
            "score": 0,
            "target_score": 0,
            "p_points": 0,
            "deck_count": 0,
            "last_action": "",
            "last_reason": "",
            "last_source": "",
        }
        self.logs: List[Dict[str, Any]] = []

        self._catalog_cache: Optional[Dict[str, Any]] = None

    def add_log(self, level: str, message: str) -> None:
        with self.lock:
            self.logs.append({
                "id": f"hif-{int(time.time() * 1000)}-{len(self.logs)}",
                "time": now_hms(),
                "level": level,
                "message": message,
            })
            if len(self.logs) > 200:
                self.logs = self.logs[-200:]

    def clear_logs(self) -> Dict[str, Any]:
        with self.lock:
            self.logs.clear()
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True}

    def get_frame_png(self) -> bytes:
        with self.lock:
            return self._frame_png_bytes

    def _signature_card_rows(self, p_idol_id: int) -> List[Dict[str, Any]]:
        cur = self.db.connection.cursor()
        rows = cur.execute(
            """
            SELECT c.id, c.base_id, c.name, c.rarity, c.type, c.upgraded, pc.icon_path
            FROM cards c
            LEFT JOIN plan_cards pc ON pc.card_id = c.id
            WHERE c.p_idol_id = ? AND c.is_signature = 1 AND c.hidden = 0
            ORDER BY c.upgraded DESC, c.id ASC
            """,
            (int(p_idol_id),),
        ).fetchall()
        return [dict(r) for r in rows]

    def build_catalog(self) -> Dict[str, Any]:
        if self._catalog_cache is None:
            cur = self.db.connection.cursor()
            sig_rows = cur.execute(
                """
                SELECT c.p_idol_id, c.id, c.name, c.upgraded, c.rarity, pc.icon_path
                FROM cards c
                LEFT JOIN plan_cards pc ON pc.card_id = c.id
                WHERE c.is_signature = 1 AND c.p_idol_id IS NOT NULL AND c.hidden = 0
                ORDER BY c.upgraded ASC, c.id ASC
                """
            ).fetchall()
            sig_by_pid: Dict[int, List[Dict[str, Any]]] = {}
            for r in sig_rows:
                sig_by_pid.setdefault(int(r["p_idol_id"]), []).append(dict(r))

            pidols_out: List[Dict[str, Any]] = []
            for p_row in self.db.get_p_idols():
                pid = int(p_row["id"])
                guide = get_p_idol_guide(pid)
                sigs = sig_by_pid.get(pid, [])
                icon_row = next((s for s in sigs if s.get("icon_path")), None)
                item = dict(p_row)
                item["icon_url"] = to_asset_url(icon_row.get("icon_path")) if icon_row else None
                item["signature_card_ids"] = [int(s["id"]) for s in sigs]
                item["stat_order"] = list(get_idol_stat_order(pid, db=self.db))
                item["has_guide"] = guide is not None
                item["guide_summary"] = (
                    list(guide.core_mechanics[:2]) if guide is not None else []
                )
                item["unique_p_item"] = (
                    dict(guide.unique_p_item) if guide is not None else None
                )
                pidols_out.append(item)

            mem_cat = build_hif_memory_catalog(self.db)
            for _plan, groups in mem_cat.get("plans", {}).items():
                for _gkey, items in groups.items():
                    for entry in items:
                        entry["icon_url"] = to_asset_url(entry.get("icon_path"))
                        entry["upgraded_icon_url"] = to_asset_url(entry.get("upgraded_icon_path"))

            p_items_catalog: List[Dict[str, Any]] = [
                {
                    "item_id": int(d["item_id"]),
                    "name": str(d["name"]),
                    "canonical_name": str(d["name"]),
                    "mode": str(d.get("mode") or "produce"),
                    "source_type": str(d.get("source_type") or "support"),
                    "rarity": str(d.get("rarity") or "SSR"),
                    "description": str(d.get("description") or ""),
                    "p_idol_id": 163 if int(d["item_id"]) == 509 else None,
                }
                for d in KNOWN_P_ITEM_DEFS
            ]
            seen_item_names = {it["canonical_name"] for it in p_items_catalog}
            for g in P_IDOL_GUIDE_REGISTRY.values():
                up = g.unique_p_item or {}
                nm = str(up.get("name") or "").strip()
                if nm and nm not in seen_item_names:
                    seen_item_names.add(nm)
                    p_items_catalog.append({
                        "item_id": abs(hash((g.p_idol_id, nm))) % 90000 + 10000,
                        "name": nm,
                        "canonical_name": nm,
                        "mode": "stage",
                        "source_type": "pIdol",
                        "rarity": g.rarity,
                        "description": f"{up.get('trigger', '')}: {up.get('effect', '')}".strip(": "),
                        "p_idol_id": g.p_idol_id,
                    })

            customizations_map: Dict[str, Dict[str, Any]] = {}
            for crow in self.db.get_all_customizations():
                cid_i = int(crow["id"])
                try:
                    prices = json.loads(crow.get("prices_json") or "[]")
                except Exception:
                    prices = []
                customizations_map[str(cid_i)] = {
                    "id": cid_i,
                    "alias": str(crow.get("alias") or ""),
                    "name": str(crow.get("name") or f"強化#{cid_i}"),
                    "type": str(crow.get("type") or ""),
                    "max_level": max(1, int(crow.get("max_level") or 1)),
                    "prices": prices if isinstance(prices, list) else [],
                    "force_initial_hand": int(crow.get("force_initial_hand") or 0) == 1,
                    "limit_count": crow.get("limit_count"),
                    "summary": _summarize_customization(crow),
                }

            self._catalog_cache = {
                "plans": PLAN_METADATA,
                "p_idols": pidols_out,
                "p_items": p_items_catalog,
                "memory_catalog": mem_cat,
                "customizations": customizations_map,
                "scenarios": [
                    {
                        "scenario_id": sid,
                        "name": spec.name,
                        "short_name": spec.short_name,
                        "total_weeks": spec.total_weeks,
                    }
                    for sid, spec in SCENARIO_REGISTRY.items()
                ],
            }

        libs = sorted(
            [p.name for p in PROJECT_ROOT.glob("produce_library*.json") if p.is_file()],
            reverse=True,
        )
        res = dict(self._catalog_cache)
        res["existing_libraries"] = libs
        return res

    def _enrich_card_summary(self, cid: int) -> Dict[str, Any]:
        from demo.core.idol_guides import resolve_card_family_meta

        row = self.db.get_card_by_id(int(cid))
        hooks = resolve_card_family_meta(self.card_memory_hooks, cid, db=self.db, row=row)
        custs = resolve_card_family_meta(self.card_customizations, cid, db=self.db, row=row)
        avail_cust_ids: List[int] = []
        if row and row.get("available_customization_ids_json"):
            try:
                parsed_ids = json.loads(row["available_customization_ids_json"])
                if isinstance(parsed_ids, list):
                    avail_cust_ids = [int(x) for x in parsed_ids if x is not None]
            except Exception:
                avail_cust_ids = []
        return {
            "card_id": int(cid),
            "base_id": int(row.get("base_id") or cid) if row else int(cid),
            "name": str(row["name"]) if row else f"Card#{cid}",
            "card_type": str(row.get("type") or "") if row else "",
            "rarity": str(row.get("rarity") or "") if row else "",
            "upgraded": bool(row.get("upgraded")) if row else False,
            "limit_count": row.get("limit_count") if row else None,
            "description": str(row.get("description") or "") if row else "",
            "icon_url": to_asset_url(row.get("icon_path")) if row else None,
            "details_url": to_asset_url(row.get("details_path")) if row else None,
            "available_customization_ids": avail_cust_ids,
            "memory_hooks": hooks,
            "customizations": custs,
        }

    def _enrich_drink_summary(self, did: int) -> Dict[str, Any]:
        row = self.db.get_drink_by_id(int(did))
        return {
            "drink_id": int(did),
            "name": str(row["name"]) if row else f"Drink#{did}",
            "rarity": str(row.get("rarity") or "") if row else "",
            "plan": str(row.get("plan") or "") if row else "",
            "description": str(row.get("description") or "") if row else "",
            "icon_url": to_asset_url(row.get("icon_path")) if row else None,
        }

    def _seed_idol_defaults(self, pid: int) -> None:
        self.idol_stats = get_default_idol_stats(pid, db=self.db)
        sig_rows = self._signature_card_rows(pid)
        non_l = [r for r in sig_rows if str(r.get("rarity") or "").upper() != "L"]
        by_base: Dict[int, Dict[str, Any]] = {}
        for r in non_l:
            bid = int(r.get("base_id") or r["id"])
            if bid not in by_base or (bool(r.get("upgraded")) and not bool(by_base[bid].get("upgraded"))):
                by_base[bid] = r
        self.deck_ids = [int(r["id"]) for r in by_base.values()]

        guide = get_p_idol_guide(pid)
        if guide is not None:
            up = guide.unique_p_item or {}
            nm = str(up.get("name") or "").strip()
            known = next(
                (d for d in KNOWN_P_ITEM_DEFS if nm and (nm in d["name"] or d["name"] in nm)),
                None,
            )
            if known is not None:
                self.p_items = [{
                    "item_id": int(known["item_id"]),
                    "name": str(known["name"]),
                    "canonical_name": str(known["name"]),
                    "mode": str(known.get("mode") or "stage"),
                    "source_type": str(known.get("source_type") or "pIdol"),
                    "rarity": str(known.get("rarity") or "SSR"),
                    "description": str(known.get("description") or ""),
                }]
            elif nm:
                self.p_items = [{
                    "item_id": abs(hash((pid, nm))) % 90000 + 10000,
                    "name": nm,
                    "canonical_name": nm,
                    "mode": "stage",
                    "source_type": "pIdol",
                    "rarity": guide.rarity,
                    "description": f"{up.get('trigger', '')}: {up.get('effect', '')}".strip(": "),
                }]
        else:
            self.p_items = []

    def select_idol(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        plan_in = payload.get("plan")
        raw_pid = payload.get("p_idol_id")
        auto_seed = bool(payload.get("auto_seed_signature", True))

        pid_val: Optional[int] = None
        p_row: Optional[Dict[str, Any]] = None
        if raw_pid not in (None, "", 0):
            pid_val = int(raw_pid)
            p_row = self.db.get_p_idol_by_id(pid_val)
            if p_row is None:
                raise KeyError(f"Unknown p_idol_id: {pid_val}")

        if plan_in is not None:
            plan_str = str(plan_in).strip().lower()
            if plan_str not in PLAN_METADATA:
                raise ValueError(f"Invalid plan: {plan_str!r}")
        elif p_row is not None:
            plan_str = str(p_row["plan"]).lower()
        else:
            plan_str = self.selected_plan

        with self.lock:
            prev_pid = self.selected_p_idol_id
            self.selected_plan = plan_str
            if pid_val is not None:
                self.selected_p_idol_id = pid_val
                if p_row and p_row.get("plan"):
                    self.selected_plan = str(p_row["plan"]).lower()
                if auto_seed and (prev_pid != pid_val or not self.deck_ids):
                    self._seed_idol_defaults(pid_val)
            elif "p_idol_id" in payload and raw_pid is None:
                self.selected_p_idol_id = None

            label = f"[{p_row['title']}] {p_row['idol_name']}" if p_row else self.selected_plan
            self.add_log("info", f"已切换培养角色 -> {label} ({self.selected_plan.upper()})")

        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True, "state": self.get_state()}

    def update_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if "loadout_mode" in payload and payload["loadout_mode"] is not None:
                lm = str(payload["loadout_mode"]).strip().lower()
                if lm not in ("auto", "manual", "hybrid"):
                    raise ValueError(f"Invalid loadout_mode: {lm!r}")
                self.loadout_mode = lm

            if "adb_host" in payload and payload["adb_host"] is not None:
                self.run_config["adb_host"] = str(payload["adb_host"]).strip() or "127.0.0.1"
            if "adb_port" in payload and payload["adb_port"] is not None:
                self.run_config["adb_port"] = max(1, min(65535, int(payload["adb_port"])))
            if "scenario_id" in payload and payload["scenario_id"] is not None:
                sid = str(payload["scenario_id"]).strip()
                if sid not in SCENARIO_REGISTRY:
                    raise ValueError(f"Invalid scenario_id: {sid!r}")
                self.run_config["scenario_id"] = sid
            if "max_steps" in payload and payload["max_steps"] is not None:
                self.run_config["max_steps"] = max(1, min(2000, int(payload["max_steps"])))
            if "delay" in payload and payload["delay"] is not None:
                self.run_config["delay"] = max(0.0, min(10.0, float(payload["delay"])))
            if "ppo_episodes" in payload and payload["ppo_episodes"] is not None:
                self.run_config["ppo_episodes"] = max(100, min(10000, int(payload["ppo_episodes"])))
            if "ismcts_sims" in payload and payload["ismcts_sims"] is not None:
                self.run_config["ismcts_sims"] = max(8, min(512, int(payload["ismcts_sims"])))
            if "stop_on_home_return" in payload and payload["stop_on_home_return"] is not None:
                self.run_config["stop_on_home_return"] = bool(payload["stop_on_home_return"])
            for txt_k in ("library_out", "csv_out", "debug_dir"):
                if txt_k in payload and payload[txt_k] is not None:
                    self.run_config[txt_k] = str(payload[txt_k]).strip()

            self.add_log(
                "info",
                f"已更新 H.I.F 运行配置 (mode={self.loadout_mode}, adb={self.run_config['adb_host']}:{self.run_config['adb_port']})",
            )
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True, "state": self.get_state()}

    def _normalize_card_customization_entry(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        levels_out: Dict[str, int] = {}
        raw_lvls = raw.get("levels") if isinstance(raw.get("levels"), dict) else {}
        for rk, rv in raw_lvls.items():
            try:
                cust_id = int(rk)
                lvl = int(rv or 0)
            except (TypeError, ValueError):
                continue
            if lvl <= 0:
                continue
            crow = self.db.get_customization_by_id(cust_id)
            if crow is None:
                continue
            max_l = max(1, int(crow.get("max_level") or 1))
            levels_out[str(cust_id)] = min(max_l, lvl)

        mods_src = raw.get("modifiers") if isinstance(raw.get("modifiers"), dict) else {}
        mods_out: Dict[str, Any] = {}
        if bool(mods_src.get("force_initial_hand")):
            mods_out["force_initial_hand"] = True
        if bool(mods_src.get("remove_limit")):
            mods_out["remove_limit"] = True
        for nk in (
            "cost_reduction",
            "score_delta",
            "good_impression_delta",
            "motivation_delta",
            "genki_delta",
            "good_condition_delta",
            "concentration_delta",
            "full_power_charge_delta",
        ):
            if nk in mods_src and mods_src[nk] not in (None, "", 0, "0"):
                try:
                    iv = max(-50, min(200, int(mods_src[nk])))
                    if iv != 0:
                        mods_out[nk] = iv
                except (TypeError, ValueError):
                    pass

        out: Dict[str, Any] = {}
        if levels_out:
            out["levels"] = levels_out
        if mods_out:
            out["modifiers"] = mods_out
        return out

    def update_loadout(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            if "seed_signature" in payload and payload["seed_signature"] and self.selected_p_idol_id:
                self._seed_idol_defaults(self.selected_p_idol_id)
            if "deck_ids" in payload and payload["deck_ids"] is not None:
                cleaned_deck: List[int] = []
                for raw_cid in payload["deck_ids"]:
                    cid = int(raw_cid)
                    if self.db.get_card_by_id(cid) is not None:
                        cleaned_deck.append(cid)
                self.deck_ids = cleaned_deck
            if "p_items" in payload and payload["p_items"] is not None:
                cleaned_items: List[Dict[str, Any]] = []
                for raw_it in payload["p_items"]:
                    if not isinstance(raw_it, dict):
                        continue
                    nm = str(raw_it.get("canonical_name") or raw_it.get("name") or "").strip()
                    if not nm:
                        continue
                    iid = int(raw_it.get("item_id") or (abs(hash(nm)) % 90000 + 10000))
                    cleaned_items.append({
                        "item_id": iid,
                        "name": str(raw_it.get("name") or nm),
                        "canonical_name": nm,
                        "mode": str(raw_it.get("mode") or "stage"),
                        "source_type": str(raw_it.get("source_type") or "support"),
                        "rarity": str(raw_it.get("rarity") or "SSR"),
                        "description": str(raw_it.get("description") or ""),
                    })
                self.p_items = cleaned_items
            if isinstance(payload.get("idol_stats"), dict):
                st = payload["idol_stats"]
                for sk in ("vocal", "dance", "visual"):
                    if sk in st and st[sk] is not None:
                        self.idol_stats[sk] = max(0, min(5000, int(st[sk])))
            if "card_customizations" in payload and isinstance(payload["card_customizations"], dict):
                cleaned_custs: Dict[int, Dict[str, Any]] = {}
                for raw_cid, raw_entry in payload["card_customizations"].items():
                    if not isinstance(raw_entry, dict):
                        continue
                    try:
                        cid_i = int(raw_cid)
                    except (TypeError, ValueError):
                        continue
                    if self.db.get_card_by_id(cid_i) is None:
                        continue
                    norm_c = self._normalize_card_customization_entry(raw_entry)
                    if norm_c:
                        cleaned_custs[cid_i] = norm_c
                self.card_customizations = cleaned_custs
            self.add_log(
                "info",
                f"已保存技能卡/道具录入 (卡组 {len(self.deck_ids)} 张, P道具 {len(self.p_items)} 件, 自定义强化 {len(self.card_customizations)} 张)",
            )
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True, "state": self.get_state()}

    @staticmethod
    def _normalize_hook_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "force_initial_hand" in raw:
            out["force_initial_hand"] = bool(raw["force_initial_hand"])
        if "exiles_trouble_on_play" in raw:
            out["exiles_trouble_on_play"] = bool(raw["exiles_trouble_on_play"])
        extra_lim = int(raw.get("extra_use_limit") or 0)
        free_extra = bool(raw.get("free_extra_use_first_play", extra_lim > 0))
        if free_extra and extra_lim <= 0:
            extra_lim = 1
        if "free_extra_use_first_play" in raw or "extra_use_limit" in raw:
            out["free_extra_use_first_play"] = bool(free_extra)
            out["extra_use_limit"] = max(0, min(5, extra_lim if free_extra else 0))
        if "draw_on_play" in raw:
            out["draw_on_play"] = max(0, min(3, int(raw.get("draw_on_play") or 0)))
        if raw.get("note"):
            out["note"] = str(raw["note"])[:120]
        return out

    def _apply_memories_locked(self, payload: Dict[str, Any]) -> None:
        if "card_memory_hooks" in payload and isinstance(payload["card_memory_hooks"], dict):
            cleaned_hooks: Dict[int, Dict[str, Any]] = {}
            for raw_k, raw_v in payload["card_memory_hooks"].items():
                if not isinstance(raw_v, dict):
                    continue
                cid = int(raw_k)
                if self.db.get_card_by_id(cid) is None:
                    continue
                norm = self._normalize_hook_entry(raw_v)
                if any(
                    bool(norm.get(k))
                    for k in ("force_initial_hand", "exiles_trouble_on_play", "free_extra_use_first_play", "draw_on_play")
                ) or "force_initial_hand" in norm:
                    cleaned_hooks[cid] = norm
            self.card_memory_hooks = cleaned_hooks

        if "memory_slots" in payload and isinstance(payload["memory_slots"], list):
            slots_out: List[Dict[str, Any]] = []
            derived_passives: List[Dict[str, Any]] = []
            for idx in range(1, 5):
                src = (
                    payload["memory_slots"][idx - 1]
                    if (idx - 1) < len(payload["memory_slots"]) and isinstance(payload["memory_slots"][idx - 1], dict)
                    else {}
                )
                passives_clean: List[Dict[str, Any]] = []
                for mp in src.get("passives") or ():
                    if not isinstance(mp, dict):
                        continue
                    key = str(mp.get("key") or "").strip()
                    if not key:
                        continue
                    entry = {
                        "key": key,
                        "label": str(mp.get("label") or key),
                        "value": int(mp.get("value") or 0),
                        "note": str(mp.get("note") or "").strip(),
                    }
                    passives_clean.append(entry)
                    derived_passives.append(entry)
                slots_out.append({
                    "slot": idx,
                    "label": str(src.get("label") or f"メモリー #{idx}"),
                    "owner_name": str(src.get("owner_name") or ""),
                    "passives": passives_clean,
                })
            self.memory_slots = slots_out
            if "memory_passives" not in payload:
                self.memory_passives = derived_passives

        if "memory_passives" in payload and isinstance(payload["memory_passives"], list):
            self.memory_passives = [
                {
                    "key": str(mp.get("key") or ""),
                    "label": str(mp.get("label") or mp.get("key") or ""),
                    "value": int(mp.get("value") or 0),
                    "note": str(mp.get("note") or "").strip(),
                }
                for mp in payload["memory_passives"]
                if isinstance(mp, dict) and mp.get("key")
            ]

    def update_memories(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self._apply_memories_locked(payload)
            self.add_log(
                "info",
                f"已保存手动回忆设置 ({len(self.card_memory_hooks)} 条技能回忆效果, {len(self.memory_passives)} 条数值词条)",
            )
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True, "state": self.get_state()}

    def update_preferences(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prefs = ProducePreferences.from_dict(payload)
        prefs.priority_card_ids = [cid for cid in prefs.priority_card_ids if self.db.get_card_by_id(cid) is not None]
        prefs.avoid_card_ids = [cid for cid in prefs.avoid_card_ids if self.db.get_card_by_id(cid) is not None]
        prefs.keep_card_ids = [cid for cid in prefs.keep_card_ids if self.db.get_card_by_id(cid) is not None]
        prefs.preferred_drink_ids = [did for did in prefs.preferred_drink_ids if self.db.get_drink_by_id(did) is not None]
        prefs.banned_drink_ids = [did for did in prefs.banned_drink_ids if self.db.get_drink_by_id(did) is not None]

        with self.lock:
            self.preferences = prefs
            self.add_log(
                "info",
                f"已保存培育偏好 (高优卡 {len(prefs.priority_card_ids)} / 避坑卡 {len(prefs.avoid_card_ids)} / "
                f"优先饮料 {len(prefs.preferred_drink_ids)} / 禁用饮料 {len(prefs.banned_drink_ids)})",
            )
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True, "state": self.get_state()}

    def build_library_payload(self) -> Dict[str, Any]:
        with self.lock:
            pid = self.selected_p_idol_id
            p_row = self.db.get_p_idol_by_id(pid) if pid else None
            rec_eff = str(p_row["recommended_effect"]) if (p_row and p_row.get("recommended_effect")) else None
            card_entries = [self._enrich_card_summary(cid) for cid in self.deck_ids]
            return {
                "p_idol_id": pid,
                "plan": self.selected_plan,
                "scenario_id": self.run_config.get("scenario_id", "hif_honsen"),
                "recommended_effect": rec_eff,
                "initial_loadout_captured": self.loadout_mode == "manual",
                "idol_stats": dict(self.idol_stats),
                "deck_count": len(self.deck_ids),
                "deck_ids": list(self.deck_ids),
                "cards": card_entries,
                "p_items": [dict(it) for it in self.p_items],
                "card_memory_hooks": {str(k): dict(v) for k, v in self.card_memory_hooks.items()},
                "card_customizations": {str(k): dict(v) for k, v in self.card_customizations.items()},
                "memory_slots": [dict(s) for s in self.memory_slots],
                "memory_passives": [dict(p) for p in self.memory_passives],
                "preferences": self.preferences.to_dict(),
            }

    def import_library(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_data: Optional[Dict[str, Any]] = None
        if isinstance(payload.get("data"), dict):
            raw_data = payload["data"]
        elif payload.get("filename"):
            fname = str(payload["filename"]).strip()
            if (
                not fname
                or ".." in fname
                or "/" in fname
                or "\\" in fname
                or not fname.lower().endswith(".json")
            ):
                raise PermissionError(f"Invalid library filename: {fname!r}")
            target = (PROJECT_ROOT / fname).resolve()
            if target.parent != PROJECT_ROOT.resolve():
                raise PermissionError(f"Path outside project root denied: {fname!r}")
            if not target.is_file():
                raise KeyError(f"Library file not found: {fname}")
            raw_data = json.loads(target.read_text(encoding="utf-8"))
        else:
            raise ValueError("Provide either 'filename' or 'data' to import")

        if not isinstance(raw_data, dict):
            raise ValueError("Imported library payload must be a JSON object")

        with self.lock:
            imp_pid = raw_data.get("p_idol_id")
            if imp_pid is not None:
                p_row = self.db.get_p_idol_by_id(int(imp_pid))
                if p_row is not None:
                    self.selected_p_idol_id = int(imp_pid)
                    self.selected_plan = str(p_row.get("plan") or raw_data.get("plan") or self.selected_plan).lower()
            elif raw_data.get("plan") in PLAN_METADATA:
                self.selected_plan = str(raw_data["plan"]).lower()

            if raw_data.get("scenario_id") in SCENARIO_REGISTRY:
                self.run_config["scenario_id"] = str(raw_data["scenario_id"])
            if isinstance(raw_data.get("idol_stats"), dict):
                for sk in ("vocal", "dance", "visual"):
                    if sk in raw_data["idol_stats"] and raw_data["idol_stats"][sk] is not None:
                        self.idol_stats[sk] = max(0, min(5000, int(raw_data["idol_stats"][sk])))

            deck_raw = raw_data.get("deck_ids") or [
                c.get("card_id") for c in (raw_data.get("cards") or ()) if isinstance(c, dict)
            ]
            self.deck_ids = [
                int(cid) for cid in (deck_raw or ())
                if cid and self.db.get_card_by_id(int(cid)) is not None
            ]
            self.p_items = [
                dict(it) for it in (raw_data.get("p_items") or ()) if isinstance(it, dict)
            ]
            hooks_raw = raw_data.get("card_memory_hooks") or {}
            self.card_memory_hooks = {
                int(k): self._normalize_hook_entry(v)
                for k, v in hooks_raw.items()
                if isinstance(v, dict) and self.db.get_card_by_id(int(k)) is not None
            }
            custs_raw = raw_data.get("card_customizations") or {}
            cleaned_imp_custs: Dict[int, Dict[str, Any]] = {}
            if isinstance(custs_raw, dict):
                for raw_k, raw_v in custs_raw.items():
                    if not isinstance(raw_v, dict):
                        continue
                    try:
                        cid_k = int(raw_k)
                    except (TypeError, ValueError):
                        continue
                    if self.db.get_card_by_id(cid_k) is None:
                        continue
                    norm_c = self._normalize_card_customization_entry(raw_v)
                    if norm_c:
                        cleaned_imp_custs[cid_k] = norm_c
            self.card_customizations = cleaned_imp_custs
            imp_slots = raw_data.get("memory_slots")
            imp_passives = raw_data.get("memory_passives")
            if isinstance(imp_slots, list):
                self._apply_memories_locked({"memory_slots": imp_slots})
            if isinstance(imp_passives, list):
                if imp_passives or not self.memory_passives:
                    self._apply_memories_locked({"memory_passives": imp_passives})
                if imp_passives and not any(s.get("passives") for s in self.memory_slots):
                    if not self.memory_slots:
                        self.memory_slots = _empty_memory_slots()
                    self.memory_slots[0]["passives"] = [dict(p) for p in self.memory_passives]
            if isinstance(raw_data.get("preferences"), dict):
                self.preferences = ProducePreferences.from_dict(raw_data["preferences"])

            src_name = payload.get("filename") or "JSON"
            self.add_log(
                "info",
                f"已导入配置 ({src_name}): p_idol={self.selected_p_idol_id}, 卡组 {len(self.deck_ids)} 张, "
                f"P道具 {len(self.p_items)} 件, 回忆 {len(self.card_memory_hooks)} 条",
            )
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True, "state": self.get_state()}

    def capture_preview(self) -> Dict[str, Any]:
        host = str(self.run_config["adb_host"])
        port = int(self.run_config["adb_port"])
        default_sdk_adb = Path(r"C:\Users\josha\AppData\Local\Android\Sdk\platform-tools\adb.exe")
        resolved_adb_path = str(default_sdk_adb) if default_sdk_adb.exists() else None
        ctrl = AdbController(
            AdbConfig(
                host=host,
                port=port,
                adb_path=resolved_adb_path,
                prefer_maa=True,
                connect_timeout=3.0,
                command_timeout=4.0,
                fallback_to_mock=False,
            )
        )
        try:
            if not ctrl.connect():
                msg = f"无法连接 ADB ({host}:{port})"
                self.add_log("error", msg)
                self._broadcast("hif_state", {"ts": time.time()})
                return {"ok": False, "message": msg}
            frame = ctrl.screencap()
            png, w, h = encode_png(frame)
            with self.lock:
                self._frame_png_bytes = png
                self.frame_width = w
                self.frame_height = h
                self.frame_seq += 1
                self.frame_source = f"adb://{host}:{port}"
                self.last_error = ""
            self.add_log("info", f"ADB 实机画面抓取成功 ({host}:{port}, {w}x{h})")
            self._broadcast("hif_state", {"ts": time.time()})
            return {"ok": True, "width": w, "height": h, "message": f"截图成功 ({w}x{h})"}
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
            self.add_log("error", f"ADB 预览抓取异常: {exc}")
            self._broadcast("hif_state", {"ts": time.time()})
            return {"ok": False, "message": f"ADB 抓取失败: {exc}"}
        finally:
            try:
                ctrl.disconnect()
            except Exception:
                pass

    def _on_runner_frame(self, frame: np.ndarray) -> None:
        try:
            png, w, h = encode_png(frame)
            with self.lock:
                self._frame_png_bytes = png
                self.frame_width = w
                self.frame_height = h
                self.frame_seq += 1
                self.frame_source = f"adb://{self.run_config['adb_host']}:{self.run_config['adb_port']}"
        except Exception:
            pass

    def _on_runner_step(self, res: StepResult) -> None:
        obs = res.observation
        act = res.action_taken
        dec = res.decision
        with self.lock:
            self.step_count = int(res.step_idx)
            runner = self._active_runner
            deck_len = len(getattr(runner.tracker, "produce_deck", []) or []) if runner else len(self.deck_ids)
            if obs is not None:
                self.runtime_kpi = {
                    "scene_type": obs.scene_type,
                    "week": int(obs.week or 0),
                    "turn": int(obs.turn or 0),
                    "turns_remaining": int(obs.turns_remaining or 0),
                    "stamina": int(obs.stamina or 0),
                    "max_stamina": int(obs.max_stamina or 0),
                    "genki": int(obs.genki or 0),
                    "score": int(obs.score or 0),
                    "target_score": int(obs.target_score or 0),
                    "p_points": int(obs.p_points or 0),
                    "deck_count": deck_len,
                    "last_action": act.description if act else "WAIT",
                    "last_reason": dec.reason if dec else "",
                    "last_source": dec.source if dec else "",
                }
            log_lvl = "error" if not res.success else (dec.source if dec else "info")
            msg = (
                f"Step #{res.step_idx:03d} [{res.scene_type}] "
                f"W{obs.week if obs else '?'} T{obs.turn if obs else '?'}/rem={obs.turns_remaining if obs else '?'} "
                f"-> {act.description if act else 'WAIT'} | {dec.reason if dec else (res.error or '')}"
            )
            self.logs.append({
                "id": f"hif-step-{res.step_idx}-{int(time.time() * 1000)}",
                "time": now_hms(),
                "level": log_lvl,
                "message": msg,
            })
            if len(self.logs) > 200:
                self.logs = self.logs[-200:]
        self._broadcast("hif_state", {"ts": time.time()})

    def start(self) -> Dict[str, Any]:
        with self.lock:
            if self.running:
                return {"ok": True, "already_running": True}
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return {"ok": True, "already_running": True}

            cfg = dict(self.run_config)
            loadout_mode = self.loadout_mode
            plan = self.selected_plan
            p_idol_id = self.selected_p_idol_id
            loadout_dict = self.build_library_payload()
            prefs_dict = self.preferences.to_dict()

            ts_tag = time.strftime("%Y%m%d_%H%M%S")
            pid_tag = str(p_idol_id or "auto")
            if not str(cfg.get("csv_out") or "").strip():
                cfg["csv_out"] = str(PROJECT_ROOT / f"hif_run_{plan}_{pid_tag}_{ts_tag}.csv")
            if not str(cfg.get("library_out") or "").strip():
                cfg["library_out"] = str(PROJECT_ROOT / f"produce_library_{plan}_{pid_tag}_{ts_tag}.json")

            self.running = True
            self._stop_event.clear()
            self.last_error = ""
            self.step_count = 0
            self.add_log(
                "info",
                f"启动 H.I.F 自动化: plan={plan}, p_idol={p_idol_id}, mode={loadout_mode}, "
                f"adb={cfg['adb_host']}:{cfg['adb_port']} | 回忆钩子 {len(self.card_memory_hooks)} 条, "
                f"开局词条 {len(self.memory_passives)} 条, "
                f"单卡强化 {len(self.card_customizations)} 张, 三维 Vo={self.idol_stats.get('vocal')}/Da={self.idol_stats.get('dance')}/Vi={self.idol_stats.get('visual')} | "
                f"日志: {Path(cfg['csv_out']).name}",
            )

        def _worker() -> None:
            runner = None
            try:
                runner = build_local_hif_runner(
                    adb_addr=f"{cfg['adb_host']}:{cfg['adb_port']}",
                    plan=plan,
                    scenario_id=str(cfg["scenario_id"]),
                    p_idol_id=p_idol_id,
                    loadout_mode=loadout_mode,
                    ppo_episodes=int(cfg["ppo_episodes"]),
                    ismcts_sims=int(cfg["ismcts_sims"]),
                    initial_loadout_dict=loadout_dict,
                    preferences=prefs_dict,
                    library_out=cfg["library_out"] or None,
                    debug_dir=cfg["debug_dir"] or None,
                    csv_out=cfg["csv_out"] or None,
                    on_step_callback=self._on_runner_step,
                    on_frame_callback=self._on_runner_frame,
                )
                with self.lock:
                    self._active_runner = runner
                runner.run_loop(
                    max_steps=int(cfg["max_steps"]),
                    delay_between_steps=float(cfg["delay"]),
                    stop_on_home_return=bool(cfg["stop_on_home_return"]),
                )
                self.add_log("info", "H.I.F 自动化运行已结束")
            except Exception as exc:
                logger.exception("H.I.F workbench worker failed")
                with self.lock:
                    self.last_error = str(exc)
                self.add_log("error", f"H.I.F 自动化异常中止: {exc}")
            finally:
                if runner is not None:
                    try:
                        runner.stop()
                    except Exception:
                        pass
                    if getattr(runner, "_csv_file_handle", None) is not None:
                        try:
                            runner._csv_file_handle.close()
                        except Exception:
                            pass
                    if hasattr(runner.agent, "close"):
                        try:
                            runner.agent.close()
                        except Exception:
                            pass
                    if runner.loadout_scanner is not None and hasattr(runner.loadout_scanner, "close"):
                        try:
                            runner.loadout_scanner.close()
                        except Exception:
                            pass
                    try:
                        runner.db.close()
                    except Exception:
                        pass
                with self.lock:
                    self._active_runner = None
                    self.running = False
                self._broadcast("hif_state", {"ts": time.time()})

        t = threading.Thread(target=_worker, name="gakumas-hif-worker", daemon=True)
        with self.lock:
            self._worker_thread = t
        t.start()
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True}

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            self._stop_event.set()
            runner = self._active_runner
            was_running = self.running
            self.running = False
        if runner is not None:
            try:
                runner.stop()
            except Exception:
                pass
        if was_running:
            self.add_log("info", "已请求停止 H.I.F 自动化")
        self._broadcast("hif_state", {"ts": time.time()})
        return {"ok": True}

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            p_row = self.db.get_p_idol_by_id(self.selected_p_idol_id) if self.selected_p_idol_id else None
            p_idol_info: Optional[Dict[str, Any]] = None
            if p_row is not None:
                guide = get_p_idol_guide(int(p_row["id"]))
                sigs = self._signature_card_rows(int(p_row["id"]))
                icon_row = next((s for s in sigs if s.get("icon_path")), None)
                p_idol_info = {
                    **dict(p_row),
                    "icon_url": to_asset_url(icon_row.get("icon_path")) if icon_row else None,
                    "has_guide": guide is not None,
                    "unique_p_item": dict(guide.unique_p_item) if guide is not None else None,
                    "guide_mechanics": list(guide.core_mechanics) if guide is not None else [],
                    "guide_drinks": list(guide.drink_synergies) if guide is not None else [],
                    "guide_avoids": list(guide.anti_synergies) if guide is not None else [],
                }

            deck_entries = [self._enrich_card_summary(cid) for cid in self.deck_ids]
            hooks_enriched: Dict[str, Dict[str, Any]] = {}
            for cid, meta in self.card_memory_hooks.items():
                summary = self._enrich_card_summary(cid)
                hooks_enriched[str(cid)] = {
                    **summary,
                    "hooks": dict(meta),
                }

            prefs_dict = self.preferences.to_dict()
            prefs_enriched = {
                **prefs_dict,
                "priority_cards": [self._enrich_card_summary(cid) for cid in self.preferences.priority_card_ids],
                "avoid_cards": [self._enrich_card_summary(cid) for cid in self.preferences.avoid_card_ids],
                "keep_cards": [self._enrich_card_summary(cid) for cid in self.preferences.keep_card_ids],
                "preferred_drinks": [self._enrich_drink_summary(did) for did in self.preferences.preferred_drink_ids],
                "banned_drinks": [self._enrich_drink_summary(did) for did in self.preferences.banned_drink_ids],
            }

            return {
                "selected_plan": self.selected_plan,
                "selected_p_idol_id": self.selected_p_idol_id,
                "p_idol": p_idol_info,
                "loadout_mode": self.loadout_mode,
                "run_config": dict(self.run_config),
                "idol_stats": dict(self.idol_stats),
                "deck_ids": list(self.deck_ids),
                "deck_cards": deck_entries,
                "p_items": [dict(it) for it in self.p_items],
                "card_memory_hooks": {str(k): dict(v) for k, v in self.card_memory_hooks.items()},
                "card_customizations": {str(k): dict(v) for k, v in self.card_customizations.items()},
                "hooks_enriched": hooks_enriched,
                "memory_slots": [dict(s) for s in self.memory_slots],
                "memory_passives": [dict(p) for p in self.memory_passives],
                "preferences": prefs_enriched,
                "running": self.running,
                "step_count": self.step_count,
                "frame_seq": self.frame_seq,
                "frame_meta": {"width": self.frame_width, "height": self.frame_height},
                "frame_source": self.frame_source,
                "last_error": self.last_error,
                "runtime_kpi": dict(self.runtime_kpi),
                "logs": list(self.logs[-80:]),
            }
