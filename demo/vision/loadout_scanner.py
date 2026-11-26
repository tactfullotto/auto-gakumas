"""Long-press / Detail Modal Loadout Scanner for H.I.F 本戦 initial cards & P-items."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import difflib
from itertools import product
import json
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from demo.core.db import GakumasDB
from demo.maa_bridge.base_controller import BaseController
from demo.vision.live_adb_recognizer import LiveScreenRecognizer, _clean_ocr_text, detect_modal_back_button_xy
from demo.vision.win_ocr import get_win_ocr


KNOWN_P_ITEM_DEFS: List[Dict[str, Any]] = [
    {
        "item_id": 509,
        "name": "星々を統べる輝き+",
        "aliases": ("星々を統べる輝き+", "星々を統べる輝き"),
        "mode": "stage",
        "source_type": "pIdol",
        "rarity": "SSR",
        "description": "直接効果で好印象が4回増加毎: 2T 好印象追加発動+1 & 好印象強化+10%",
    },
    {
        "item_id": 404,
        "name": "咲季の完全食・改",
        "aliases": ("咲季の完全食・改", "咲季の完全食"),
        "mode": "stage",
        "source_type": "support",
        "rarity": "SSR",
        "description": "ターン内にスキルカードを2回使用時: 好印象+2(間接) & 永久好印象強化+25% (レッスン内1回)",
    },
    {
        "item_id": 429,
        "name": "H.I.F応援棒（黄）",
        "aliases": ("H.I.F応援棒", "応援棒"),
        "mode": "stage",
        "source_type": "produce",
        "rarity": "SSR",
        "description": "試験開始時: 山札が22枚未満なら基本スキルカードをランダム位置に補填",
    },
    {
        "item_id": 436,
        "name": "ポーチ（黄）",
        "aliases": ("花ロボ(黄)", "花ロボ（黄）", "花ロボ", "ポーチ(黄)", "ポーチ（黄）", "ポーチ"),
        "mode": "produce",
        "source_type": "produce",
        "rarity": "SSR",
        "description": "レッスン終了時Pドリンク獲得・以降の試験開始時 好印象/やる気上昇",
    },
    {
        "item_id": 447,
        "name": "H.I.Fワッペン",
        "aliases": ("H.I.Fワッペン", "HIFワッペン", "ワッペン"),
        "mode": "produce",
        "source_type": "produce",
        "rarity": "SSR",
        "description": "スキルカード獲得時 スター性+10",
    },
    {
        "item_id": 61,
        "name": "美鈴からの贈り物",
        "aliases": ("美鈴からの贈り物",),
        "mode": "produce",
        "source_type": "support",
        "rarity": "SR",
        "description": "Vo SPレッスン終了時 ランダムなスキルカードを強化",
    },
    {
        "item_id": 510,
        "name": "過激め？チョイス",
        "aliases": ("過激め?チョイス", "過激め？チョイス"),
        "mode": "produce",
        "source_type": "support",
        "rarity": "SSR",
        "description": "ロジックのスキルカード獲得時 ボーカル上昇+25 & ランダムなスキルカードを強化",
    },
    {
        "item_id": 487,
        "name": "素敵なお仕事",
        "aliases": ("素敵なお仕事",),
        "mode": "produce",
        "source_type": "support",
        "rarity": "SSR",
        "description": "スキルカードチェンジ時 ビジュアル上昇+25 & 体力回復2",
    },
    {
        "item_id": 9139,
        "name": "憧れ続けた輝き",
        "aliases": ("憧れ続けた輝き",),
        "mode": "stage",
        "source_type": "pIdol",
        "rarity": "SSR",
        "description": "好調が8ターン以上の場合、好調効果のスキルカードを4回使用するごとに、絶好調1ターン・使用数+1・1ドロー・固定体力消費1",
    },
    {
        "item_id": 9140,
        "name": "包まれるやさしさ",
        "aliases": ("包まれるやさしさ",),
        "mode": "stage",
        "source_type": "support",
        "rarity": "SSR",
        "description": "【ビジュアルターンのみ】ターン開始時、好調が10ターン以上の場合、絶好調4ターン・元気+10（レッスン内1回）",
    },
    {
        "item_id": 9141,
        "name": "ひとりだちシューズ",
        "aliases": ("ひとりだちシューズ",),
        "mode": "stage",
        "source_type": "support",
        "rarity": "SSR",
        "description": "【ダンズレッスンのみ】ターン時、体力が50%以下の場合",
    },
    {
        "item_id": 9142,
        "name": "ほっこりまんぶく",
        "aliases": ("ほっこりまんぶく",),
        "mode": "produce",
        "source_type": "support",
        "rarity": "SSR",
        "description": "好調効果のスキルカード獲得時、ランダムなPドリンクを獲得",
    },
    {
        "item_id": 9136,
        "name": "夢を創るステッキ",
        "aliases": ("夢を創るステッキ",),
        "mode": "stage",
        "source_type": "pIdol",
        "rarity": "SSR",
        "p_idol_id": 136,
        "upgraded": False,
        "upgrade_pair_id": 91361,
        "description": "3ターンごとのターン終了時、元気の110%分パラメータ上昇・元気を0にする・次のターン、スキルカードを引く（レッスン内3回）",
    },
    {
        "item_id": 91361,
        "name": "夢を創るステッキ+",
        "aliases": ("夢を創るステッキ+",),
        "mode": "stage",
        "source_type": "pIdol",
        "rarity": "SSR",
        "p_idol_id": 136,
        "upgraded": True,
        "upgrade_pair_id": 9136,
        "description": "3ターンごとのターン終了時、元気の130%分パラメータ上昇・元気を0にする・次のターン、スキルカードを引く（レッスン内3回）",
    },
]


def _normalize_title_line(raw: str) -> str:
    txt = _clean_ocr_text(raw or "")
    txt = re.sub(r"^[\s・、。,.`'\-\d◆+]+", "", txt)
    # Only strip leading isolated ASCII noise while keeping H.I.F / +
    if txt.startswith(("H.", "HI", "SP")):
        return txt.strip()
    return txt.strip()


def match_p_item_by_text(title: str, body: str = "") -> Dict[str, Any]:
    """Resolve scanned P-item title & body OCR text to structured metadata."""
    clean_t = _normalize_title_line(title).replace("工", "I")
    norm_t = clean_t.replace("（", "(").replace("）", ")").replace("＋", "+")
    t_has_plus = norm_t.endswith("+")

    def _to_meta(item_def: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "item_id": item_def["item_id"],
            "name": clean_t or item_def["name"],
            "canonical_name": item_def["name"],
            "mode": item_def["mode"],
            "source_type": item_def["source_type"],
            "rarity": item_def["rarity"],
            "description": body or item_def["description"],
        }

    # Pass 1: exact alias match (preserves + vs unupgraded distinction e.g. 夢を創るステッキ vs 夢を創るステッキ+)
    for item_def in KNOWN_P_ITEM_DEFS:
        for alias in item_def["aliases"]:
            norm_a = alias.replace("（", "(").replace("）", ")").replace("＋", "+")
            if norm_a == norm_t:
                return _to_meta(item_def)

    # Pass 2: substring match preferring same trailing '+' status and longer aliases
    sub_candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for item_def in KNOWN_P_ITEM_DEFS:
        for alias in item_def["aliases"]:
            norm_a = alias.replace("（", "(").replace("）", ")").replace("＋", "+")
            if norm_a and (norm_a in norm_t or norm_t in norm_a):
                plus_match = 1 if (norm_a.endswith("+") == t_has_plus) else 0
                sub_candidates.append((plus_match, len(norm_a), item_def))
    if sub_candidates:
        sub_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return _to_meta(sub_candidates[0][2])

    best_def: Optional[Dict[str, Any]] = None
    best_ratio = 0.0
    for item_def in KNOWN_P_ITEM_DEFS:
        for alias in item_def["aliases"]:
            norm_a = alias.replace("（", "(").replace("）", ")").replace("＋", "+")
            ratio = difflib.SequenceMatcher(None, norm_a, norm_t).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_def = item_def

    if best_def is not None and best_ratio >= 0.72:
        return _to_meta(best_def)

    # Body-text fallback when title OCR is garbled near viewport boundaries
    body_rules = (
        (("直接効果" in body and "好印象" in body and "4回" in body), 509),
        (("2回使用" in body and "好印象強化" in body), 404),
        (("22枚" in body or "応援棒" in body or "補填" in body), 429),
        (("ドリンク" in body and "試験開始時" in body), 436),
        (("スター性+10" in body or ("スキルカード獲得時" in body and "スター性" in body)), 447),
        (("SPレッスン終了時" in body or "ボーカルSP" in body), 61),
        (("ロジック" in body and ("ボーカル" in body or "強化" in body)), 510),
        (("チェンジ時" in body or "体力回復2" in body), 487),
        (("絶好調1ターン" in body and "4回使用" in body), 9139),
        (("絶好調4ターン" in body and "元気+10" in body), 9140),
        (("ひとりだち" in body or ("50%以下" in body and "ダンス" in body)), 9141),
        (("好調効果のスキルカード獲得時" in body and "Pドリンク" in body), 9142),
        (("3ターンごと" in body and ("元気の130%" in body or ("元気を0" in body and "130%" in body))), 91361),
        (("3ターンごと" in body and ("元気を0" in body or "元気の110%" in body)), 9136),
    )
    for matched_cond, known_id in body_rules:
        if matched_cond:
            item_def = next(d for d in KNOWN_P_ITEM_DEFS if d["item_id"] == known_id)
            return {
                "item_id": item_def["item_id"],
                "name": item_def["name"],
                "canonical_name": item_def["name"],
                "mode": item_def["mode"],
                "source_type": item_def["source_type"],
                "rarity": item_def["rarity"],
                "description": body or item_def["description"],
            }

    is_stage = ("レッスン内" in body or "ターン" in body or "毎ターン" in body or "試験" in body) and (
        "プロデュース中" not in body
    )
    return {
        "item_id": abs(hash(clean_t)) % 90000 + 10000,
        "name": clean_t or "不明Pアイテム",
        "canonical_name": clean_t or "不明Pアイテム",
        "mode": "stage" if is_stage else "produce",
        "source_type": "support",
        "rarity": "SSR",
        "description": body,
    }


_FULLWIDTH_TRANS = str.maketrans(
    "０１２３４５６７８９＋－−％（）×：，．　",
    "0123456789+--%()x:,. ",
)

_TEXT_NUM_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("score", r"(?<!増加)パラメータ\+(\d+)"),
    ("genki", r"(?<!カード)元気\+(\d+)"),
    ("goodImpressionTurns", r"好印象\+(\d+)"),
    ("motivation", r"やる気\+(\d+)"),
    ("goodConditionTurns", r"(?<!絶)好調(\d+)ターン"),
    ("perfectConditionTurns", r"絶好調(\d+)ターン"),
    ("concentration", r"集中\+(\d+)"),
    ("halfCostTurns", r"消費体力減少(\d+)ターン"),
    ("fullPowerCharge", r"全力値\+(\d+)"),
    ("growth_score", r"パラメータ値増加\+(\d+)"),
    ("genki_per_card", r"元気増加量\+(\d+)"),
)

_TEXT_PCT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("scoreBy_goodImpressionTurns", r"好印象の(\d+)%分パラメータ"),
    ("scoreBy_motivation", r"やる気の(\d+)%分パラメータ"),
    ("scoreBy_genki", r"元気の(\d+)%分パラメータ"),
    ("scoreBy_goodConditionTurns", r"(?<!絶)好調の(\d+)%分パラメータ"),
    ("gi_effect_buff_pct", r"好印象強化\+(\d+)%"),
)

_TYPED_COST_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("stamina", r"体力消費(\d+)"),
    ("goodImpressionTurns", r"好印象消費(\d+)"),
    ("motivation", r"やる気消費(\d+)"),
    ("concentration", r"集中消費(\d+)"),
    ("goodConditionTurns", r"(?<!絶)好調消費(\d+)"),
    ("perfectConditionTurns", r"絶好調消費(\d+)"),
    ("fullPowerCharge", r"全力値消費(\d+)"),
)

_COND_MIN_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("genki", r"元気が(\d+)"),
    ("goodImpressionTurns", r"好印象が(\d+)"),
    ("goodConditionTurns", r"(?<!絶)好調が(\d+)"),
    ("concentration", r"集中が(\d+)"),
    ("preservationTimes", r"温存になった回数が(\d+)"),
)

_DIRECT_GAIN_FIELDS = frozenset({
    "genki",
    "goodImpressionTurns",
    "motivation",
    "goodConditionTurns",
    "perfectConditionTurns",
    "concentration",
    "halfCostTurns",
    "fullPowerCharge",
})

_SCALE_IDENT_KEYS = frozenset({
    "goodImpressionTurns",
    "motivation",
    "genki",
    "goodConditionTurns",
})


def _normalize_ocr_effect_text(text: str) -> str:
    """Normalize Japanese OCR card description text for reliable customization matching."""
    if not text:
        return ""
    s = _clean_ocr_text(str(text)).translate(_FULLWIDTH_TRANS)
    s = re.sub(r"([パラメータ元気好印象好調絶好調集中やる気全力値上昇増加回数強])十(\d+)", r"\1+\2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_text_customization_features(norm_text: str) -> Dict[str, Any]:
    """Extract comparable numeric, percentage, cost, condition, and clause features from card body text."""
    nums: Dict[str, List[int]] = {k: [] for k, _ in _TEXT_NUM_PATTERNS}
    for key, pat in _TEXT_NUM_PATTERNS:
        nums[key] = [int(m.group(1)) for m in re.finditer(pat, norm_text)]

    score_hits = 0
    for m in re.finditer(r"(?<!増加)パラメータ\+(\d+)(?:\([^)]*?(\d+)回[^)]*\))?", norm_text):
        score_hits += int(m.group(2) or 1)

    pcts: Dict[str, List[int]] = {}
    for key, pat in _TEXT_PCT_PATTERNS:
        pcts[key] = [int(m.group(1)) for m in re.finditer(pat, norm_text)]

    typed_costs: Dict[str, Optional[int]] = {}
    for key, pat in _TYPED_COST_PATTERNS:
        m = re.search(pat, norm_text)
        typed_costs[key] = int(m.group(1)) if m else None

    cond_mins: Dict[str, List[int]] = {}
    for key, pat in _COND_MIN_PATTERNS:
        cond_mins[key] = [int(m.group(1)) for m in re.finditer(pat, norm_text)]

    flags = {
        "has_fih": ("レッスン開始時手札に入る" in norm_text) or ("開始時手札に入る" in norm_text),
        "has_limit_1": bool(re.search(r"レッスン中\s*1\s*回", norm_text)),
        "has_extra_use": ("スキルカード使用数追加" in norm_text) or ("使用数追加+" in norm_text),
        "has_upgrade_hand": bool(re.search(r"手札.*強化|レッスン中強化", norm_text)),
        "has_preservation": "温存に変更" in norm_text,
        "has_strength": "強気に変更" in norm_text,
        "has_stance_lv2": ("強気2段階目" in norm_text) or ("温存2段階目" in norm_text),
        "has_hold": "保留に移動" in norm_text,
        "has_turn_add": "ターン追加" in norm_text,
        "has_top_of_deck": "山札の一番上に移動" in norm_text,
        "has_gc_mult_2": "好調効果を2倍" in norm_text,
        "has_conc_mult_3": "集中効果を3倍" in norm_text,
        "has_mot_mult_15": bool(re.search(r"やる気.*?1\.?5倍", norm_text)),
        "has_eff_text": bool(
            re.search(r"(?:パラメータ|元気|好印象|やる気|集中|好調|全力|変更|保留|除外)", norm_text)
        ),
    }
    return {
        "nums": nums,
        "score_hits": score_hits,
        "pcts": pcts,
        "typed_costs": typed_costs,
        "cond_mins": cond_mins,
        "flags": flags,
        "text_len": len(norm_text),
    }


def _extract_mul_scale(rhs: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Return `(ident, pct_int)` if `rhs` represents `<ident> * <number>` or `<number> * <ident>`."""
    if not isinstance(rhs, dict):
        return None
    if rhs.get("type") == "identifier":
        nm = str(rhs.get("name") or "")
        if nm in _SCALE_IDENT_KEYS or nm == "cardsUsed":
            return nm, 100
        return None
    if rhs.get("type") == "binary" and rhs.get("op") == "*":
        left = rhs.get("left") or {}
        right = rhs.get("right") or {}
        if left.get("type") == "identifier" and right.get("type") == "number":
            return str(left.get("name") or ""), int(round(float(right.get("value") or 0.0) * 100))
        if right.get("type") == "identifier" and left.get("type") == "number":
            return str(right.get("name") or ""), int(round(float(left.get("value") or 0.0) * 100))
    return None


def _extract_ast_customization_features(
    parsed: Dict[str, Any],
    *,
    force_initial_hand: bool,
    limit_count: Optional[int],
    dsl_actions: str,
    customization_levels: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Project `apply_card_customizations()` output into the same feature schema as OCR card text."""
    nums: Dict[str, List[int]] = {k: [] for k, _ in _TEXT_NUM_PATTERNS}
    score_hits = 0
    pcts: Dict[str, List[int]] = {k: [] for k, _ in _TEXT_PCT_PATTERNS}
    typed_costs: Dict[str, Optional[int]] = {k: None for k, _ in _TYPED_COST_PATTERNS}
    cond_mins: Dict[str, List[int]] = {k: [] for k, _ in _COND_MIN_PATTERNS}
    flags: Dict[str, Any] = {
        "has_fih": bool(force_initial_hand),
        "has_limit_1": (limit_count == 1),
        "has_extra_use": ("cardUsesRemaining+=" in str(dsl_actions or "")),
        "has_upgrade_hand": False,
        "has_preservation": False,
        "has_strength": False,
        "has_stance_lv2": any(int(cid) in (40, 41) and int(lv or 0) > 0 for cid, lv in (customization_levels or {}).items()),
        "has_hold": False,
        "has_turn_add": False,
        "has_top_of_deck": False,
        "has_gc_mult_2": False,
        "has_conc_mult_3": False,
        "has_mot_mult_15": False,
    }

    cost_var = "cost"
    cost_amt = 0
    for cblock in parsed.get("cost") or ():
        if not isinstance(cblock, dict):
            continue
        for act in cblock.get("actions") or ():
            if not isinstance(act, dict) or act.get("type") != "assignment" or act.get("op") != "-=":
                continue
            rhs = act.get("rhs")
            if isinstance(rhs, dict) and rhs.get("type") == "number":
                lhs = str(act.get("lhs") or "cost")
                val = int(round(float(rhs.get("value") or 0)))
                cost_var = lhs
                cost_amt = max(0, val)
                if lhs in typed_costs:
                    typed_costs[lhs] = cost_amt

    def _walk_conditions(node: Any) -> None:
        if isinstance(node, list):
            for it in node:
                _walk_conditions(it)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "comparison" and node.get("op") in (">=", ">"):
            left = node.get("left") or {}
            right = node.get("right") or {}
            if left.get("type") == "identifier" and right.get("type") == "number":
                nm = str(left.get("name") or "")
                if nm in cond_mins:
                    cond_mins[nm].append(int(round(float(right.get("value") or 0))))
        for sub_k in ("conditions", "left", "right", "operand"):
            if sub_k in node:
                _walk_conditions(node.get(sub_k))

    def _walk_actions(node: Any) -> None:
        nonlocal score_hits
        if isinstance(node, list):
            for it in node:
                _walk_actions(it)
            return
        if not isinstance(node, dict):
            return
        if "conditions" in node:
            _walk_conditions(node.get("conditions"))
        ntype = node.get("type")
        if ntype == "assignment":
            lhs = str(node.get("lhs") or "")
            op = str(node.get("op") or "")
            rhs = node.get("rhs") or {}
            if op == "+=":
                if lhs == "score":
                    if rhs.get("type") == "number":
                        nums["score"].append(int(round(float(rhs.get("value") or 0))))
                        score_hits += 1
                    else:
                        sc = _extract_mul_scale(rhs)
                        if sc and sc[0] in _SCALE_IDENT_KEYS:
                            pcts[f"scoreBy_{sc[0]}"].append(sc[1])
                elif lhs in _DIRECT_GAIN_FIELDS:
                    if rhs.get("type") == "number":
                        iv = int(round(float(rhs.get("value") or 0)))
                        if iv > 0:
                            nums[lhs].append(iv)
                    elif lhs == "genki" and rhs.get("type") == "binary":
                        # Handle `12*cardsUsed+2` style scaling (#102 おっきなおにぎり+)
                        for side in (rhs.get("left"), rhs.get("right")):
                            sc = _extract_mul_scale(side) if isinstance(side, dict) else None
                            if sc and sc[0] == "cardsUsed":
                                nums["genki_per_card"].append(max(1, sc[1] // 100))
                elif lhs == "g.score" and rhs.get("type") == "number":
                    nums["growth_score"].append(int(round(float(rhs.get("value") or 0))))
                elif lhs == "cardUsesRemaining" and rhs.get("type") == "number" and float(rhs.get("value") or 0) > 0:
                    flags["has_extra_use"] = True
                elif lhs == "turnsRemaining" and rhs.get("type") == "number" and float(rhs.get("value") or 0) > 0:
                    flags["has_turn_add"] = True
            elif op == "=" and rhs.get("type") == "number":
                fval = float(rhs.get("value") or 0.0)
                if lhs == "goodConditionTurnsMultiplier" and int(round(fval)) == 2:
                    flags["has_gc_mult_2"] = True
                elif lhs == "concentrationMultiplier" and int(round(fval)) == 3:
                    flags["has_conc_mult_3"] = True
                elif lhs == "motivationMultiplier" and abs(fval - 1.5) < 1e-6:
                    flags["has_mot_mult_15"] = True
        elif ntype == "call":
            cname = str(node.get("name") or "")
            args = node.get("args") or []
            if cname == "upgradeHand":
                flags["has_upgrade_hand"] = True
            elif cname in ("holdSelected", "holdRandom"):
                flags["has_hold"] = True
            elif cname in ("moveRandomToTopOfDeck", "moveSelectedToTopOfDeck"):
                flags["has_top_of_deck"] = True
            elif cname == "setStance" and args:
                st = str((args[0] or {}).get("name") or "")
                if st == "preservation":
                    flags["has_preservation"] = True
                elif st == "strength":
                    flags["has_strength"] = True
            elif cname == "setGoodImpressionTurnsEffectBuff" and args:
                a0 = args[0] or {}
                if a0.get("type") == "number":
                    pcts["gi_effect_buff_pct"].append(int(round(float(a0.get("value") or 0.0) * 100)))

        for sub_k in ("actions", "effects"):
            if sub_k in node:
                _walk_actions(node.get(sub_k))

    _walk_conditions(parsed.get("conditions") or ())
    _walk_actions(parsed.get("actions") or ())
    _walk_actions(parsed.get("effects") or ())

    return {
        "nums": nums,
        "score_hits": score_hits,
        "pcts": pcts,
        "typed_costs": typed_costs,
        "cond_mins": cond_mins,
        "flags": flags,
        "cost_var": cost_var,
        "cost_amt": cost_amt,
    }


def _score_customization_candidate(
    *,
    combo: Dict[int, int],
    cand: Dict[str, Any],
    base: Dict[str, Any],
    obs: Dict[str, Any],
    observed_cost: Optional[int],
    valid_candidate_costs: set[int],
    badge_detected: bool,
    badge_count: Optional[int],
    has_customization_marker: bool,
) -> Optional[float]:
    """Score how well a candidate customization combo explains OCR text & icon visuals vs baseline."""
    has_visual_evidence = bool(badge_detected or has_customization_marker)
    total_lv = sum(combo.values())
    if total_lv > 3:
        return None
    if total_lv == 0:
        return 0.0

    # Without an icon badge / customization marker, disallow silent/memory-ambiguous flags (#29/#36/#37/#38)
    if not has_visual_evidence and any(cid in (29, 36, 37, 38) for cid in combo):
        return None

    score = 0.0
    strong_text_evidence = 0

    # 1. Flat numeric fields & score hit count (#4 パラメータ上昇回数+ etc.)
    for key in cand["nums"]:
        b_vals = base["nums"][key]
        c_vals = cand["nums"][key]
        if Counter(c_vals) == Counter(b_vals):
            continue
        o_vals = obs["nums"][key]
        if not o_vals and not b_vals and len(c_vals) == 1:
            # New numeric line added by customization, missing from OCR
            score -= 3.5
            continue
        b_ctr = Counter(b_vals)
        c_ctr = Counter(c_vals)
        o_ctr = Counter(o_vals)
        added_vals = list((c_ctr - b_ctr).elements())
        removed_vals = list((b_ctr - c_ctr).elements())
        if Counter(o_vals) == c_ctr:
            score += 5.0
            strong_text_evidence += 1
        elif added_vals and all(o_ctr[v] >= c_ctr[v] for v in set(added_vals)):
            score += 4.0 + (1.0 if all(o_ctr[v] < b_ctr[v] for v in set(removed_vals)) else 0.0)
            strong_text_evidence += 1
        elif c_vals and o_vals and max(c_vals) in o_vals and max(c_vals) not in b_vals:
            # Handle #4 + #64 where AST mutates first copy [12, 9] while game renders `パラメータ+12(2回)`
            score += 4.5
            strong_text_evidence += 1
        elif removed_vals and all(o_ctr[v] >= b_ctr[v] for v in set(removed_vals)):
            score -= 4.5
        else:
            score -= 2.5

    if cand["score_hits"] != base["score_hits"] and obs["score"]:
        if obs["score_hits"] >= cand["score_hits"]:
            score += 4.0
            strong_text_evidence += 1
        elif obs["score_hits"] <= base["score_hits"]:
            score -= 4.5

    # 2. Percentage scaling fields
    for key in cand["pcts"]:
        b_vals = base["pcts"][key]
        c_vals = cand["pcts"][key]
        if Counter(c_vals) == Counter(b_vals):
            continue
        o_vals = obs["pcts"][key]
        b_ctr, c_ctr, o_ctr = Counter(b_vals), Counter(c_vals), Counter(o_vals)
        added_vals = list((c_ctr - b_ctr).elements())
        if added_vals and all(o_ctr[v] >= c_ctr[v] for v in set(added_vals)):
            score += 4.5 + (0.8 if all(v not in o_vals for v in (b_ctr - c_ctr)) else 0.0)
            strong_text_evidence += 1
        else:
            score -= 4.5

    # 3. Typed cost reductions (#30..#35, #104)
    for t_key in cand["typed_costs"]:
        b_cost = base["typed_costs"][t_key]
        c_cost = cand["typed_costs"][t_key]
        if b_cost == c_cost:
            continue
        o_cost = obs["typed_costs"][t_key]
        if c_cost == 0:
            if o_cost == 0 or (o_cost is None and obs["text_len"] >= 12):
                score += 4.5
                strong_text_evidence += 1
            elif o_cost == b_cost:
                score -= 4.5
        else:
            if o_cost == c_cost:
                score += 4.5
                strong_text_evidence += 1
            elif o_cost == b_cost:
                score -= 4.5

    # 4. Condition minimum threshold changes (#85, #87, #88, #97, #100, #101, #105)
    for k in cand["cond_mins"]:
        b_vals = base["cond_mins"][k]
        c_vals = cand["cond_mins"][k]
        if Counter(c_vals) == Counter(b_vals):
            continue
        o_vals = obs["cond_mins"][k]
        added_vals = list((Counter(c_vals) - Counter(b_vals)).elements())
        if added_vals and all(v in o_vals for v in added_vals):
            score += 4.5
            strong_text_evidence += 1
        elif b_vals and any(v in o_vals for v in b_vals):
            score -= 4.5

    # 5. Added/changed clause flags
    clause_flags = (
        "has_upgrade_hand",
        "has_preservation",
        "has_strength",
        "has_stance_lv2",
        "has_hold",
        "has_turn_add",
        "has_top_of_deck",
        "has_gc_mult_2",
        "has_conc_mult_3",
        "has_mot_mult_15",
    )
    for fk in clause_flags:
        if cand["flags"][fk] and not base["flags"][fk]:
            if obs["flags"][fk]:
                score += 4.2
                strong_text_evidence += 1
            else:
                score -= 4.5

    # 6. Memory-sensitive flags (#36 force_initial_hand, #37 limit remove, #38 cardUsesRemaining+=1)
    if cand["flags"]["has_fih"] and not base["flags"]["has_fih"]:
        if obs["flags"]["has_fih"]:
            score += 3.0
        else:
            score -= 5.0
    if cand["flags"]["has_extra_use"] and not base["flags"]["has_extra_use"]:
        if obs["flags"]["has_extra_use"]:
            score += 3.0
        else:
            score -= 5.0
    if base["flags"]["has_limit_1"] and not cand["flags"]["has_limit_1"]:
        has_substantive_body = obs["flags"]["has_eff_text"] and obs["text_len"] >= 18
        if has_substantive_body and not obs["flags"]["has_limit_1"]:
            score += 3.0
        else:
            score -= 5.0

    # 7. Standard cost reduction (#29 g.cost+=2)
    if 29 in combo:
        if observed_cost is not None and observed_cost in valid_candidate_costs:
            if observed_cost == cand["cost_amt"]:
                score += 4.2
            else:
                score -= 4.5
        elif observed_cost is not None and base["cost_amt"] != cand["cost_amt"]:
            pass

    if not has_visual_evidence:
        if strong_text_evidence == 0 or score < 3.5:
            return None

    if has_visual_evidence and badge_count in (1, 2, 3):
        if total_lv == badge_count:
            score += 2.5
        elif total_lv > badge_count:
            score -= 3.5
        else:
            score -= 1.5

    return score


def infer_card_customizations_from_scan(
    db: GakumasDB,
    card_row: Dict[str, Any],
    body_txt: str,
    *,
    badge_detected: bool = False,
    badge_count: Optional[int] = None,
    observed_cost: Optional[int] = None,
    has_customization_marker: bool = False,
) -> Dict[str, Any]:
    """Infer specific card customization IDs and levels from left-icon badge + OCR effect text."""
    raw_avail = card_row.get("available_customization_ids_json")
    try:
        avail_ids = [int(x) for x in json.loads(raw_avail or "[]") if int(x) > 0]
    except Exception:
        avail_ids = []
    if not avail_ids:
        return {}

    cust_rows: Dict[int, Dict[str, Any]] = {}
    ranges: List[List[int]] = []
    for cid in avail_ids:
        crow = db.get_customization_by_id(cid)
        if crow is None:
            continue
        cust_rows[cid] = crow
        max_lv = max(1, min(3, int(crow.get("max_level") or 1)))
        ranges.append(list(range(max_lv + 1)))
    if not ranges:
        return {}

    from demo.sim.battle_simulator import apply_card_customizations

    raw_parsed = json.loads(card_row.get("parsed_effects_json") or "{}")
    base_fih = bool(card_row.get("force_initial_hand"))
    base_lim = int(card_row["limit_count"]) if card_row.get("limit_count") is not None else None
    base_dsl = str(card_row.get("dsl_actions") or "")

    base_feat = _extract_ast_customization_features(
        raw_parsed,
        force_initial_hand=base_fih,
        limit_count=base_lim,
        dsl_actions=base_dsl,
        customization_levels={},
    )
    norm_text = _normalize_ocr_effect_text(body_txt)
    obs_feat = _extract_text_customization_features(norm_text)

    ordered_cids = [cid for cid in avail_ids if cid in cust_rows]
    candidates: List[Tuple[Dict[int, int], Dict[str, Any]]] = []
    valid_costs: set[int] = set()
    for lvs in product(*ranges):
        if sum(lvs) > 3:
            continue
        combo = {ordered_cids[i]: lvs[i] for i in range(len(ordered_cids)) if lvs[i] > 0}
        if not combo:
            c_feat = base_feat
        else:
            p_cand, c_fih, c_lim, c_dsl = apply_card_customizations(
                raw_parsed,
                force_initial_hand=base_fih,
                limit_count=base_lim,
                dsl_actions=base_dsl,
                db=db,
                customization_levels=combo,
            )
            c_feat = _extract_ast_customization_features(
                p_cand,
                force_initial_hand=c_fih,
                limit_count=c_lim,
                dsl_actions=c_dsl,
                customization_levels=combo,
            )
        if c_feat["cost_var"] == "cost":
            valid_costs.add(int(c_feat["cost_amt"]))
        candidates.append((combo, c_feat))

    has_visual_evidence = bool(badge_detected or has_customization_marker)
    best_combo: Dict[int, int] = {}
    best_key: Tuple[float, int, int] = (0.0, 0, 0)
    for combo, c_feat in candidates:
        if not combo:
            continue
        sc = _score_customization_candidate(
            combo=combo,
            cand=c_feat,
            base=base_feat,
            obs=obs_feat,
            observed_cost=observed_cost,
            valid_candidate_costs=valid_costs,
            badge_detected=badge_detected,
            badge_count=badge_count,
            has_customization_marker=has_customization_marker,
        )
        if sc is None:
            continue
        if sc < (-1.0 if has_visual_evidence else 3.5):
            continue
        if not has_visual_evidence and sc <= 0.0:
            continue
        # Prefer highest match score; break ties by preferring fewer total levels (parsimony), then deterministic IDs
        key = (round(sc, 4), -sum(combo.values()), -sum(combo.keys()))
        if key > best_key or not best_combo:
            best_key = key
            best_combo = combo

    if not best_combo:
        return {}
    return {"levels": {str(k): int(v) for k, v in sorted(best_combo.items())}}


@dataclass
class LoadoutScanResult:
    """Complete scanned initial loadout extracted from the character/memory detail screen."""

    p_idol_id: Optional[int] = None
    idol_name: str = ""
    title: str = ""
    plan: str = "logic"
    recommended_effect: Optional[str] = None
    idol_stats: Dict[str, int] = field(default_factory=lambda: {"vocal": 180, "dance": 160, "visual": 150})
    deck_card_ids: List[int] = field(default_factory=list)
    card_names: List[str] = field(default_factory=list)
    p_items: List[Dict[str, Any]] = field(default_factory=list)
    card_memory_hooks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    card_customizations: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    screenshots_scanned: int = 0


class HifLoadoutScanner:
    """Scans the long-press / 詳細 loadout modal for P-idol, P-items, and full initial deck."""

    def __init__(self, db: Optional[GakumasDB] = None, plan: str = "logic"):
        self._owns_db = db is None
        self.db = db or GakumasDB("gakumas_library.db")
        self.ocr = get_win_ocr("ja")
        self.reco = LiveScreenRecognizer(db=self.db, plan=(plan or "logic").lower(), scenario_id="hif_honsen")

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    @staticmethod
    def _detect_back_button_xy(frame: np.ndarray) -> Tuple[int, int]:
        """Return bottom-left '<<' back button coordinates (handling 3-icon vs 1-icon pills)."""
        return detect_modal_back_button_xy(frame)

    @staticmethod
    def _estimate_scroll_delta(prev_vp: np.ndarray, cur_vp: np.ndarray) -> Tuple[int, float]:
        """Estimate vertical pixel scroll delta between adjacent list viewport captures."""
        h, w = prev_vp.shape[:2]
        x1, x2 = int(w * 0.05), int(w * 0.90)
        prev_gray = cv2.cvtColor(prev_vp[:, x1:x2], cv2.COLOR_RGB2GRAY)
        cur_gray = cv2.cvtColor(cur_vp[:, x1:x2], cv2.COLOR_RGB2GRAY)

        best_corr = -1.0
        best_dy = 0

        # Forward candidates: lower strips from prev_vp located inside cur_vp
        for f1, f2 in ((0.68, 0.85), (0.50, 0.68), (0.34, 0.52)):
            t_y1, t_y2 = int(h * f1), int(h * f2)
            if t_y2 - t_y1 < 64:
                continue
            tmpl = prev_gray[t_y1:t_y2, :]
            if float(tmpl.std()) < 4.0 or cur_gray.shape[0] <= tmpl.shape[0]:
                continue
            res = cv2.matchTemplate(cur_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            cand_dy = t_y1 - int(max_loc[1])
            if cand_dy >= -8 and float(max_val) > best_corr:
                best_corr = float(max_val)
                best_dy = cand_dy

        # Reverse candidate: top strip of cur_vp located inside prev_vp (handles large downward scrolls)
        for f1, f2 in ((0.03, 0.21), (0.16, 0.34)):
            t_y1, t_y2 = int(h * f1), int(h * f2)
            if t_y2 - t_y1 < 64:
                continue
            tmpl = cur_gray[t_y1:t_y2, :]
            if float(tmpl.std()) < 4.0 or prev_gray.shape[0] <= tmpl.shape[0]:
                continue
            res = cv2.matchTemplate(prev_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            cand_dy = int(max_loc[1]) - t_y1
            if cand_dy >= -8 and float(max_val) > best_corr:
                best_corr = float(max_val)
                best_dy = cand_dy

        if best_corr < 0.0:
            return 0, 0.0
        return best_dy, best_corr

    @staticmethod
    def _find_left_icon_spans(vp: np.ndarray, w: int) -> List[Tuple[int, int]]:
        """Segment card / P-item thumbnails along the left margin of a detail list viewport."""
        icon_band = vp[:, int(w * 0.068) : int(w * 0.245)]
        if icon_band.size == 0:
            return []
        row_nw = (icon_band.min(axis=2) < 232).mean(axis=1)
        non_white_rows = row_nw > 0.25
        raw_spans: List[Tuple[int, int]] = []
        r = 0
        n = len(non_white_rows)
        while r < n:
            if non_white_rows[r]:
                s = r
                while r < n and (
                    non_white_rows[r] or any(non_white_rows[r : min(n, r + 10)])
                ):
                    r += 1
                if r - s >= 35:
                    raw_spans.append((s, r))
            r += 1
        max_icon_h = max(135, int(vp.shape[0] * 0.145))
        spans: List[Tuple[int, int]] = []
        stack = list(raw_spans)
        while stack:
            s, r = stack.pop(0)
            if (r - s) > max_icon_h:
                mid_lo = s + int((r - s) * 0.30)
                mid_hi = s + int((r - s) * 0.70)
                if mid_hi > mid_lo:
                    cut = mid_lo + int(np.argmin(row_nw[mid_lo : mid_hi + 1]))
                    if cut - s >= 35 and r - cut >= 35:
                        stack.insert(0, (cut + 1, r))
                        stack.insert(0, (s, cut))
                        continue
            spans.append((s, r))
        return spans

    def _ocr_stat_int(self, crop: np.ndarray) -> Optional[int]:
        """Read a single 2-4 digit stat number from an upscaled crop using English/digit OCR."""
        if crop.size == 0:
            return None
        up = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        res = self.reco.en_ocr.recognize_array(up)
        txt = _clean_ocr_text(res.get("text", ""))
        nums = [int(x) for x in re.findall(r"\d{2,4}", txt) if 50 <= int(x) <= 3200]
        if nums:
            return nums[-1]
        res_ja = self.ocr.recognize_array(up)
        txt_ja = _clean_ocr_text(res_ja.get("text", ""))
        nums_ja = [int(x) for x in re.findall(r"\d{2,4}", txt_ja) if 50 <= int(x) <= 3200]
        return nums_ja[-1] if nums_ja else None

    def _detect_icon_customization_badge(self, icon_patch: np.ndarray) -> Tuple[bool, Optional[int]]:
        """Detect bright-green circular customization badge (1)/(2)/(3) on a detail-list card icon."""
        if icon_patch is None or icon_patch.size == 0:
            return False, 0
        ih, iw = icon_patch.shape[:2]
        if ih < 30 or iw < 30:
            return False, 0
        roi = icon_patch[int(ih * 0.50) : int(ih * 0.92), int(iw * 0.28) : int(iw * 0.72)]
        if roi.size == 0:
            return False, 0
        r = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        b = roi[:, :, 2].astype(np.int16)
        gm = (g >= 115) & ((g - r) >= 55) & ((g - b) >= 40) & (r < 105) & (b < 115)
        roi_area = max(1, roi.shape[0] * roi.shape[1])
        min_px = max(20, int(roi_area * 0.015))
        if int(gm.sum()) < min_px:
            return False, 0

        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(gm.astype(np.uint8), connectivity=8)
        best_bbox: Optional[Tuple[int, int, int, int, int]] = None
        rh, rw = roi.shape[:2]
        for lbl in range(1, num_labels):
            bx, by, bw, bh, area = [int(v) for v in stats[lbl]]
            if area < min_px or bh <= 0 or bw <= 0:
                continue
            if bw > int(rw * 0.88) or bh > int(rh * 0.92):
                continue
            aspect = bw / float(bh)
            fill_ratio = area / float(bw * bh)
            if 0.50 <= aspect <= 1.85 and fill_ratio >= 0.28:
                if best_bbox is None or area > best_bbox[4]:
                    best_bbox = (bx, by, bw, bh, area)
        if best_bbox is None:
            return False, 0

        bx, by, bw, bh, _ = best_bbox
        sub = roi[by : by + bh, bx : bx + bw]
        en_ocr = getattr(self.reco, "en_ocr", None)
        if sub.size >= 9 and en_ocr is not None:
            wm = (sub[:, :, 0] > 180) & (sub[:, :, 1] > 180) & (sub[:, :, 2] > 180)
            ys_w, xs_w = np.where(wm)
            if len(xs_w) > 0:
                dsub = wm[ys_w.min() : ys_w.max() + 1, xs_w.min() : xs_w.max() + 1].astype(np.uint8) * 255
                up = cv2.resize(dsub, (max(1, dsub.shape[1] * 4), max(1, dsub.shape[0] * 4)), interpolation=cv2.INTER_CUBIC)
                _, up = cv2.threshold(up, 115, 255, cv2.THRESH_BINARY)
                inv = 255 - up
                dh, dw = inv.shape[:2]
                pad_y = 24
                fscale = max(0.75, dh / 24.0)
                (tw_p, _), _ = cv2.getTextSize("10+", cv2.FONT_HERSHEY_SIMPLEX, fscale, 4)
                (sw_p, _), _ = cv2.getTextSize("+10", cv2.FONT_HERSHEY_SIMPLEX, fscale, 4)
                x0 = 20
                dx = x0 + tw_p + 10
                sx = dx + dw + 10
                canvas = np.full((dh + pad_y * 2, sx + sw_p + 24, 3), 255, dtype=np.uint8)
                cv2.putText(canvas, "10+", (x0, pad_y + dh - 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), 4, cv2.LINE_AA)
                canvas[pad_y : pad_y + dh, dx : dx + dw] = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
                cv2.putText(canvas, "+10", (sx, pad_y + dh - 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), 4, cv2.LINE_AA)
                txt = str(en_ocr.recognize_array(canvas).get("text", "") or "")
                m_c = re.search(r"10\+\s*([1-3])\s*\+10", txt)
                if m_c:
                    return True, int(m_c.group(1))
        return True, None

    def _ocr_icon_cost_badge(self, icon_patch: np.ndarray) -> Optional[int]:
        """OCR the top-left Genki/Stamina cost integer from a detail-list card icon."""
        if icon_patch is None or icon_patch.size == 0:
            return None
        en_ocr = getattr(self.reco, "en_ocr", None)
        if en_ocr is None:
            return None
        ih, iw = icon_patch.shape[:2]
        if ih < 24 or iw < 24:
            return None
        cost_roi = icon_patch[0 : max(12, int(ih * 0.40)), 0 : max(12, int(iw * 0.40))]
        if cost_roi.size == 0:
            return None
        up = cv2.resize(cost_roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        txt = _clean_ocr_text(str(en_ocr.recognize_array(up).get("text", "") or ""))
        vals = [int(x) for x in re.findall(r"\b(\d{1,2})\b", txt) if 0 <= int(x) <= 15]
        if vals:
            return vals[0]
        wm = (cost_roi[:, :, 0] > 195) & (cost_roi[:, :, 1] > 195) & (cost_roi[:, :, 2] > 195)
        ys_w, xs_w = np.where(wm)
        if len(xs_w) > 0:
            dsub = wm[ys_w.min() : ys_w.max() + 1, xs_w.min() : xs_w.max() + 1].astype(np.uint8) * 255
            inv = 255 - cv2.resize(dsub, (max(1, dsub.shape[1] * 4), max(1, dsub.shape[0] * 4)), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((inv.shape[0] + 40, inv.shape[1] + 40, 3), 255, dtype=np.uint8)
            canvas[20 : 20 + inv.shape[0], 20 : 20 + inv.shape[1]] = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
            txt2 = _clean_ocr_text(str(en_ocr.recognize_array(canvas).get("text", "") or ""))
            vals2 = [int(x) for x in re.findall(r"\b(\d{1,2})\b", txt2) if 0 <= int(x) <= 15]
            if vals2:
                return vals2[0]
        return None

    def _parse_header_metadata(self, frame: np.ndarray, default_plan: str) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        hdr_crop = frame[int(h * 0.254) : int(h * 0.415), :]
        hdr_res = self.ocr.recognize_array(hdr_crop)
        hdr_lines = [_clean_ocr_text(l["text"]) for l in hdr_res.get("lines", []) if _clean_ocr_text(l["text"])]
        hdr_joined = " ".join(hdr_lines)
        hdr_compact = hdr_joined.replace(" ", "").replace("　", "")

        matched_pid: Optional[int] = None
        matched_title = ""
        matched_idol = ""
        matched_plan = default_plan.lower()
        matched_rec: Optional[str] = None
        best_score = (0, 0, 0)

        for pid_row in self.db.get_p_idols():
            t = _clean_ocr_text(str(pid_row.get("title") or ""))
            nm = _clean_ocr_text(str(pid_row.get("idol_name") or "")).replace(" ", "")
            has_title = bool(t and t in hdr_compact)
            has_idol = bool(nm and nm in hdr_compact)
            if not has_idol and pid_row.get("idol_name"):
                parts = [p for p in str(pid_row["idol_name"]).split() if len(p) >= 1]
                if parts and all(p in hdr_compact for p in parts):
                    has_idol = True
            plan_match = 1 if str(pid_row.get("plan") or "").lower() == default_plan.lower() else 0
            if has_title and has_idol:
                score = (3, plan_match, len(t) + len(nm))
            elif has_idol:
                score = (2, plan_match, len(nm))
            elif has_title:
                score = (1, plan_match, len(t))
            else:
                continue
            if score > best_score:
                best_score = score
                matched_pid = int(pid_row["id"])
                matched_title = str(pid_row.get("title") or "")
                matched_idol = str(pid_row.get("idol_name") or "")
                matched_plan = str(pid_row.get("plan") or matched_plan).lower()
                matched_rec = str(pid_row["recommended_effect"]) if pid_row.get("recommended_effect") else None

        # Parse Vo / Da / Vi stats near y=0.350..0.382 using column crops + upscaled digit OCR
        stats = {"vocal": 180, "dance": 160, "visual": 150}
        vo_v = self._ocr_stat_int(frame[int(h * 0.350) : int(h * 0.382), int(w * 0.29) : int(w * 0.46)])
        da_v = self._ocr_stat_int(frame[int(h * 0.350) : int(h * 0.382), int(w * 0.47) : int(w * 0.65)])
        vi_v = self._ocr_stat_int(frame[int(h * 0.350) : int(h * 0.382), int(w * 0.66) : int(w * 0.86)])
        if vo_v and da_v and vi_v:
            stats = {"vocal": min(3200, vo_v), "dance": min(3200, da_v), "visual": min(3200, vi_v)}
        else:
            band = frame[int(h * 0.350) : int(h * 0.388), int(w * 0.27) : int(w * 0.88)]
            band_up = cv2.resize(band, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            stat_band_txt = _clean_ocr_text(self.reco.en_ocr.recognize_array(band_up).get("text", ""))
            stat_nums = [int(x) for x in re.findall(r"\d{3,4}", stat_band_txt) if 150 <= int(x) <= 3200]
            if len(stat_nums) >= 3:
                stats = {
                    "vocal": min(3200, stat_nums[-3]),
                    "dance": min(3200, stat_nums[-2]),
                    "visual": min(3200, stat_nums[-1]),
                }

        return {
            "p_idol_id": matched_pid,
            "title": matched_title,
            "idol_name": matched_idol,
            "plan": matched_plan,
            "recommended_effect": matched_rec,
            "idol_stats": stats,
        }

    def parse_detail_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        plan: str = "logic",
    ) -> LoadoutScanResult:
        """Parse captured list-view frames of the 基本情報 detail modal into LoadoutScanResult."""
        if not frames:
            return LoadoutScanResult(plan=plan.lower())

        meta = self._parse_header_metadata(frames[0], default_plan=plan)
        active_plan = meta["plan"]

        cum_scroll = 0
        prev_vp: Optional[np.ndarray] = None
        current_section = "p_item"
        seen_entries: List[Tuple[int, str]] = []
        parsed_p_items: List[Dict[str, Any]] = []
        seen_p_names: set[str] = set()
        deck_card_ids: List[int] = []
        card_names: List[str] = []
        card_hooks: Dict[int, Dict[str, Any]] = {}
        card_customizations: Dict[int, Dict[str, Any]] = {}

        for f_idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            y_top, y_bot = int(h * 0.254), int(h * 0.858)
            vp = frame[y_top:y_bot, :]
            if prev_vp is not None:
                dy, corr = self._estimate_scroll_delta(prev_vp, vp)
                if corr >= 0.72:
                    if dy > 0:
                        cum_scroll += dy
                elif dy > 0 and corr >= 0.50:
                    cum_scroll += dy
                else:
                    # Only fall back to +520 when viewport genuinely changed (never duplicate bottom-of-list bounce frames)
                    diff_vp = float(np.mean(np.abs(vp.astype(np.float32) - prev_vp.astype(np.float32))))
                    if diff_vp > 12.0:
                        cum_scroll += 520
            prev_vp = vp

            res = self.ocr.recognize_array(vp)
            lines = sorted(res.get("lines", []), key=lambda ln: ln["bbox"][1])

            # Track visible section headers ('Pアイテム', 'スキルカード') strictly in left gutter (x < 0.16*w)
            sec_markers: List[Tuple[int, str]] = []
            for ln in lines:
                t = _clean_ocr_text(ln["text"])
                if ln["bbox"][0] < int(w * 0.16) and len(t) <= 12:
                    if t.startswith("Pアイテム"):
                        sec_markers.append((ln["bbox"][1], "p_item"))
                    elif t.startswith("スキルカード"):
                        sec_markers.append((ln["bbox"][1], "card"))
            sec_markers.sort(key=lambda pair: pair[0])

            first_sec_y = sec_markers[0][0] if sec_markers else None
            spans = self._find_left_icon_spans(vp, w)
            for sp_idx, (sy1, sy2) in enumerate(spans):
                # On page 0, ignore avatar/star icons above the first 'Pアイテム' section header
                if f_idx == 0 and first_sec_y is not None and sy1 < first_sec_y:
                    continue
                # Skip top-clipped sliver icons whose header scrolled above viewport, and bottom slivers
                if (f_idx > 0 and sy1 <= 6 and (sy2 - sy1) < 55) or sy1 < 0 or sy1 > vp.shape[0] - 32:
                    continue
                doc_y = cum_scroll + sy1
                if any(abs(doc_y - old_y) < 30 for old_y, _old_k in seen_entries):
                    continue

                if current_section == "p_item" and sy1 >= 45 and not any(m == "card" and my < sy1 for my, m in sec_markers):
                    hdr_strip = vp[max(0, sy1 - 90) : sy1, 0 : int(w * 0.26)]
                    if hdr_strip.size > 0:
                        strip_up = cv2.resize(hdr_strip, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        for sln in self.ocr.recognize_array(strip_up).get("lines", []):
                            st = _clean_ocr_text(sln["text"])
                            if st.startswith("スキルカード") and len(st) <= 12:
                                sec_markers.append((max(0, sy1 - 20), "card"))
                                sec_markers.sort(key=lambda pair: pair[0])
                                break

                item_sec = current_section
                for my, m_sec in sec_markers:
                    if my < sy1:
                        item_sec = m_sec

                hdr_lines = [
                    ln for ln in lines
                    if sy1 - 24 <= ln["bbox"][1] <= sy1 + 48 and ln["bbox"][0] >= int(w * 0.16)
                ]
                box_lines = [
                    ln for ln in lines
                    if sy1 - 24 <= ln["bbox"][1] <= sy2 + 135 and ln["bbox"][0] >= int(w * 0.16)
                ]
                raw_hdr = _normalize_title_line(hdr_lines[0]["text"]) if hdr_lines else ""
                body_txt = " | ".join(_clean_ocr_text(ln["text"]) for ln in box_lines)

                # Always run 2x upscaled tight title-band fallback when header is missing or lacks '+' on cards
                need_tight_title = (not raw_hdr) or (
                    item_sec != "p_item" and ("+" not in raw_hdr and "＋" not in raw_hdr and "十" not in raw_hdr[-2:])
                )
                if need_tight_title:
                    t_crop = vp[max(0, sy1 - 12) : min(vp.shape[0], sy1 + 46), int(w * 0.24) : int(w * 0.90)]
                    if t_crop.size > 0:
                        t_up = cv2.resize(t_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        t_lines = sorted(self.ocr.recognize_array(t_up).get("lines", []), key=lambda ln: ln["bbox"][1])
                        if t_lines:
                            cand_hdr = _normalize_title_line(t_lines[0]["text"])
                            if cand_hdr and (not raw_hdr or "+" in cand_hdr or "＋" in cand_hdr or len(cand_hdr) >= len(raw_hdr)):
                                raw_hdr = cand_hdr

                if not raw_hdr and not body_txt:
                    continue

                if item_sec == "p_item":
                    p_info = match_p_item_by_text(raw_hdr, body_txt)
                    pname_key = p_info["canonical_name"]
                    entry_key = f"p:{pname_key}"
                    if any(
                        (old_k == entry_key and abs(doc_y - old_y) < 115)
                        or (old_k != entry_key and abs(doc_y - old_y) < 42)
                        for old_y, old_k in seen_entries
                    ):
                        current_section = item_sec
                        continue
                    seen_entries.append((doc_y, entry_key))
                    if pname_key not in seen_p_names:
                        seen_p_names.add(pname_key)
                        parsed_p_items.append(p_info)
                else:
                    plus_patch = vp[
                        max(0, sy1 + 38) : min(vp.shape[0], sy1 + 88),
                        int(w * 0.218) : int(w * 0.250),
                    ]
                    has_red_plus = bool(
                        plus_patch.size > 0
                        and int(
                            (
                                (plus_patch[:, :, 0] > 215)
                                & (plus_patch[:, :, 1] < 135)
                                & (plus_patch[:, :, 2] < 85)
                            ).sum()
                        )
                        >= 20
                    )
                    is_up = "+" in raw_hdr or "＋" in raw_hdr or "十" in raw_hdr[-2:] or has_red_plus
                    matched_card = (
                        self.reco.match_card_by_name(raw_hdr, plan=active_plan, upgraded=is_up)
                        if raw_hdr
                        else None
                    )
                    if matched_card is None and raw_hdr:
                        matched_card = self.reco.match_card_by_name(raw_hdr, plan="sense", upgraded=is_up)
                    if matched_card is None and raw_hdr:
                        matched_card = self.reco.match_card_by_name(raw_hdr, plan="logic", upgraded=is_up)
                    # Retry tight-title crop if full-viewport OCR misread the card name completely
                    if matched_card is None and not need_tight_title:
                        t_crop = vp[max(0, sy1 - 12) : min(vp.shape[0], sy1 + 46), int(w * 0.24) : int(w * 0.90)]
                        if t_crop.size > 0:
                            t_up = cv2.resize(t_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                            t_lines = sorted(self.ocr.recognize_array(t_up).get("lines", []), key=lambda ln: ln["bbox"][1])
                            if t_lines:
                                retry_hdr = _normalize_title_line(t_lines[0]["text"])
                                if retry_hdr:
                                    retry_up = "+" in retry_hdr or "＋" in retry_hdr or "十" in retry_hdr[-2:] or has_red_plus
                                    matched_card = (
                                        self.reco.match_card_by_name(retry_hdr, plan=active_plan, upgraded=retry_up)
                                        or self.reco.match_card_by_name(retry_hdr, plan="sense", upgraded=retry_up)
                                        or self.reco.match_card_by_name(retry_hdr, plan="logic", upgraded=retry_up)
                                    )
                    if matched_card is None and body_txt:
                        if "ターン追加" in body_txt and ("好印象" in body_txt or "スキルカード使用数追加" in body_txt):
                            star_cid = (
                                171
                                if (is_up or "スキルカードを引く" in body_txt or "好印象消費1" in body_txt)
                                else 170
                            )
                            matched_card = self.db.get_card_by_id(star_cid)
                    if matched_card is not None:
                        cid = int(matched_card["id"])
                        entry_key = f"c:{cid}"
                        if any(
                            (old_k == entry_key and abs(doc_y - old_y) < 115)
                            or (old_k != entry_key and abs(doc_y - old_y) < 42)
                            for old_y, old_k in seen_entries
                        ):
                            current_section = item_sec
                            continue
                        seen_entries.append((doc_y, entry_key))
                        deck_card_ids.append(cid)
                        card_names.append(str(matched_card["name"]))
                        hooks = dict(card_hooks.get(cid, {}))
                        dsl_a = str(
                            matched_card.get("dsl_actions")
                            or ((self.db.get_card_by_id(cid) or {}).get("dsl_actions"))
                            or ""
                        )
                        if "レッスン開始時手札に入る" in body_txt:
                            hooks["force_initial_hand"] = True
                        if (
                            "removeRandom[trouble" not in dsl_a
                            and (
                                "トラブルカードを除外" in body_txt
                                or "トラブルカード1枚を除外" in body_txt
                                or "トラブルカードを1枚除外" in body_txt
                                or "眠気を除外" in body_txt
                                or (
                                    "トラブル" in body_txt
                                    and "除外" in body_txt
                                    and cid not in (426, 427)
                                    and "除外以外" not in body_txt
                                    and "除外にある" not in body_txt
                                )
                            )
                        ):
                            hooks["exiles_trouble_on_play"] = True
                        if "スキルカード使用数追加" in body_txt or "使用数追加" in body_txt:
                            if "cardUsesRemaining+=" not in dsl_a:
                                hooks["free_extra_use_first_play"] = True
                        if "カスタマイズ" in body_txt:
                            hooks["has_customization_marker"] = True

                        card_row = matched_card
                        if card_row.get("available_customization_ids_json") is None:
                            card_row = self.db.get_card_by_id(cid) or card_row
                        icon_patch = vp[max(0, sy1 - 4) : min(vp.shape[0], sy2 + 4), int(w * 0.060) : int(w * 0.252)]
                        badge_detected, badge_count = self._detect_icon_customization_badge(icon_patch)
                        if badge_detected:
                            hooks["has_customization_marker"] = True
                        next_sy1 = spans[sp_idx + 1][0] if sp_idx + 1 < len(spans) else (sy2 + 135)
                        effect_bot = min(sy2 + 135, next_sy1 - 8)
                        card_eff_lines = [
                            ln for ln in lines
                            if sy1 - 24 <= ln["bbox"][1] <= effect_bot and ln["bbox"][0] >= int(w * 0.16)
                        ]
                        card_eff_txt = " | ".join(_clean_ocr_text(ln["text"]) for ln in card_eff_lines) or body_txt

                        raw_avail_cust = card_row.get("available_customization_ids_json")
                        try:
                            avail_cust_ids = [int(x) for x in json.loads(raw_avail_cust or "[]") if int(x) > 0]
                        except Exception:
                            avail_cust_ids = []
                        obs_cost: Optional[int] = None
                        if 29 in avail_cust_ids and (badge_detected or hooks.get("has_customization_marker")):
                            obs_cost = self._ocr_icon_cost_badge(icon_patch)
                        inferred_cust = infer_card_customizations_from_scan(
                            self.db,
                            card_row,
                            card_eff_txt,
                            badge_detected=badge_detected,
                            badge_count=badge_count,
                            observed_cost=obs_cost,
                            has_customization_marker=bool(hooks.get("has_customization_marker")),
                        )
                        if inferred_cust:
                            prev_cust = card_customizations.get(cid)
                            prev_sum = sum(int(v or 0) for v in ((prev_cust or {}).get("levels") or {}).values())
                            new_sum = sum(int(v or 0) for v in (inferred_cust.get("levels") or {}).values())
                            if prev_cust is None or new_sum >= prev_sum:
                                card_customizations[cid] = inferred_cust
                        if hooks:
                            card_hooks[cid] = hooks

                current_section = item_sec

            if sec_markers:
                current_section = sec_markers[-1][1]

        return LoadoutScanResult(
            p_idol_id=meta["p_idol_id"],
            idol_name=meta["idol_name"],
            title=meta["title"],
            plan=active_plan,
            recommended_effect=meta["recommended_effect"],
            idol_stats=meta["idol_stats"],
            deck_card_ids=deck_card_ids,
            card_names=card_names,
            p_items=parsed_p_items,
            card_memory_hooks=card_hooks,
            card_customizations=card_customizations,
            screenshots_scanned=len(frames),
        )

    def scan_via_controller(
        self,
        controller: BaseController,
        *,
        open_xy: Tuple[int, int] = (240, 1185),
        open_via_long_press: bool = True,
        plan: str = "logic",
        max_scrolls: int = 40,
    ) -> LoadoutScanResult:
        """Open detail modal via long-press or button click, scan full list, and close modal cleanly."""
        list_markers = ("終了時", "増加時", "獲得時", "使用時", "直接効果", "好印象", "やる気", "好調", "集中", "全力")
        setup_screen_markers = (
            "アイドル選択",
            "開始確認",
            "プロデュース開始",
            "育成情報",
            "特訓段階",
            "獲得量上昇",
            "Pアイドル設定",
        )
        strict_modal_markers = (
            "基本情報",
            "アビリティ",
            "履歴",
            "所持Pアイテム",
            "所持スキルカード",
            "Pアイテム",
        )

        frame0: Optional[np.ndarray] = None
        top_txt = ""
        modal_opened = False
        for attempt in range(2):
            dur_ms = 950 if attempt == 0 else 1050
            if open_via_long_press:
                controller.long_press(open_xy[0], open_xy[1], duration_ms=dur_ms)
            else:
                controller.click(open_xy[0], open_xy[1])
            time.sleep(1.3 if attempt == 0 else 1.5)

            frame0 = controller.screencap()
            h, w = frame0.shape[:2]
            hdr_txt = _clean_ocr_text(self.ocr.recognize_array(frame0[int(h * 0.14) : int(h * 0.42), :]).get("text", ""))
            body_full_txt = _clean_ocr_text(
                self.ocr.recognize_array(frame0[int(h * 0.40) : int(h * 0.90), :]).get("text", "")
            )
            top_txt = _clean_ocr_text(self.ocr.recognize_array(frame0[int(h * 0.42) : int(h * 0.88), int(w * 0.24) :]).get("text", ""))
            nav_bar_txt = _clean_ocr_text(self.ocr.recognize_array(frame0[int(h * 0.83) : int(h * 0.92), :]).get("text", ""))
            combined = hdr_txt + " " + body_full_txt + " " + top_txt + " " + nav_bar_txt
            is_still_setup = any(sm in combined for sm in setup_screen_markers)
            has_modal_marker = any(tok in combined for tok in strict_modal_markers) or any(tok in combined for tok in list_markers)
            if (not is_still_setup) and has_modal_marker:
                modal_opened = True
                break
            if attempt == 0:
                time.sleep(0.8)

        if not modal_opened or frame0 is None:
            # Modal did not open — do NOT click '<<' back so setup stays on 1.アイドル選択 for safe retry
            return LoadoutScanResult(plan=plan.lower(), screenshots_scanned=0)

        h, w = frame0.shape[:2]

        # If currently in icon-grid view, click the list toggle icon at (648, 1312) and verify list markers appear
        if not any(tok in top_txt for tok in list_markers):
            for _tog_try in range(3):
                tx = int(round(w * (648.0 / 720.0)))
                ty = int(round(h * (1312.0 / 1600.0)))
                controller.click(tx, ty)
                time.sleep(1.0)
                frame0 = controller.screencap()
                h, w = frame0.shape[:2]
                top_txt = _clean_ocr_text(
                    self.ocr.recognize_array(frame0[int(h * 0.42) : int(h * 0.88), int(w * 0.24) :]).get("text", "")
                )
                if any(tok in top_txt for tok in list_markers):
                    break
            if not any(tok in top_txt for tok in list_markers):
                self._dismiss_detail_modal(controller, frame0)
                return LoadoutScanResult(plan=plan.lower(), screenshots_scanned=0)

        frames: List[np.ndarray] = []
        prev_vp: Optional[np.ndarray] = None
        stale_count = 0
        for s_idx in range(max_scrolls):
            cur = controller.screencap()
            frames.append(cur)
            y_top, y_bot = int(h * 0.254), int(h * 0.858)
            cur_vp = cur[y_top:y_bot, :]
            if prev_vp is not None:
                dy, corr = self._estimate_scroll_delta(prev_vp, cur_vp)
                if corr >= 0.94 and abs(dy) <= 8:
                    stale_count += 1
                    if stale_count >= 2:
                        break
                else:
                    stale_count = 0
            prev_vp = cur_vp
            if s_idx + 1 < max_scrolls:
                sx = int(round(w * 0.50))
                sy1 = int(round(h * 0.700))
                sy2 = int(round(h * 0.500))
                controller.swipe(sx, sy1, sx, sy2, duration_ms=950)
                time.sleep(0.45)

        # Return cleanly to the previous setup page via bottom-left '<<' button (auto-detecting 3-icon vs 1-icon pill)
        self._dismiss_detail_modal(controller, frames[-1] if frames else frame0)
        return self.parse_detail_frames(frames, plan=plan)

    def _dismiss_detail_modal(self, controller: BaseController, frame: np.ndarray) -> None:
        """Close the 基本情報 detail modal via bottom-left '<<' until navigation bar disappears."""
        back_x, back_y = self._detect_back_button_xy(frame)
        for _back_try in range(2):
            controller.click(back_x, back_y)
            time.sleep(1.2)
            after_fr = controller.screencap()
            ah = after_fr.shape[0]
            nav_txt = _clean_ocr_text(self.ocr.recognize_array(after_fr[int(ah * 0.84) : int(ah * 0.91), :]).get("text", ""))
            if "基本情報" not in nav_txt:
                break
            back_x, back_y = self._detect_back_button_xy(after_fr)
