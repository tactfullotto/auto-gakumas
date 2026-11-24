"""Long-press / Detail Modal Loadout Scanner for H.I.F 本戦 initial cards & P-items."""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
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
    best_def: Optional[Dict[str, Any]] = None
    best_ratio = 0.0
    for item_def in KNOWN_P_ITEM_DEFS:
        for alias in item_def["aliases"]:
            norm_a = alias.replace("（", "(").replace("）", ")")
            norm_t = clean_t.replace("（", "(").replace("）", ")")
            if norm_a in norm_t or norm_t in norm_a:
                return {
                    "item_id": item_def["item_id"],
                    "name": clean_t or item_def["name"],
                    "canonical_name": item_def["name"],
                    "mode": item_def["mode"],
                    "source_type": item_def["source_type"],
                    "rarity": item_def["rarity"],
                    "description": body or item_def["description"],
                }
            ratio = difflib.SequenceMatcher(None, norm_a, norm_t).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_def = item_def

    if best_def is not None and best_ratio >= 0.72:
        return {
            "item_id": best_def["item_id"],
            "name": clean_t or best_def["name"],
            "canonical_name": best_def["name"],
            "mode": best_def["mode"],
            "source_type": best_def["source_type"],
            "rarity": best_def["rarity"],
            "description": body or best_def["description"],
        }

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
            for sy1, sy2 in spans:
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
