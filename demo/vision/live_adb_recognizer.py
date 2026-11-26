"""
Live Real-Device Screen Recognizer for Gakumas Android / Desktop Screenshots.

Combines fast Windows WinRT Japanese OCR (`win_ocr.py`) with GakumasDB fuzzy
card/drink matching and OpenCV color/button localization so live 720x1600
(or any portrait aspect) frames from ADB produce accurate ScreenObservations.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from PIL import Image

from demo.core.db import GakumasDB
from demo.core.scenarios import compute_week_stats_and_stage_bonus, get_scenario_spec
from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation
from demo.vision.win_ocr import get_win_ocr

logger = logging.getLogger(__name__)


def _normalize_jp(text: str) -> str:
    return re.sub(r"[\s・･!！?？♡★☆「」『』()（）\[\]【】+＋\-ー―~〜,、.。]", "", text or "")


def _clean_ocr_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip()


def _normalize_katakana_ocr(text: str) -> str:
    return (
        (text or "")
        .replace("チェンツ", "チェンジ")
        .replace("チェンシ", "チェンジ")
        .replace("チエンジ", "チェンジ")
        .replace("チエンシ", "チェンジ")
        .replace("ェンジする", "チェンジする")
        .replace("エンジする", "チェンジする")
        .replace("ェンジで獲得", "チェンジで獲得")
        .replace("カスタマイス", "カスタマイズ")
        .replace("くたさい", "ください")
        .replace("くださぃ", "ください")
    )


def detect_modal_back_button_xy(frame: np.ndarray) -> Tuple[int, int]:
    """Locate the bottom-left '<<' button in detail modals (3-icon wide pill vs 1-icon pill)."""
    h, w = frame.shape[:2]
    if h >= 100 and w >= 100:
        bot = frame[int(h * 0.925) : int(h * 0.985), 0 : int(w * 0.56), :3]
        if bot.size > 0:
            r = bot[:, :, 0].astype(np.int16)
            g = bot[:, :, 1].astype(np.int16)
            b = bot[:, :, 2].astype(np.int16)
            gray_pill = (r > 105) & (r < 195) & (np.abs(r - g) < 18) & (np.abs(g - b) < 18)
            _by, bx = np.where(gray_pill)
            if len(bx) >= 200 and int(bx.max()) > int(w * 0.35):
                return (int(round(w * (302.0 / 720.0))), int(round(h * (1532.0 / 1600.0))))
    return (int(round(w * (88.0 / 720.0))), int(round(h * (1532.0 / 1600.0))))


class LiveScreenRecognizer:
    """Recognizes real Gakumas Android/Win32 frames using WinRT OCR + GakumasDB."""

    _LEFT_BUFF_TOAST_TOKENS: Tuple[str, ...] = (
        "好印象",
        "印象",
        "やる気",
        "好調",
        "絶好調",
        "集中",
        "元気",
        "消費",
        "増加",
        "減少",
        "無効",
        "低下",
        "全力",
        "温存",
        "強気",
    )

    def __init__(self, db: Optional[GakumasDB] = None, plan: str = "logic", scenario_id: str = "hajime_regular"):
        self.db = db or GakumasDB()
        self.plan = (plan or "logic").lower()
        self.scenario_id = scenario_id
        self.ocr = get_win_ocr("ja")
        self.en_ocr = get_win_ocr("en")
        self._cards_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._drinks_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._last_week: int = 1
        self._last_week_label: str = ""
        self._last_stats: Dict[str, int] = {"vocal": 130, "dance": 100, "visual": 105}
        self._last_stamina: Tuple[int, int] = (44, 44) if self.scenario_id in ("hif_honsen", "hif") else (23, 31)
        self._pending_stamina_reading: Optional[Tuple[int, int]] = None
        self._pending_stamina_streak: int = 0
        self._last_p_points: int = 0
        self._last_turns_remaining: int = 8
        self._turn_drop_candidate: Optional[Tuple[int, int]] = None
        self._in_battle_scene: bool = False
        self._non_battle_streak: int = 0
        self._shop_visited_slots: Set[str] = set()
        self._shop_last_clicked_slot: Optional[str] = None
        self._shop_card_icon_cache: Dict[str, List[Tuple[int, np.ndarray]]] = {}
        self._reward_visited_slots: Set[int] = set()
        self._drink_templates_cache: Optional[List[Tuple[int, str, np.ndarray, np.ndarray]]] = None
        self._last_buff_hud_occluded: bool = False
        self.banned_drink_ids: Set[int] = {19}
        self._banned_drink_names: Tuple[str, ...] = ("初星スペシャル青汁", "青汁")

    def set_banned_drink_ids(self, drink_ids: Optional[Sequence[int]]) -> None:
        """Configure banned P-drink IDs (triggers avoidance & auto-discard options)."""
        ids = {int(x) for x in (drink_ids or ())}
        self.banned_drink_ids = ids
        names: List[str] = []
        if 19 in ids:
            names.extend(["初星スペシャル青汁", "青汁"])
        for did in sorted(ids):
            try:
                row = self.db.get_drink_by_id(did)
                if row and row.get("name"):
                    nm = str(row["name"]).strip()
                    if nm and nm not in names:
                        names.append(nm)
            except Exception:
                pass
        self._banned_drink_names = tuple(names)

    @classmethod
    def _has_left_buff_toast_in_lines(
        cls,
        lines: List[Dict[str, Any]],
        w: int,
        h: int,
    ) -> bool:
        """Detect whether a floating left-HUD buff toast (e.g. 好印象/やる気) is covering stack numbers."""
        for ln in lines or ():
            bbox = ln.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            if int(bbox[0]) < int(w * 0.44) and int(h * 0.09) <= int(bbox[1]) <= int(h * 0.32):
                txt = str(ln.get("text") or "")
                if any(tok in txt for tok in cls._LEFT_BUFF_TOAST_TOKENS):
                    return True
        return False

    def export_state(self) -> Dict[str, Any]:
        """Serialize sticky HUD fallback state for process interrupt recovery."""
        return {
            "last_week": int(self._last_week),
            "last_week_label": str(self._last_week_label),
            "last_stats": {str(k): int(v) for k, v in self._last_stats.items()},
            "last_stamina": [int(self._last_stamina[0]), int(self._last_stamina[1])],
            "last_p_points": int(self._last_p_points),
            "last_turns_remaining": int(self._last_turns_remaining),
            "in_battle_scene": bool(self._in_battle_scene),
            "shop_visited_slots": sorted(self._shop_visited_slots),
            "shop_last_clicked_slot": self._shop_last_clicked_slot,
            "reward_visited_slots": sorted(self._reward_visited_slots),
        }

    def load_state(self, data: Dict[str, Any]) -> None:
        """Restore sticky HUD fallback state from a saved checkpoint."""
        if not isinstance(data, dict):
            return
        if "last_week" in data and data["last_week"] is not None:
            max_w = max(13, get_scenario_spec(self.scenario_id).total_weeks)
            self._last_week = max(1, min(max_w, int(data["last_week"])))
        if "last_week_label" in data and data["last_week_label"] is not None:
            self._last_week_label = str(data["last_week_label"])
        if isinstance(data.get("last_stats"), dict):
            for k in ("vocal", "dance", "visual"):
                if k in data["last_stats"] and data["last_stats"][k] is not None:
                    self._last_stats[k] = int(data["last_stats"][k])
        if isinstance(data.get("last_stamina"), (list, tuple)) and len(data["last_stamina"]) >= 2:
            self._last_stamina = (int(data["last_stamina"][0]), int(data["last_stamina"][1]))
            self._pending_stamina_reading = None
            self._pending_stamina_streak = 0
        if "last_p_points" in data and data["last_p_points"] is not None:
            self._last_p_points = int(data["last_p_points"])
        if "last_turns_remaining" in data and data["last_turns_remaining"] is not None:
            self._last_turns_remaining = max(0, min(15, int(data["last_turns_remaining"])))
        if "in_battle_scene" in data:
            self._in_battle_scene = bool(data["in_battle_scene"])
        if isinstance(data.get("shop_visited_slots"), (list, tuple)):
            self._shop_visited_slots = {str(x) for x in data["shop_visited_slots"]}
        if "shop_last_clicked_slot" in data:
            self._shop_last_clicked_slot = str(data["shop_last_clicked_slot"]) if data["shop_last_clicked_slot"] else None
        if isinstance(data.get("reward_visited_slots"), (list, tuple)):
            self._reward_visited_slots = {int(x) for x in data["reward_visited_slots"]}

    def _get_plan_cards(self, plan: str) -> List[Dict[str, Any]]:
        key = plan.lower()
        if key not in self._cards_cache:
            self._cards_cache[key] = self.db.get_cards_for_plan(key)
        return self._cards_cache[key]

    def _get_plan_drinks(self, plan: str) -> List[Dict[str, Any]]:
        key = plan.lower()
        if key not in self._drinks_cache:
            self._drinks_cache[key] = self.db.get_drinks_for_plan(key)
        return self._drinks_cache[key]

    def match_card_by_name(self, raw_name: str, plan: str, upgraded: bool = False) -> Optional[Dict[str, Any]]:
        norm_in = _normalize_jp(raw_name)
        if len(norm_in) < 2:
            return None
        candidates = self._get_plan_cards(plan)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        has_plus = upgraded or ("+" in raw_name) or ("＋" in raw_name)
        in_has_paren = ("(" in raw_name) or ("（" in raw_name) or (")" in raw_name) or ("）" in raw_name)
        for c in candidates:
            cname = c.get("name") or ""
            cnorm = _normalize_jp(cname)
            if not cnorm:
                continue
            c_plus = bool(c.get("upgraded", 0) == 1 or "+" in cname)
            c_has_paren = ("(" in cname) or ("（" in cname)
            if cnorm == norm_in:
                score = 1.0 + (0.05 if c_plus == has_plus else 0.0)
            elif norm_in in cnorm or cnorm in norm_in:
                ratio = min(len(norm_in), len(cnorm)) / max(len(norm_in), len(cnorm))
                score = 0.72 + 0.25 * ratio + (0.05 if c_plus == has_plus else 0.0)
            else:
                score = SequenceMatcher(None, norm_in, cnorm).ratio()
                if c_plus == has_plus:
                    score += 0.03
            if in_has_paren:
                score += 0.12 if c_has_paren else -0.05
            elif c_has_paren:
                score -= 0.25
            if score > best_score:
                best_score = score
                best = c
        return best if best_score >= 0.62 else None

    def match_drink_by_name(self, raw_name: str, plan: str) -> Optional[Dict[str, Any]]:
        """Fuzzy-match an OCR'd P-Drink name against DB drinks, preferring the current plan."""
        norm_in = _normalize_jp(raw_name)
        if len(norm_in) < 2:
            return None
        pool: List[Dict[str, Any]] = []
        seen_ids: Set[int] = set()
        ordered_plans = [plan.lower()] + [p for p in ("logic", "sense", "anomaly") if p != plan.lower()]
        for p_name in ordered_plans:
            for d in self._get_plan_drinks(p_name):
                did = int(d["id"])
                if did not in seen_ids:
                    seen_ids.add(did)
                    pool.append(d)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for d in pool:
            nd = _normalize_jp(d.get("name") or "")
            if not nd:
                continue
            if nd == norm_in:
                score = 1.0
            elif nd in norm_in or norm_in in nd:
                ratio = min(len(norm_in), len(nd)) / max(len(norm_in), len(nd))
                score = 0.78 + 0.20 * ratio
            else:
                score = SequenceMatcher(None, norm_in, nd).ratio()
            if score > best_score:
                best_score = score
                best = d
        return best if best_score >= 0.68 else None

    def _match_square_drink_slot_with_score(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Identify a candidate P-Drink thumbnail inside a square reward slot and return (drink_dict, corr_score)."""
        h, w = frame.shape[:2]
        templates = self._load_drink_templates()
        if not templates or x2 <= x1 or y2 <= y1:
            return None, 0.0
        ref_mat = np.stack([ref for _, _, _, ref in templates], axis=0)
        mask = templates[0][2]
        sw = max(24, x2 - x1)
        cx = (x1 + x2) // 2
        cy_cands = [
            y1 + int(sw * 0.42),
            y1 + int(sw * 0.46),
            y1 + int(sw * 0.50),
            y1 + int(sw * 0.54),
            (y1 + y2) // 2,
        ]
        rad_cands = sorted({max(12, int(sw * 0.44)), max(14, int(sw * 0.50)), max(16, int(sw * 0.55))})
        best_sc = -1.0
        best_did: Optional[int] = None
        for cy in cy_cands:
            for rad in rad_cands:
                crop = frame[max(0, cy - rad) : min(h, cy + rad), max(0, cx - rad) : min(w, cx + rad)]
                if crop.size == 0:
                    continue
                sm = np.array(Image.fromarray(crop).resize((24, 24), Image.Resampling.BILINEAR), dtype=np.float32).flatten()
                v = sm[mask]
                v = (v - v.mean()) / (v.std() + 1e-6)
                scs = (ref_mat @ v) / len(v)
                idx = int(np.argmax(scs))
                sc = float(scs[idx])
                if sc > best_sc:
                    best_sc = sc
                    best_did = int(templates[idx][0])
        if best_did is not None and best_sc >= 0.55:
            return self.db.get_drink_by_id(best_did), best_sc
        return None, max(0.0, best_sc)

    def _match_square_drink_slot(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> Optional[Dict[str, Any]]:
        """Identify a candidate P-Drink thumbnail inside a square reward slot via template correlation."""
        matched, _ = self._match_square_drink_slot_with_score(frame, x1, y1, x2, y2)
        return matched

    @staticmethod
    def _is_banned_drink(
        drink_id: Optional[int],
        name: str = "",
        desc: str = "",
        banned_ids: Optional[Sequence[int]] = None,
        banned_names: Optional[Sequence[str]] = None,
    ) -> bool:
        """Return True for banned P-drinks (default #19 初星スペシャル青汁)."""
        eff_ids = {19} if banned_ids is None else {int(x) for x in banned_ids}
        if drink_id is not None and int(drink_id) in eff_ids:
            return True
        combined = f"{name or ''} {desc or ''}"
        if 19 in eff_ids and ("青汁" in combined or "ランダムな強化済みスキルカード" in combined):
            return True
        for bnm in banned_names or ():
            if bnm and bnm in combined:
                return True
        return False

    @classmethod
    def _append_banned_drink_discards(
        cls,
        options: List[Dict[str, Any]],
        held_drinks: Sequence[RecognizedDrink],
        banned_ids: Optional[Sequence[int]] = None,
        banned_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Append `破棄候補_<id>` options for any banned drink in `held_drinks` and re-index `choice_id`."""
        existing_discard_ids = {
            int(o["drink_id"])
            for o in options
            if str(o.get("name") or "").startswith("破棄候補_") and o.get("drink_id") is not None
        }
        added = False
        for di, d in enumerate(held_drinks or (), 1):
            if (
                cls._is_banned_drink(
                    d.drink_id,
                    d.name,
                    d.dsl_summary,
                    banned_ids=banned_ids,
                    banned_names=banned_names,
                )
                and d.drink_id not in existing_discard_ids
            ):
                options.append({
                    "choice_id": len(options),
                    "drink_id": d.drink_id,
                    "name": f"破棄候補_{d.drink_id}",
                    "title": f"点击饮料栏旧饮料{di}《{d.name}》打开详情删除(捨てる/破棄)",
                    "description": f"发现禁用饮料《{d.name}》({d.dsl_summary})！点击左下角饮料栏第{di}瓶打开详情点『捨てる/破棄』删除",
                    "bbox": d.bbox.model_dump(),
                })
                existing_discard_ids.add(d.drink_id)
                added = True
        if added:
            for idx, o in enumerate(options):
                o["choice_id"] = idx

    def _ocr_crop_upscaled(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        scale: int = 3,
        use_en: bool = False,
        mode: str = "raw",
        pad: int = 0,
    ) -> str:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
        if x2 <= x1 or y2 <= y1:
            return ""
        patch = frame[y1:y2, x1:x2]
        if mode == "white":
            mask = (patch[:, :, 0] > 195) & (patch[:, :, 1] > 195) & (patch[:, :, 2] > 195)
            patch = np.where(mask[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
        elif mode == "dark_stat":
            gray = np.mean(patch[:, :, :3], axis=2)
            mask = gray < 185
            patch = np.where(mask[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
        elif mode == "dark_on_light":
            gray = np.mean(patch[:, :, :3], axis=2)
            mask = gray < 215
            patch = np.where(mask[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
        pil_img = Image.fromarray(patch)
        if scale != 1:
            pil_img = pil_img.resize(
                (max(1, (x2 - x1) * scale), max(1, (y2 - y1) * scale)),
                Image.Resampling.BICUBIC,
            )
        if pad > 0:
            from PIL import ImageOps
            pil_img = ImageOps.expand(pil_img, border=pad, fill=(255, 255, 255))
        engine = self.en_ocr if use_en else self.ocr
        res = engine.recognize_array(np.array(pil_img))
        return _clean_ocr_text(res.get("text", ""))

    def _parse_week_and_stats(self, lines: List[Dict[str, Any]], frame: np.ndarray) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        full_text = "".join(ln["text"] for ln in lines)

        # Top-left header box: padded 1x captures "中間まで 5週", dark_on_light captures "授業" / "おでかけ"
        week_crop_pad = self._ocr_crop_upscaled(
            frame, (int(w * 0.04), int(h * 0.022), int(w * 0.26), int(h * 0.095)), scale=1, mode="raw", pad=25
        )
        hdr_label_crop = self._ocr_crop_upscaled(
            frame,
            (int(w * 0.03), int(h * 0.018), int(w * 0.27), int(h * 0.055)),
            scale=3,
            mode="dark_on_light",
        )
        week_header_text = f"{week_crop_pad} {hdr_label_crop}".strip()
        week_search_text = f"{week_header_text} {full_text}"
        if any(k in week_search_text for k in ("中間まで", "最終まで", "本戦まで")) and not re.search(
            r"(中間まで|最終まで|本戦まで).*?\d+\s*(?:週|日)", week_search_text
        ):
            rem_num_txt = self._ocr_crop_upscaled(
                frame, (int(w * 0.048), int(h * 0.048), int(w * 0.24), int(h * 0.095)), scale=1, mode="raw", pad=25
            )
            # WinRT ja OCR occasionally misreads giant '3' in '3週'/'3日' as 'ろ' or 'B'
            rem_num_txt = re.sub(
                r"([0-9ろBSloOgq])\s*([週日])",
                lambda m: {"ろ": "3", "B": "3", "l": "1", "I": "1", "S": "5", "s": "5", "g": "9", "q": "9", "o": "0", "O": "0"}.get(m.group(1), m.group(1))
                + m.group(2),
                rem_num_txt,
            )
            week_search_text = f"{week_crop_pad}{rem_num_txt} {hdr_label_crop} {full_text}"

        week_label = self._last_week_label
        week_num = self._last_week

        banner_rem: Optional[int] = None
        if "本戦まで" in week_search_text:
            for d_x1_frac, d_x2_frac in ((0.070, 0.185), (0.048, 0.165)):
                d_patch = frame[int(h * 0.053) : int(h * 0.093), int(w * d_x1_frac) : int(w * d_x2_frac)]
                if d_patch.size > 0:
                    dmask = (d_patch[:, :, :3] < 130).all(axis=2)
                    cand_rem = self._ocr_bw_banner_number(dmask, prefix="D:", max_val=9, scale=2, min_val=1)
                    if cand_rem is not None and 1 <= cand_rem <= 6:
                        banner_rem = cand_rem
                        break
        elif "中間まで" in week_search_text or "最終まで" in week_search_text:
            w_dig = frame[int(h * 0.046) : int(h * 0.094), int(w * 0.055) : int(w * 0.205)]
            if w_dig.size > 0:
                w_core = self._extract_outlined_white_core(w_dig, radius=3, white_min=205, dark_max=135)
                cand_rem = self._ocr_bw_banner_number(w_core, prefix="W:", max_val=9, scale=3, min_val=1)
                if cand_rem is not None and 1 <= cand_rem <= 6:
                    banner_rem = cand_rem

        is_hif = (
            self.scenario_id in ("hif_honsen", "hif")
            or "H.I.F" in week_search_text
            or "本戦まで" in week_search_text
            or "本戦ラウンド" in week_search_text
        )
        m_hif = re.search(r"本戦まで.*?(\d+)\s*日", week_search_text)
        m_mid = re.search(r"中間まで.*?(\d+)週", week_search_text)
        m_fin = re.search(r"最終まで.*?(\d+)週", week_search_text)
        if m_hif or ("本戦まで" in week_search_text and banner_rem is not None):
            rem = banner_rem if banner_rem is not None else int(m_hif.group(1))
            rem = max(1, min(6, rem))
            week_label = f"H.I.F本戦まで {rem}日"
            week_num = max(1, min(6, 7 - rem))
        elif is_hif and ("ラウンド1" in week_header_text or "本戦ラウンド1" in week_search_text):
            week_label = "H.I.F本戦ラウンド1"
            week_num = 7
        elif is_hif and ("インターバル" in week_header_text or "インターバル" in week_search_text):
            week_label = "インターバル"
            week_num = 8
        elif is_hif and ("ラウンド2" in week_header_text or "本戦ラウンド2" in week_search_text or "FINAL" in week_header_text.upper()):
            week_label = "H.I.F本戦ラウンド2"
            week_num = 9
        elif m_mid or ("中間まで" in week_search_text and banner_rem is not None):
            rem = banner_rem if banner_rem is not None else int(m_mid.group(1))
            week_label = f"中間まで {rem}週"
            week_num = max(1, min(6, 6 - rem)) if self.scenario_id == "hajime_regular" else max(1, 8 - rem)
        elif m_fin or ("最終まで" in week_search_text and banner_rem is not None):
            rem = banner_rem if banner_rem is not None else int(m_fin.group(1))
            week_label = f"最終まで {rem}週"
            week_num = max(7, min(13, 13 - rem))
        elif "中間試験" in week_header_text:
            week_label = "中間試験"
            week_num = 6
        elif "最終試験" in week_header_text:
            week_label = "最終試験"
            week_num = 13

        # Stamina via en-US OCR on top-center bar ("23/31" white on dark, or "44/44" dark on light in H.I.F)
        stamina, max_stamina = self._last_stamina
        hp_found = False

        def _is_suspicious_stamina_drop(s_val: int, m_val: int) -> bool:
            prev_s, prev_m = self._last_stamina
            if self._in_battle_scene or prev_m <= 0 or abs(m_val - prev_m) > 6:
                return False
            if prev_s >= 15 and s_val < 10 and (prev_s - s_val) >= 15:
                return True
            if prev_s >= 25 and (prev_s - s_val) >= 24:
                return True
            return False

        white_probe = (
            (int(w * 0.42), int(h * 0.028), int(w * 0.62), int(h * 0.068)),
            "white",
            True,
        )
        dol_probes = (
            ((int(round(w * 0.4514)), int(round(h * 0.0406)), int(round(w * 0.5972)), int(round(h * 0.0638))), "raw", True),
            ((int(round(w * 0.4514)), int(round(h * 0.0406)), int(round(w * 0.5972)), int(round(h * 0.0638))), "dark_on_light", False),
            ((int(round(w * 0.4514)), int(round(h * 0.0425)), int(round(w * 0.6042)), int(round(h * 0.0625))), "dark_on_light", False),
            ((int(round(w * 0.458)), int(round(h * 0.0406)), int(round(w * 0.5972)), int(round(h * 0.0638))), "dark_on_light", True),
        )
        hp_probes = (dol_probes + (white_probe,)) if is_hif else ((white_probe,) + dol_probes)
        suspicious_candidates: List[Tuple[int, int]] = []

        for hp_box, hp_mode, hp_en in hp_probes:
            hp_crop_text = self._ocr_crop_upscaled(
                frame,
                hp_box,
                scale=3,
                use_en=hp_en,
                mode=hp_mode,
                pad=15,
            )
            m_hp = re.search(r"(\d{1,3})/(\d{2,3})", hp_crop_text)
            if m_hp:
                s_val, m_val = int(m_hp.group(1)), int(m_hp.group(2))
                if 0 <= s_val <= m_val <= 150:
                    if not _is_suspicious_stamina_drop(s_val, m_val):
                        stamina, max_stamina = s_val, m_val
                        self._last_stamina = (stamina, max_stamina)
                        self._pending_stamina_reading = None
                        self._pending_stamina_streak = 0
                        hp_found = True
                        break
                    suspicious_candidates.append((s_val, m_val))

        if not hp_found:
            for ln in lines:
                if int(w * 0.35) <= ln["bbox"][0] <= int(w * 0.56) and ln["bbox"][1] <= int(h * 0.08):
                    m_hp_ln = re.search(r"(\d{1,3})/(\d{2,3})", ln["text"])
                    if m_hp_ln:
                        s_val, m_val = int(m_hp_ln.group(1)), int(m_hp_ln.group(2))
                        if 0 <= s_val <= m_val <= 150:
                            if not _is_suspicious_stamina_drop(s_val, m_val):
                                stamina, max_stamina = s_val, m_val
                                self._last_stamina = (stamina, max_stamina)
                                self._pending_stamina_reading = None
                                self._pending_stamina_streak = 0
                                hp_found = True
                                break
                            suspicious_candidates.append((s_val, m_val))

        if (not hp_found) and suspicious_candidates:
            cand_s, cand_m = max(suspicious_candidates, key=lambda pair: pair[0])
            if (
                self._pending_stamina_reading is not None
                and self._pending_stamina_reading[1] == cand_m
                and abs(self._pending_stamina_reading[0] - cand_s) <= 2
            ):
                self._pending_stamina_streak += 1
                if self._pending_stamina_streak >= 2:
                    stamina, max_stamina = cand_s, cand_m
                    self._last_stamina = (stamina, max_stamina)
                    self._pending_stamina_reading = None
                    self._pending_stamina_streak = 0
                    hp_found = True
            else:
                self._pending_stamina_reading = (cand_s, cand_m)
                self._pending_stamina_streak = 1

        # P-points below stamina (only parse on out-of-battle schedule/shop HUD to avoid reading battle scoreboard digits)
        p_points = self._last_p_points
        p_found = False

        def _is_digit_subseq(sub: str, full: str) -> bool:
            if len(sub) >= len(full):
                return False
            it = iter(full)
            return all(ch in it for ch in sub)

        def _is_truncated_p_points(val: int) -> bool:
            prev = self._last_p_points
            if prev < 50 or val >= prev // 3:
                return False
            s_val, s_prev = str(val), str(prev)
            if s_prev.startswith(s_val):
                return True
            if prev >= 100 and val < 10 and (prev % 5 == 0) and (val % 5 != 0):
                return True
            # Guard middle-digit / leading-digit drops on high P-point counts (e.g. 449->49, 439->49, 455->55, 255->55),
            # while still allowing legitimate single-step ~100-150P spends (105->5, 120->20).
            if (prev - val) >= 180 and (
                _is_digit_subseq(s_val, s_prev)
                or s_prev.endswith(s_val)
                or (val < 100 and (s_val[0] == s_prev[0] or s_val[-1] == s_prev[-1]))
            ):
                return True
            return False

        def _ocr_hud_p_points_sandwich() -> Optional[int]:
            """Extract outlined white P-point digits (0..9999) via adaptive white-core + '10+<d>+10' sandwich OCR."""
            if h < 400 or w < 240:
                return None
            crop = frame[int(h * 0.056) : int(h * 0.092), int(w * 0.50) : int(w * 0.62)]
            if crop.size == 0:
                return None
            ch, cw = crop.shape[:2]
            lum = crop.mean(axis=2)
            sat = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
            p99 = float(np.percentile(lum, 99))
            if p99 < 145.0:
                return None
            thr = max(135.0, p99 - 42.0)
            wmask = (lum > thr) & (sat < 35)
            num_l, labels, stats_cc, _ = cv2.connectedComponentsWithStats(wmask.astype(np.uint8))
            clean = np.zeros_like(wmask, dtype=np.uint8)
            for idx_cc in range(1, num_l):
                area = int(stats_cc[idx_cc, cv2.CC_STAT_AREA])
                bw_cc = int(stats_cc[idx_cc, cv2.CC_STAT_WIDTH])
                bh_cc = int(stats_cc[idx_cc, cv2.CC_STAT_HEIGHT])
                if area >= 8 and int(ch * 0.22) <= bh_cc <= int(ch * 0.85) and bw_cc <= int(cw * 0.45):
                    clean[labels == idx_cc] = 255
            ys_cc, xs_cc = np.where(clean > 0)
            if len(xs_cc) == 0:
                return None
            sub = clean[ys_cc.min() : ys_cc.max() + 1, xs_cc.min() : xs_cc.max() + 1]
            up = cv2.resize(sub, (max(1, sub.shape[1] * 3), max(1, sub.shape[0] * 3)), interpolation=cv2.INTER_CUBIC)
            _, up = cv2.threshold(up, 115, 255, cv2.THRESH_BINARY)
            inv = 255 - up
            dh, dw = inv.shape[:2]
            pad_y = 32
            pref, suff, gap = "10+", "+10", 8
            fscale = max(0.8, dh / 24.0)
            (tw_p, _), _ = cv2.getTextSize(pref, cv2.FONT_HERSHEY_SIMPLEX, fscale, 4)
            (sw_p, _), _ = cv2.getTextSize(suff, cv2.FONT_HERSHEY_SIMPLEX, fscale, 4)
            x0 = 30
            dx = x0 + tw_p + gap
            sx = dx + dw + gap
            canvas = np.full((dh + pad_y * 2, sx + sw_p + 40, 3), 255, dtype=np.uint8)
            cv2.putText(canvas, pref, (x0, pad_y + dh - 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), 4, cv2.LINE_AA)
            canvas[pad_y : pad_y + dh, dx : dx + dw] = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
            cv2.putText(canvas, suff, (sx, pad_y + dh - 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), 4, cv2.LINE_AA)
            en_txt = self.en_ocr.recognize_array(canvas).get("text", "")
            m_sw = re.search(r"10\+\s*(\d{1,4})\s*\+10", en_txt)
            if m_sw:
                val_sw = int(m_sw.group(1))
                if 0 <= val_sw <= 9999:
                    return val_sw
            return None

        is_shop_hud = ("Pポイント" in full_text) or ("交換する" in full_text)
        if hp_found or is_shop_hud:
            p_sw = _ocr_hud_p_points_sandwich()
            if p_sw is not None and not _is_truncated_p_points(p_sw):
                p_points = p_sw
                self._last_p_points = p_points
                p_found = True
            if not p_found:
                p_radius = max(3, int(round(w / 240.0)))
                p_dig = frame[int(h * 0.056) : int(h * 0.090), int(w * 0.50) : int(w * 0.63)]
                p_val = self._ocr_bw_banner_number(
                    self._extract_outlined_white_core(p_dig, radius=p_radius, white_min=205, dark_max=135),
                    prefix="P:",
                    max_val=9999,
                    scale=3,
                )
                if p_val is not None and p_val < 100 and h >= 400 and w >= 240:
                    wide_cands: List[int] = []
                    for w_mode in ("white", "raw"):
                        w_txt = self._ocr_crop_upscaled(
                            frame,
                            (int(w * 0.44), int(h * 0.052), int(w * 0.64), int(h * 0.095)),
                            scale=3,
                            use_en=True,
                            mode=w_mode,
                            pad=15,
                        )
                        w_clean = re.sub(r"[',.`\s]", "", w_txt)
                        m_w = re.search(r"(\d{1,4})", w_clean)
                        if m_w:
                            cand_w = int(m_w.group(1))
                            if 100 <= cand_w <= 9999:
                                wide_cands.append(cand_w)
                    sfx_hit = next((c for c in wide_cands if str(c).endswith(str(p_val))), None)
                    if sfx_hit is not None:
                        p_val = sfx_hit
                    elif len(wide_cands) >= 2 and len(set(wide_cands)) == 1:
                        p_val = wide_cands[0]
                if p_val is not None and not _is_truncated_p_points(p_val):
                    p_points = p_val
                    self._last_p_points = p_points
                    p_found = True
            if not p_found:
                for p_mode in ("white", "dark_on_light", "raw"):
                    for en_flag in (True, False):
                        p_crop_text = self._ocr_crop_upscaled(
                            frame,
                            (int(w * 0.48), int(h * 0.064), int(w * 0.64), int(h * 0.095)),
                            scale=3,
                            use_en=en_flag,
                            mode=p_mode,
                            pad=15,
                        )
                        p_clean = re.sub(r"[',.`\s]", "", p_crop_text)
                        m_pp = re.search(r"(\d{1,4})", p_clean)
                        if m_pp:
                            cand_p = int(m_pp.group(1))
                            if (not is_shop_hud) and self._last_p_points >= 50 and cand_p < 10:
                                continue
                            if _is_truncated_p_points(cand_p):
                                continue
                            p_points = cand_p
                            self._last_p_points = p_points
                            p_found = True
                            break
                    if p_found:
                        break
                if not p_found and h >= 400:
                    p_wide = frame[int(h * 0.056) : int(h * 0.095), int(w * 0.48) : int(w * 0.64)]
                    p_val2 = self._ocr_bw_banner_number(
                        self._extract_outlined_white_core(p_wide, radius=p_radius, white_min=195, dark_max=145),
                        prefix="P:",
                        max_val=9999,
                        scale=3,
                        min_val=0,
                    )
                    if p_val2 is not None and not _is_truncated_p_points(p_val2):
                        p_points = p_val2
                        self._last_p_points = p_points

        # Vo / Da / Vi stats: locate "/1000" or "/3200" row, or fall back to fixed known bands
        stats = dict(self._last_stats)
        y_1000_candidates = [
            ln["bbox"][1]
            for ln in lines
            if ("1000" in ln["text"] or "/108" in ln["text"] or "/100" in ln["text"] or bool(re.search(r"/\s*[12358]\d00", ln["text"])))
            and (int(h * 0.25) <= ln["bbox"][1] <= int(h * 0.72))
        ]
        probe_y_bases = list(y_1000_candidates[:1])
        is_confirm_screen = any(
            k in full_text or k in hdr_label_crop
            for k in ("開始確認", "獲得量上昇", "プロデュース開始", "ース開始")
        )
        if not probe_y_bases:
            if any(k in week_search_text for k in ("中間まで", "最終まで", "本戦まで")) or any(
                k in full_text for k in ("行動を選択", "を選択してください", "を選択")
            ):
                probe_y_bases = [int(h * 0.651)]
            elif any(k in hdr_label_crop for k in ("授業", "おでかけ", "活動支給", "公開レッスン", "差し入れ", "相談")):
                probe_y_bases = [int(h * 0.332)]
            elif is_confirm_screen:
                probe_y_bases = [int(h * 0.528)]

        if probe_y_bases:
            y_base = min(probe_y_bases)
            stat_y1 = int(h * 0.495) if is_confirm_screen else max(0, y_base - int(h * 0.032))
            stat_y2 = int(h * 0.528) if is_confirm_screen else max(stat_y1 + 10, y_base - 2)
            if is_confirm_screen:
                bands = [
                    ("vocal", (int(w * 0.505), stat_y1, int(w * 0.655), stat_y2)),
                    ("dance", (int(w * 0.680), stat_y1, int(w * 0.820), stat_y2)),
                    ("visual", (int(w * 0.840), stat_y1, int(w * 0.985), stat_y2)),
                ]
            else:
                bands = [
                    ("vocal", (int(w * 0.240), stat_y1, int(w * 0.396), stat_y2)),
                    ("dance", (int(w * 0.435), stat_y1, int(w * 0.605), stat_y2)),
                    ("visual", (int(w * 0.630), stat_y1, int(w * 0.820), stat_y2)),
                ]
            stat_max_cap = 3200
            parsed_band: Dict[str, int] = {}
            for k, box in bands:
                raw_box = (box[0], box[1], box[2], min(h, box[3] + int(h * 0.004)))
                cands: List[Tuple[int, int, int]] = []
                mode_seq = (
                    (
                        (box, "dark_stat", 3, 15),
                        (box, "white", 3, 12),
                        (box, "raw", 3, 15),
                        (box, "dark_on_light", 2, 15),
                    )
                    if is_confirm_screen
                    else (
                        (box, "white", 3, 12),
                        (raw_box, "dark_stat", 3, 15),
                        (raw_box, "raw", 3, 15),
                        (raw_box, "dark_on_light", 2, 15),
                    )
                )
                for mode_idx, (m_box, m_name, m_scale, m_pad) in enumerate(mode_seq):
                    t_stat = self._ocr_crop_upscaled(frame, m_box, scale=m_scale, use_en=True, mode=m_name, pad=m_pad)
                    t_head = re.sub(r"[',.`\s]", "", t_stat.split("/", 1)[0])
                    for m_num in re.finditer(r"(\d{2,4})", t_head):
                        dig_s = m_num.group(1)
                        v_cand = int(dig_s)
                        if 10 <= v_cand <= stat_max_cap:
                            cands.append((len(str(v_cand)), -mode_idx, v_cand))
                    if cands and max(c[0] for c in cands) >= 4:
                        break
                if cands:
                    best_v = max(cands, key=lambda c: (c[0], c[1]))[2]
                    parsed_band[k] = min(stat_max_cap, best_v)
            is_setup_stat_screen = is_confirm_screen or any(
                sk in full_text
                for sk in ("アイドル選択", "プロデュース開始", "開始確認", "選抜試験メモリー", "育成情報", "Pアイドル設定")
            )
            if len(parsed_band) >= 2:
                # Guard against single-stat 4-digit OCR truncation (e.g. Vi 1644 -> 69 or 1824 -> 14) mid-produce
                if (
                    self._last_stats
                    and week_num >= self._last_week
                    and not is_setup_stat_screen
                    and any(
                        self._last_stats.get(mk, 0) > 0 and mv >= self._last_stats[mk] * 0.85
                        for mk, mv in parsed_band.items()
                    )
                ):
                    for sk, sv in list(parsed_band.items()):
                        prev_v = min(stat_max_cap, int(self._last_stats.get(sk, 0) or 0))
                        if prev_v >= 300 and sv < prev_v * 0.65:
                            parsed_band[sk] = prev_v
                stats.update(parsed_band)
                stats = {sk: min(stat_max_cap, max(0, int(sv))) for sk, sv in stats.items()}
                if not is_setup_stat_screen:
                    self._last_stats = dict(stats)
        else:
            stat_max_cap = 3200
            is_shop_or_reward = any(k in full_text for k in ("相談", "交換", "交換する", "受け取る"))
            stat_vals: List[Tuple[int, int]] = []
            if not is_shop_or_reward:
                for ln in lines:
                    cx = (ln["bbox"][0] + ln["bbox"][2]) // 2
                    cy = (ln["bbox"][1] + ln["bbox"][3]) // 2
                    if int(h * 0.56) <= cy <= int(h * 0.67):
                        m_num = re.search(r"(?<!/)(\d{2,4})(?!\d|%)", ln["text"])
                        if m_num and not any(cap_tok in ln["text"] for cap_tok in ("1000", "3200", "5200")):
                            val = int(m_num.group(1))
                            if 20 <= val <= stat_max_cap:
                                stat_vals.append((cx, min(stat_max_cap, val)))
                stat_vals.sort(key=lambda t: t[0])
                if len(stat_vals) >= 3:
                    cand_stats = {
                        "vocal": min(stat_max_cap, stat_vals[0][1]),
                        "dance": min(stat_max_cap, stat_vals[1][1]),
                        "visual": min(stat_max_cap, stat_vals[2][1]),
                    }
                    if all(cand_stats[k] >= self._last_stats.get(k, 0) for k in ("vocal", "dance", "visual")):
                        stats = cand_stats
                        self._last_stats = dict(stats)
        stats = {sk: min(3200, max(0, int(sv))) for sk, sv in stats.items()}

        if self._last_week >= 2 and week_num < self._last_week and max(stats.values() or [0]) > 175:
            week_num = self._last_week
            week_label = self._last_week_label
        self._last_week = week_num
        self._last_week_label = week_label

        return {
            "week": week_num,
            "total_weeks": get_scenario_spec(self.scenario_id).total_weeks,
            "week_label": week_label,
            "header_label": hdr_label_crop,
            "stamina": stamina,
            "max_stamina": max_stamina,
            "p_points": p_points,
            "stats": stats,
        }

    def recognize(self, frame: np.ndarray, plan: Optional[str] = None) -> ScreenObservation:
        active_plan = (plan or self.plan).lower()
        h, w = frame.shape[:2]
        ocr_res = self.ocr.recognize_array(frame)
        lines = list(ocr_res.get("lines", []))
        if len(lines) <= 10 and h >= 400 and w >= 300:
            # Sparse full-frame OCR on Home / 1.アイドル選択 screens: probe top-left step pill and bottom nav bar at 2x
            tl_w, tl_h = int(w * 0.48), int(h * 0.075)
            tl_up = cv2.resize(frame[0:tl_h, 0:tl_w], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            tl_uh, tl_uw = tl_up.shape[:2]
            for tln in self.ocr.recognize_array(tl_up).get("lines", []):
                tb = tln["bbox"]
                if tb[2] > tl_uw + 4 or tb[3] > tl_uh + 4:
                    continue
                shifted = [tb[0] // 2, tb[1] // 2, tb[2] // 2, tb[3] // 2]
                if not any(tln["text"].strip() == ex["text"].strip() and abs(shifted[1] - ex["bbox"][1]) <= 12 for ex in lines):
                    lines.append({**tln, "bbox": shifted})
            bn_y1, bn_y2 = int(h * 0.85), int(h * 0.97)
            bn_up = cv2.resize(frame[bn_y1:bn_y2, :], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            bn_uh, bn_uw = bn_up.shape[:2]
            for bln in self.ocr.recognize_array(bn_up).get("lines", []):
                bb = bln["bbox"]
                if bb[2] > bn_uw + 4 or bb[3] > bn_uh + 4:
                    continue
                shifted = [bb[0] // 2, bn_y1 + bb[1] // 2, bb[2] // 2, bn_y1 + bb[3] // 2]
                if not any(bln["text"].strip() == ex["text"].strip() and abs(shifted[1] - ex["bbox"][1]) <= 12 for ex in lines):
                    lines.append({**bln, "bbox": shifted})
            pb_y1, pb_y2 = int(h * 0.78), int(h * 0.865)
            pb_x1, pb_x2 = int(w * 0.22), int(w * 0.78)
            pb_crop = frame[pb_y1:pb_y2, pb_x1:pb_x2]
            if pb_crop.size > 0:
                pb_up = cv2.resize(pb_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                pb_uh, pb_uw = pb_up.shape[:2]
                for pln in self.ocr.recognize_array(pb_up).get("lines", []):
                    pb = pln["bbox"]
                    if pb[2] > pb_uw + 4 or pb[3] > pb_uh + 4:
                        continue
                    shifted = [pb_x1 + pb[0] // 2, pb_y1 + pb[1] // 2, pb_x1 + pb[2] // 2, pb_y1 + pb[3] // 2]
                    if not any(pln["text"].strip() == ex["text"].strip() and abs(shifted[1] - ex["bbox"][1]) <= 12 for ex in lines):
                        lines.append({**pln, "bbox": shifted})
            plv_crop = frame[int(h * 0.01) : int(h * 0.10), int(w * 0.02) : int(w * 0.22)]
            if plv_crop.size > 0 and not any("PLV" in (ex.get("text") or "").upper() for ex in lines):
                plv_up = cv2.resize(plv_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                plv_txt = self.en_ocr.recognize_array(plv_up).get("text", "")
                if "PLV" in plv_txt.upper():
                    lines.append({"text": "PLv", "bbox": [int(w * 0.04), int(h * 0.02), int(w * 0.16), int(h * 0.08)]})
            # Check washed-out fade-in bottom-right corner for subtle "NOW LOADING..."
            if len(lines) <= 6 and float(frame.mean()) > 190.0:
                ld_crop = frame[int(h * 0.92) : h, int(w * 0.68) : w]
                if ld_crop.size > 0:
                    ld_up = cv2.resize(ld_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    ld_txt = self.en_ocr.recognize_array(ld_up).get("text", "")
                    if "LOADING" in ld_txt.upper():
                        lines.append({"text": "NOW LOADING", "bbox": [int(w * 0.72), int(h * 0.94), int(w * 0.96), int(h * 0.98)]})
        if (not lines or not any(ln["bbox"][1] > int(h * 0.65) for ln in lines)) and h >= 400:
            # Bokeh/blur upper backdrop (e.g. Pドリンク詳細) can omit lower sheet in full-frame OCR; scan lower sheet directly
            y_sub = int(h * 0.62)
            sub_crop = frame[y_sub : int(h * 0.96), :]
            sub_h, sub_w = sub_crop.shape[:2]
            sub_res = self.ocr.recognize_array(sub_crop)
            for sln in sub_res.get("lines", []):
                sb = sln["bbox"]
                if sb[2] > sub_w + 4 or sb[3] > sub_h + 4:
                    continue
                shifted = [sb[0], sb[1] + y_sub, sb[2], sb[3] + y_sub]
                if not any(
                    sln["text"].strip() == ex["text"].strip() and abs(shifted[1] - ex["bbox"][1]) <= 12
                    for ex in lines
                ):
                    lines.append({**sln, "bbox": shifted})
        joined = "".join(ln["text"] for ln in lines)
        joined_upper = joined.upper()

        if ("ショップ" in joined and "ガシャ" in joined) or any(
            k in joined for k in ("プロデュース開始", "アイドル選択", "選抜試験メモリー", "Pアイドル設定")
        ):
            self._last_week = 1
            self._last_week_label = ""
            self._in_battle_scene = False
            self._non_battle_streak = 0

        # 0. Loading / Data download / Connecting / Support-card flash transition screen
        is_loadout_detail_modal = (
            "基本情報" in joined
            and any(k in joined for k in ("アビリティ", "履歴", "編成"))
            and any(k in joined for k in ("Pアイテム", "スキルカード", "ボーカル", "ダンス", "ビジュアル"))
        )
        is_support_splash = (
            not is_loadout_detail_modal
            and "スキルカードサポート" in joined
            and h >= 400
            and float(frame[int(h * 0.42) : int(h * 0.60), int(w * 0.15) : int(w * 0.85)].mean()) > 205.0
        )
        if (
            "LOADING" in joined_upper
            or "CONNECTING" in joined_upper
            or "ダウンロード中" in joined
            or "データダウンロード" in joined
            or "通信中" in joined
            or "移動中" in joined
            or "保存中" in joined
            or is_support_splash
            or (float(frame.mean()) < 16.0 and len(joined) < 12)
        ):
            ctx = compute_week_stats_and_stage_bonus(
                scenario_id=self.scenario_id,
                week=self._last_week,
                plan=active_plan,
                override_stats=self._last_stats,
            )
            return ScreenObservation(
                scene_type="loading",
                plan=active_plan,
                scenario_id=self.scenario_id,
                week=self._last_week,
                total_weeks=get_scenario_spec(self.scenario_id).total_weeks,
                week_activity=self._last_week_label or ctx["week_node"].title,
                stamina=self._last_stamina[0],
                max_stamina=self._last_stamina[1],
                p_points=self._last_p_points,
                idol_stats=dict(self._last_stats),
                p_items=ctx["p_items"],
                stage_bonuses=ctx["stage_bonuses"],
                options=[],
                raw_text=joined,
            )

        if w > h:
            return self._parse_landscape_live_screen(frame, lines, active_plan)

        meta = self._parse_week_and_stats(lines, frame)
        hdr_label = meta.get("header_label", "")
        dim_top_backdrop = float(frame[10 : min(120, h), 10 : min(120, w)].mean()) < 55.0

        # 1. Modal / Tutorial / System Error / Produce Setup ("閉じる", "データ更新", "再挑戦確認", "1.アイドル選択")
        has_close_btn = any("閉じる" in ln["text"] and ln["bbox"][1] > int(h * 0.72) for ln in lines)
        has_bottom_center_close = any(
            ("閉じる" in ln["text"] or "とじる" in ln["text"])
            and ln["bbox"][1] > int(h * 0.84)
            and int(w * 0.28) <= ((ln["bbox"][0] + ln["bbox"][2]) // 2) <= int(w * 0.72)
            for ln in lines
        )
        has_sys_modal = any(
            k in joined
            for k in (
                "データ更新",
                "タイトルへ",
                "通信エラー",
                "チュートリアル",
                "研修案内",
                "一時中断",
                "リタイア",
                "再開する",
                "中断データ",
                "中断したプロデュース",
                "続きから",
                "再挑戦確認",
                "交換確認",
                "交換しますか",
                "強化しますか",
                "削除しますか",
                "チェンジしますか",
                "Pポイントを使用して",
                "使用しますか",
                "カスタマイズしますか",
                "休み確認",
                "休みますか",
                "更新確認",
                "更新しますか",
                "ラインナップを更新",
                "商品を更新",
                "リフレッシュしますか",
                "おまかせ編成",
                "レンタル可能",
                "Pドリンク詳細",
                "ドリンク詳細",
                "ンク詳細",
                "ドリンク詳",
                "表示設定",
                "獲得ガイド表示",
                "メニュー>設定",
                "からいつでも変更できます",
            )
        ) or ("次回以降" in joined and "しますか" in joined) or (
            self._in_battle_scene and dim_top_backdrop and any(k in joined for k in ("使う", "使用する"))
        )
        has_inline_tip = any(k in joined for k in ("タップすると確認", "こちらをタップ", "確認できます"))
        is_hif_scenario_select = (
            ("選抜試験" in joined and "本戦" in joined)
            or ("Hatsuboshi" in joined and "本戦" in joined)
        )
        is_produce_setup = is_hif_scenario_select or (
            (
                any(
                    k in joined
                    for k in (
                        "定期公演",
                        "レギュラー",
                        "アイドル選択",
                        "サポート選択",
                        "サポートカード",
                        "開始確認",
                        "最終確認",
                        "プロデュース開始",
                        "ース開始",
                        "開始する",
                        "編成名変更",
                        "育成情報",
                        "選抜試験メモリー",
                        "Pアイドル設定",
                        "獲得量上昇",
                        "プランALL",
                        "プラン ALL",
                    )
                )
            )
            and not any(end_k in joined for end_k in ("最終評価", "総合評価", "メモリー生成"))
        )
        home_token_groups = (
            ("ショップ", "ショソ"),
            ("ガシャ",),
            ("コンテスト", "っンテスト"),
            ("コミュ",),
            ("プレゼント",),
            ("アチー", "アチ1"),
            ("ッション",),
            ("セール中", "セ-ル中"),
            ("親愛度",),
            ("活動費",),
            ("お仕事",),
            ("課題",),
            ("ホーム",),
            ("PLV",),
        )
        home_tokens = sum(
            1
            for grp in home_token_groups
            if any((hk in joined) or (hk in joined_upper) for hk in grp)
        )
        is_home_screen = (
            home_tokens >= 2
            and any(pk in joined for pk in ("プロデュース", "プロテ", "アイドル", "アイトル", "コンテスト", "っンテスト", "ショップ"))
            and (not dim_top_backdrop)
            and (not has_close_btn)
            and (not has_bottom_center_close)
            and (not has_sys_modal)
            and (not is_produce_setup)
        )
        has_achievement_modal = ("アチーブメント達成" in joined) and ("TAPで次へ" not in joined)
        fg_discard_btn = self._detect_drink_discard_button(frame, lines, w, h)
        is_drink_cap_reward = (
            ("受け取らない" in joined or "Pドリンク所持上限" in joined)
            and fg_discard_btn is None
            and not any(k in joined for k in ("詳細", "破棄", "捨てる", "捨てますか", "進みますか", "よろしいですか", "確認"))
        )
        norm_joined_early = _normalize_katakana_ocr(joined)
        has_move_to_hand_top_pill_early = (
            any(
                ln["text"].strip().lstrip("◆◇✦✓✔ノ・") in ("移動", "移動する", "手札に移動")
                and ln["bbox"][1] < int(h * 0.12)
                and ln["bbox"][0] < int(w * 0.35)
                for ln in lines
            )
            and (not is_home_screen)
            and (not is_produce_setup)
            and ("移動中" not in norm_joined_early)
        )
        has_move_to_hand_btn_early = has_move_to_hand_top_pill_early or (
            any(
                ln["text"].strip().lstrip("◆◇✦✓✔ノ・") in ("移動", "移動する")
                and ln["bbox"][1] > int(h * 0.74)
                for ln in lines
            )
            and ("移動中" not in norm_joined_early)
        )
        is_early_reward_or_change = has_move_to_hand_btn_early or any(
            k in norm_joined_early
            for k in (
                "スキルカードを選",
                "チェンジで獲得",
                "手放すスキルカード",
                "スキルカードが選択されていません",
                "カスタマイズするスキルカード",
                "カスタマイズを選",
                "ドリンクを選",
                "手札に移動",
                "移動するスキルカード",
                "山札か捨札",
                "受け取らない",
                "再抽選",
            )
        )
        has_modal_dialog = (
            dim_top_backdrop
            and (not is_early_reward_or_change)
            and any(k in joined for k in ("キャンセル", "決定", "確認", "はい"))
        )
        is_shop_refresh_popup = any(
            k in joined for k in ("更新する", "更新しますか", "更新確認", "ラインナップを更新", "商品を更新")
        )
        if fg_discard_btn is not None:
            if is_shop_refresh_popup or not any(
                k in joined for k in ("交換", "ドリンク詳細", "ンク詳細", "強化", "カスタマイズ", "チェンジ")
            ):
                self._shop_visited_slots.clear()
                self._shop_last_clicked_slot = None
            return self._parse_popup_screen(frame, lines, active_plan, meta, fg_discard_btn=fg_discard_btn)
        if is_home_screen:
            self._shop_visited_slots.clear()
            self._shop_last_clicked_slot = None
            self._reward_visited_slots.clear()
            return self._parse_home_screen(frame, lines, active_plan, meta)
        if is_produce_setup and (not dim_top_backdrop) and (not has_sys_modal) and (not has_close_btn) and (not has_bottom_center_close):
            self._shop_visited_slots.clear()
            self._shop_last_clicked_slot = None
            self._reward_visited_slots.clear()
            return self._parse_produce_setup_screen(frame, lines, active_plan, meta)
        if is_loadout_detail_modal:
            self._shop_visited_slots.clear()
            self._shop_last_clicked_slot = None
            self._reward_visited_slots.clear()
            back_x, back_y = detect_modal_back_button_xy(frame)
            ctx = compute_week_stats_and_stage_bonus(
                scenario_id=self.scenario_id,
                week=meta["week"],
                plan=active_plan,
                override_stats=meta["stats"],
            )
            return ScreenObservation(
                scene_type="popup",
                plan=active_plan,
                scenario_id=self.scenario_id,
                week=meta["week"],
                total_weeks=meta["total_weeks"],
                week_activity="基本情報 (详情页)",
                stamina=meta["stamina"],
                max_stamina=meta["max_stamina"],
                p_points=meta["p_points"],
                idol_stats=meta["stats"],
                p_items=ctx["p_items"],
                stage_bonuses=ctx["stage_bonuses"],
                options=[
                    {
                        "choice_id": 0,
                        "name": "戻る (<<)",
                        "title": "戻る (<<)",
                        "description": "关闭基本情報详情弹窗返回编成页",
                        "bbox": {
                            "xmin": max(10, back_x - 40),
                            "ymin": max(10, back_y - 32),
                            "xmax": min(w - 10, back_x + 40),
                            "ymax": min(h - 10, back_y + 32),
                        },
                    }
                ],
                raw_text=joined,
            )
        if not is_drink_cap_reward and (
            (has_close_btn and dim_top_backdrop)
            or has_bottom_center_close
            or has_sys_modal
            or has_inline_tip
            or is_produce_setup
            or has_achievement_modal
            or has_modal_dialog
        ):
            if is_shop_refresh_popup or not any(
                k in joined for k in ("交換", "ドリンク詳細", "ンク詳細", "強化", "カスタマイズ", "チェンジ")
            ):
                self._shop_visited_slots.clear()
                self._shop_last_clicked_slot = None
            if not any(k in joined for k in ("ドリンク", "ンク詳細", "破棄", "捨てる", "捨てますか", "詳細")):
                self._reward_visited_slots.clear()
            return self._parse_popup_screen(frame, lines, active_plan, meta)

        # 2. Battle turn detection (start at y=0.004h so Exam "残りターン" at top y=14 is never clipped)
        turn_box_txt = self._ocr_crop_upscaled(
            frame, (int(w * 0.008), int(h * 0.004), int(w * 0.28), int(h * 0.085)), scale=2, mode="white", pad=15
        )
        if "残りターン" not in turn_box_txt:
            exam_hdr_txt = self._ocr_crop_upscaled(
                frame, (int(w * 0.008), int(h * 0.004), int(w * 0.65), int(h * 0.090)), scale=1, mode="raw", pad=10
            )
            top_turn_lbl = self._ocr_crop_upscaled(
                frame, (int(w * 0.008), int(h * 0.004), int(w * 0.22), int(h * 0.035)), scale=2, mode="white", pad=12
            )
            turn_box_txt = f"{turn_box_txt} {exam_hdr_txt} {top_turn_lbl}".strip()
        sk_x1, sk_y1, sk_x2, sk_y2 = int(w * 0.875), int(h * 0.675), min(w, int(w * 0.985)), int(h * 0.728)
        sk_p = frame[sk_y1:sk_y2, sk_x1:sk_x2]
        sk_white = float(((sk_p[:, :, 0] > 220) & (sk_p[:, :, 1] > 220) & (sk_p[:, :, 2] > 220)).mean()) if sk_p.size else 0.0
        has_battle_skip_pill = False
        if not dim_top_backdrop and sk_white > 0.25:
            sk_txt = self._ocr_crop_upscaled(frame, (sk_x1, sk_y1 + int(h * 0.025), sk_x2, sk_y2), scale=2, use_en=True, mode="dark_on_light", pad=15)
            has_battle_skip_pill = ("SKIP" in sk_txt.upper()) or any(
                "SKIP" in (ln.get("text") or "").upper() and ln["bbox"][0] >= int(w * 0.80) and ln["bbox"][1] >= int(h * 0.64)
                for ln in lines
            )
        norm_joined = _normalize_katakana_ocr(joined)
        has_top_shop_submodal_pill = any(
            any(k in ln["text"] for k in ("削除", "強化", "チェンジ", "カスタマイズ"))
            and len(ln["text"].strip()) <= 8
            and ln["bbox"][1] < int(h * 0.12)
            and ln["bbox"][0] < int(w * 0.35)
            for ln in lines
        )
        has_bottom_cancel_pill = any(
            ("キャンセル" in ln["text"] or "もどる" in ln["text"])
            and ln["bbox"][1] > int(h * 0.80)
            for ln in lines
        )
        has_shop_exit_btn = any(
            len(ln["text"].strip()) <= 6
            and (
                ln["text"].strip() == "終了"
                or re.search(r"(?<!ターン)終了(?!時)", ln["text"].strip()) is not None
            )
            for ln in lines
        )
        is_shop_submodal_grid = (
            has_top_shop_submodal_pill
            and has_bottom_cancel_pill
            and ("リフレッシュ" not in joined)
            and ("Pポイントと交換" not in joined)
            and (not has_shop_exit_btn)
        )
        has_move_to_hand_btn = (
            (
                any(
                    ln["text"].strip().lstrip("◆◇✦✓✔ノ・") in ("移動", "移動する", "手札に移動")
                    and ln["bbox"][1] < int(h * 0.12)
                    and ln["bbox"][0] < int(w * 0.35)
                    for ln in lines
                )
                or any(
                    ln["text"].strip().lstrip("◆◇✦✓✔ノ・") in ("移動", "移動する")
                    and ln["bbox"][1] > int(h * 0.74)
                    for ln in lines
                )
            )
            and (not is_home_screen)
            and (not is_produce_setup)
            and ("移動中" not in norm_joined)
        )
        is_reward_or_change_modal = is_shop_submodal_grid or has_move_to_hand_btn or any(
            k in norm_joined
            for k in (
                "スキルカードを選",
                "チェンジで獲得",
                "チェンジする",
                "手放すスキルカード",
                "スキルカードが選択されていません",
                "カスタマイズするスキルカード",
                "カスタマイズ可能スキルカード",
                "カスタマイズメニュー",
                "カスタマイズを選",
                "カスタマイズする",
                "手札に移動",
                "移動するスキルカード",
                "山札か捨札",
                "ドリンクを選",
                "ドリンク獲得",
                "Pドリンク所持上限",
                "受け取らない",
                "受け取る",
                "強化する",
                "強化確認",
                "削除する",
                "再抽選",
            )
        )
        is_customize_menu_screen = (
            any(k in norm_joined for k in ("カスタマイズメニュー", "メニューを選択", "ーユーを選択"))
            and any(k in norm_joined for k in ("メニューを選択", "ーユーを選択", "一覧に戻る", "実行する", "合計あと"))
            and ("カスタマイズ可能スキルカード" not in norm_joined)
        )
        is_schedule_home_hud = (
            ("行動を選択" in joined)
            or ("P手帳" in joined)
            or ("Pポイントがありますね" in joined)
            or ("先生に相談して" in joined)
            or any(
                "休む" in ln["text"] and ln["bbox"][1] > int(h * 0.65) and ln["bbox"][0] > int(w * 0.70)
                for ln in lines
            )
        )
        is_non_shop_modal_action = any(
            rk in norm_joined
            for rk in (
                "受け取る",
                "受け取らない",
                "チェンジで獲得",
                "チェンジする",
                "手放すスキルカード",
                "再抽選",
                "スキルカード獲得",
                "Pドリンク獲得",
                "ドリンク獲得",
                "削除する",
                "強化する",
                "カスタマイズする",
            )
        )
        is_shop_like = (
            not is_customize_menu_screen
            and not is_shop_submodal_grid
            and not is_schedule_home_hud
            and not is_non_shop_modal_action
            and (
                ("Pポイントと交換" in joined)
                or ("Pポイントで利用するもの" in joined)
                or ("強化や削除ができます" in joined)
                or ("スキルカードの強化や削除" in joined)
                or ("リフレッシュ" in joined and any(k in joined for k in ("強化", "削除", "チェンジ", "カスタマイズ", "終了", "交換")))
                or ("交換" in joined and ("強化" in joined or "削除" in joined or "終了" in joined or "リフレッシュ" in joined))
                or ("チェンジ" in norm_joined and any(k in joined for k in ("強化", "終了", "Pポイント", "カスタマイズ")))
                or ("カスタマイズ" in norm_joined and "終了" in joined)
                or (
                    ("相談" in joined or "相談" in hdr_label or "インターバル" in joined or "インターバル" in hdr_label)
                    and any(k in joined for k in ("購入", "強化", "削除", "Pポイント", "交換", "終了", "次へ", "チェンジ", "カスタマイズ"))
                )
            )
        )
        has_turn_ui = (
            ("残りタ" in turn_box_txt)
            or bool(re.search(r"残り.{0,3}タ", turn_box_txt))
            or ("あと" in turn_box_txt and "ターン" in turn_box_txt)
            or bool(re.search(r"あと\s*\d+\s*ターン", joined))
            or ("残りタ" in joined)
            or bool(re.search(r"ターン終了(?!時)", joined))
            or ("もう一枚使用" in joined)
            or ("スキルカード使用数" in joined and "審査基準" not in joined and "追加+1" not in joined)
            or has_battle_skip_pill
            or ("ターゲット" in joined and "ターン" in joined)
            or (
                "ターン" in turn_box_txt
                and any(ak in f"{turn_box_txt} {joined}" for ak in ("ボーカル", "ポーカル", "ホーカル", "オーカル", "ダンス", "ビジュアル"))
                and self._estimate_exam_stamina_from_bar(frame, 44) is not None
            )
        ) and (not is_reward_or_change_modal) and (not is_shop_like) and (not is_customize_menu_screen)
        short_schedule_tokens = (
            "授業",
            "公開レッスン",
            "活動支給",
            "おでかけ",
            "差し入れ",
            "特別指導",
            "インターバル",
            "相談",
            "休む",
            "追い込みレッスン",
            "ボーカルレッスン",
            "ダンスレッスン",
            "ビジュアルレッスン",
            "VO.レッスン",
            "Vo.レッスン",
            "Da.レッスン",
            "Vi.レッスン",
            "Ⅵ.レッスン",
        )
        is_produce_home = (
            ("行動を選択" in joined)
            or ("を選択してください" in joined and not is_reward_or_change_modal and not is_shop_like)
            or any(
                (ln["bbox"][1] > int(h * 0.68) and len(ln["text"]) <= 18 and any(tok in ln["text"] for tok in short_schedule_tokens))
                for ln in lines
            )
        )

        if has_turn_ui and not is_produce_home:
            self._shop_visited_slots.clear()
            self._shop_last_clicked_slot = None
            self._reward_visited_slots.clear()
            return self._parse_battle_screen(frame, lines, active_plan, meta, turn_box_txt=turn_box_txt)

        # Guard against 授業/おでかけ option tooltips mentioning "スキルカードを選択...セレクトチェンジ"
        has_orange_reward_confirm = False
        if (
            (not is_shop_like)
            and (not is_produce_home)
            and (not is_customize_menu_screen)
            and not any(
                k in norm_joined
                for k in ("受け取る", "チェンジする", "強化する", "削除する", "カスタマイズする", "移動する")
            )
        ):
            orange_reward_btn = self._ocr_bottom_orange_button(frame, w, h)
            oc_reward_name = str((orange_reward_btn or {}).get("name") or "").strip().lstrip("ノ")
            has_orange_reward_confirm = (
                oc_reward_name
                in (
                    "受け取る",
                    "チェンジする",
                    "チェンジ",
                    "強化する",
                    "削除する",
                    "カスタマイズする",
                    "移動する",
                    "移動",
                    "受け取らない",
                )
                or (
                    oc_reward_name == "決定"
                    and any(
                        k in norm_joined
                        for k in ("スキルカード", "重複不可", "レッスン中1回", "再抽選", "ドリンク")
                    )
                )
                or (
                    oc_reward_name == "次へ"
                    and any(
                        k in norm_joined
                        for k in (
                            "チェンジで獲得",
                            "手放すスキルカード",
                            "を選んでください",
                            "を選択してください",
                            "重複不可",
                            "レッスン中1回",
                            "再抽選",
                        )
                    )
                )
            )
        is_event_tooltip_hud = (
            ("審査基準" in joined or "授業" in hdr_label or "おでかけ" in hdr_label)
            and (not has_orange_reward_confirm)
            and not any(
                k in norm_joined
                for k in (
                    "受け取る",
                    "決定",
                    "チェンジする",
                    "チェンジで獲得",
                    "カスタマイズする",
                    "削除する",
                    "強化する",
                    "重複不可",
                    "レッスン中1回",
                    "再抽選",
                    "キャンセル",
                    "受け取らない",
                    "スキルカードが選択されていません",
                    "スキルカードを選んで",
                    "を選んでください",
                    "を選択してください",
                )
            )
        )

        # 3. Card / Drink reward or Change / Customize selection
        if is_customize_menu_screen:
            self._in_battle_scene = False
            self._non_battle_streak = 0
            return self._parse_customize_menu_screen(frame, lines, active_plan, meta)

        if (not is_shop_like) and (not is_event_tooltip_hud) and (
            has_orange_reward_confirm
            or is_shop_submodal_grid
            or "スキルカードを選択" in norm_joined
            or "スキルカード獲得" in norm_joined
            or "スキルカードを選" in norm_joined
            or "受け取るスキルカード" in norm_joined
            or "チェンジで獲得" in norm_joined
            or "チェンジする" in norm_joined
            or "手放すスキルカード" in norm_joined
            or "スキルカードが選択されていません" in norm_joined
            or "スキルカートが選択されていません" in norm_joined
            or ("が選択されていません" in norm_joined and "スキル" in norm_joined)
            or "カスタマイズするスキルカード" in norm_joined
            or "カスタマイズ可能スキルカード" in norm_joined
            or "カスタマイズを選" in norm_joined
            or ("カスタマイズ" in norm_joined and "終了" not in norm_joined and "リフレッシュ" not in norm_joined and "交換" not in norm_joined)
            or ("セレクトチェンジ" in norm_joined and "終了" not in norm_joined and "リフレッシュ" not in norm_joined and "交換" not in norm_joined)
            or ("特別指導" in norm_joined and "プロデュース" not in norm_joined and "終了" not in norm_joined)
            or "Pドリンクを選択" in norm_joined
            or "Pドリンク獲得" in norm_joined
            or "Pドリンクを選" in norm_joined
            or "ドリンクを選" in norm_joined
            or "ドリンク獲得" in norm_joined
            or "受け取らない" in norm_joined
            or "Pドリンク所持上限" in norm_joined
            or "受け取る" in norm_joined
            or "強化確認" in norm_joined
            or (
                not is_shop_like
                and any(("強化" in ln["text"]) and len(ln["text"].strip()) <= 6 and ln["bbox"][1] < int(h * 0.12) for ln in lines)
            )
        ):
            was_in_battle = bool(self._in_battle_scene)
            has_out_of_battle_reward_cue = any(
                k in norm_joined
                for k in (
                    "受け取る",
                    "受け取らない",
                    "再抽選",
                    "スキルカード獲得",
                    "Pドリンク獲得",
                    "ドリンク獲得",
                    "Pドリンク所持上限",
                    "チェンジで獲得",
                    "チェンジする",
                    "手放すスキルカード",
                    "削除するスキルカード",
                    "強化するスキルカード",
                    "カスタマイズ",
                    "セレクトチェンジ",
                    "特別指導",
                    "CLEAR",
                    "PERFECT",
                    "報酬",
                )
            )
            if not (was_in_battle and not has_out_of_battle_reward_cue):
                self._in_battle_scene = False
            self._non_battle_streak = 0
            if not any(k in joined for k in ("強化", "削除", "カスタマイズ", "チェンジ")):
                self._shop_visited_slots.clear()
                self._shop_last_clicked_slot = None
            return self._parse_card_reward_screen(frame, lines, active_plan, meta)

        # 4. Shop / Consultation / H.I.F Interval ("相談" / "インターバル" / "Pポイントと交換" subscreen)
        if is_shop_like:
            self._in_battle_scene = False
            self._non_battle_streak = 0
            self._reward_visited_slots.clear()
            return self._parse_shop_screen(frame, lines, active_plan, meta)

        # 5. Produce home schedule
        if is_produce_home:
            self._in_battle_scene = False
            self._non_battle_streak = 0
            self._shop_visited_slots.clear()
            self._shop_last_clicked_slot = None
            self._reward_visited_slots.clear()
            return self._parse_produce_screen(frame, lines, active_plan, meta)

        # 6. Produce event choice screen (guarded against mid-battle modal & Home HUD fallthrough!)
        has_home_only_hud = ("PLV" in joined_upper) or any(
            hk in joined for hk in ("ガシャ", "コンテスト", "っンテスト", "コミュ", "活動費", "お仕事", "セール中", "プレゼント")
        )
        if (not self._in_battle_scene) and (not has_home_only_hud) and not any(tok in joined for tok in ("TrueEnd", "True End", "TAPで次へ", "アチーブメント", "最終評価", "総合評価", "メモリー")):
            pill_lines = self._find_event_choice_pills(frame, lines, w, h)
            if len(pill_lines) >= 2 or (is_event_tooltip_hud and len(pill_lines) >= 1):
                self._in_battle_scene = False
                self._non_battle_streak = 0
                self._shop_visited_slots.clear()
                self._shop_last_clicked_slot = None
                self._reward_visited_slots.clear()
                return self._parse_produce_event_screen(frame, lines, pill_lines, active_plan, meta)

        # 7. Fallback popup / cutscene / confirm / skip (debounce transient mid-battle transition frames)
        if self._in_battle_scene:
            self._non_battle_streak += 1
            if self._non_battle_streak >= 3 or any(
                k in joined for k in ("CLEAR", "PERFECT", "結果", "順位", "獲得", "次へ", "完了", "インターバル", "ショップ")
            ):
                self._in_battle_scene = False
                self._non_battle_streak = 0
        else:
            self._non_battle_streak = 0
        self._shop_visited_slots.clear()
        self._shop_last_clicked_slot = None
        self._reward_visited_slots.clear()
        return self._parse_generic_screen(frame, lines, active_plan, meta)

    def _split_line_clusters(self, lines: List[Dict[str, Any]], max_gap: int = 32) -> List[Dict[str, Any]]:
        """Split horizontal OCR lines whose words are separated by gaps (e.g. adjacent card labels)."""
        clusters: List[Dict[str, Any]] = []
        for ln in lines:
            words = ln.get("words") or []
            if len(words) <= 1:
                clusters.append(ln)
                continue
            cur_words = [words[0]]
            for w_tok in words[1:]:
                prev_x2 = cur_words[-1]["bbox"][2]
                cur_x1 = w_tok["bbox"][0]
                if cur_x1 - prev_x2 > max_gap:
                    clusters.append({
                        "text": "".join(x["text"] for x in cur_words),
                        "bbox": [
                            min(x["bbox"][0] for x in cur_words),
                            min(x["bbox"][1] for x in cur_words),
                            max(x["bbox"][2] for x in cur_words),
                            max(x["bbox"][3] for x in cur_words),
                        ],
                    })
                    cur_words = [w_tok]
                else:
                    cur_words.append(w_tok)
            clusters.append({
                "text": "".join(x["text"] for x in cur_words),
                "bbox": [
                    min(x["bbox"][0] for x in cur_words),
                    min(x["bbox"][1] for x in cur_words),
                    max(x["bbox"][2] for x in cur_words),
                    max(x["bbox"][3] for x in cur_words),
                ],
            })
        return clusters

    def _find_keyword_options(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], w: int, h: int
    ) -> List[Dict[str, Any]]:
        """Collect prominent clickable schedule buttons based on Japanese OCR clusters + bottom dock strip."""
        options: List[Dict[str, Any]] = []

        def _norm_label(txt: str) -> str:
            return (
                txt.replace("Ⅵ", "Vi")
                .replace("VI", "Vi")
                .replace("VO", "Vo")
                .replace("DA", "Da")
                .replace(" ", "")
            )

        # Combine full-frame word clusters + dedicated bottom dock strip (both raw and dark-on-white footer mask)
        clusters = self._split_line_clusters(lines, max_gap=28)
        sy1, sx1 = int(h * 0.72), int(w * 0.03)
        sy2, sx2 = int(h * 0.945), int(w * 0.97)
        strip = frame[sy1:sy2, sx1:sx2]
        if strip.size > 0:
            gray = np.mean(strip[:, :, :3], axis=2)
            bw = np.where((gray < 185)[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
            for strip_res in (self.ocr.recognize_array(strip), self.ocr.recognize_array(bw)):
                for sc in self._split_line_clusters(strip_res.get("lines", []), max_gap=25):
                    bx1, by1, bx2, by2 = sc["bbox"]
                    clusters.append({
                        "text": sc["text"],
                        "bbox": [bx1 + sx1, by1 + sy1, bx2 + sx1, by2 + sy1],
                    })

        # Specific training & schedule cards (check Vo/Da/Vi/追い込み/公開レッスン BEFORE generic レッスン)
        specific_keywords = [
            ("Vo.レッスン", "Vocal(Vo)属性训练レッスン", (0, -110)),
            ("ボーカルレッスン", "Vocal(Vo)属性提升训练", (0, -110)),
            ("Da.レッスン", "Dance(Da)属性训练レッスン", (0, -110)),
            ("ダンスレッスン", "Dance(Da)属性提升训练", (0, -110)),
            ("Vi.レッスン", "Visual(Vi)属性训练レッスン", (0, -110)),
            ("ビジュアルレッスン", "Visual(Vi)属性提升训练", (0, -110)),
            ("追い込みレッスン", "追い込みレッスン（阶段考核前冲刺实训战）", (0, -90)),
            ("公開レッスン", "公开训练（H.I.F属性大幅提升/自动进行）", (0, -90)),
            ("差し入れ", "差入补给（回复体力/获取饮料/P点）", (0, -60)),
            ("特別指導", "特别指导（强化或自定义卡牌）", (0, -60)),
            ("インターバル", "中场休整商店（消耗P点购卡/买饮料/强化）", (0, -60)),
            ("授業", "课堂授业（提升属性/拿卡）", (0, -60)),
            ("活動支給", "活动支给（获得P点/饮料）", (0, -60)),
            ("おでかけ", "外出（恢复体力/触发事件）", (0, -60)),
            ("相談", "咨询商店（购卡/饮料/强化）", (0, -60)),
            ("休む", "休息恢复体力", (0, -40)),
        ]
        multi_tile_keywords = {"授業", "公開レッスン", "レッスン"}

        def _classify_tile_attr(cx: int, by1: int) -> Tuple[str, str]:
            t_y1, t_y2 = max(0, by1 - int(h * 0.046)), max(0, by1 - int(h * 0.014))
            t_x1, t_x2 = max(0, cx + int(w * 0.015)), min(w, cx + int(w * 0.095))
            if t_y2 > t_y1 + 4 and t_x2 > t_x1 + 8:
                badge = frame[t_y1:t_y2, t_x1:t_x2]
                ir = badge[:, :, 0].astype(np.int16)
                ig = badge[:, :, 1].astype(np.int16)
                ib = badge[:, :, 2].astype(np.int16)
                vo_px = int(((ir > 205) & (ig < 135) & (ib > 115) & ((ir - ig) > 75)).sum())
                da_px = int(((ib > 195) & (ir < 135) & ((ib - ir) > 65)).sum())
                vi_px = int(((ir > 210) & (ig > 150) & (ib < 115) & ((ir - ib) > 95)).sum())
                if vo_px >= 120 and vo_px >= vi_px:
                    return "Vo.", "Vocal(Vo)属性提升"
                if vi_px >= 120 and vi_px > vo_px:
                    return "Vi.", "Visual(Vi)属性提升"
                mr = float(badge[:, :, 0].mean())
                mg = float(badge[:, :, 1].mean())
                mb = float(badge[:, :, 2].mean())
                if mr > 190 and mg < 165 and mb > 145 and (mr - mg) > 45:
                    return "Vo.", "Vocal(Vo)属性提升"
                if (mb > 195 and mg > 145 and mr < 145 and (mb - mr) > 70) or da_px >= 250:
                    return "Da.", "Dance(Da)属性提升"
                if mr > 205 and mg > 175 and mb < 165 and (mr - mb) > 65:
                    return "Vi.", "Visual(Vi)属性提升"
            return "", ""

        _sp_text_pat = re.compile(r"(?<![A-Z])(?:SP|ＳＰ)(?![A-Z])")
        _sp_crop_pat = re.compile(r"(?<![A-Z0-9])(?:SP|ＳＰ)(?![A-Z])")

        def _is_sp_tile(
            cx: int,
            by1: int,
            norm_txt: str,
            *,
            allow_ocr_fallback: bool = False,
        ) -> bool:
            up_txt = norm_txt.upper()
            if "SKIP" not in up_txt and _sp_text_pat.search(up_txt):
                return True
            sp_x1, sp_x2 = cx - int(w * 0.130), cx + int(w * 0.110)
            sp_y1, sp_y2 = by1 - int(h * 0.155), by1 - int(h * 0.025)
            for src_ln in lines + clusters:
                stxt = str(src_ln.get("text") or "").upper()
                if not stxt or "SKIP" in stxt or not _sp_text_pat.search(stxt):
                    continue
                sb_box = src_ln["bbox"]
                lcx = (sb_box[0] + sb_box[2]) // 2
                lcy = (sb_box[1] + sb_box[3]) // 2
                if sp_x1 <= lcx <= sp_x2 and sp_y1 <= lcy <= sp_y2:
                    return True
            sy_p1, sy_p2 = max(0, by1 - int(h * 0.145)), max(1, by1 - int(h * 0.045))
            sx_p1, sx_p2 = max(0, cx - int(w * 0.120)), min(w, cx + int(w * 0.100))
            if sy_p2 > sy_p1 + 4 and sx_p2 > sx_p1 + 8:
                sp_patch = frame[sy_p1:sy_p2, sx_p1:sx_p2]
                if sp_patch.size > 0:
                    sr = sp_patch[:, :, 0]
                    sg = sp_patch[:, :, 1]
                    sb = sp_patch[:, :, 2]
                    gold_px = int(((sr > 225) & (sg > 165) & (sg < 235) & (sb < 95)).sum())
                    mag_px = int(((sr > 215) & (sg < 115) & (sb > 175)).sum())
                    if gold_px >= 45 and mag_px >= 25:
                        return True
                    if h >= 400 and w >= 300 and (gold_px >= 25 or mag_px >= 25 or allow_ocr_fallback):
                        ocr_box = (sx_p1, sy_p1, sx_p2, max(sy_p1 + 6, by1 - int(h * 0.040)))
                        sp_up = self._ocr_crop_upscaled(
                            frame,
                            ocr_box,
                            scale=2,
                            use_en=True,
                            mode="white",
                            pad=10,
                        )
                        up_w = sp_up.upper()
                        if "SKIP" not in up_w and _sp_crop_pat.search(up_w):
                            return True
                        if allow_ocr_fallback:
                            sp_raw = self._ocr_crop_upscaled(
                                frame,
                                ocr_box,
                                scale=2,
                                use_en=True,
                                mode="raw",
                                pad=10,
                            )
                            up_r = sp_raw.upper()
                            if "SKIP" not in up_r and _sp_crop_pat.search(up_r):
                                return True
            return False

        def _is_dup_tile(cx: int, cy: int) -> bool:
            for existing in options:
                eb = existing["bbox"]
                ecx = (eb["xmin"] + eb["xmax"]) // 2
                ecy = (eb["ymin"] + eb["ymax"]) // 2
                if abs(cx - ecx) < max(65, int(w * 0.118)) and abs(cy - ecy) < max(70, int(h * 0.056)):
                    return True
            return False

        lesson_by1_map: List[int] = []

        def _scan_keywords(cluster_list: List[Dict[str, Any]], seen_set: set) -> None:
            for kw, desc, (dx, dy) in specific_keywords:
                is_multi = kw in multi_tile_keywords
                if not is_multi and kw in seen_set:
                    continue
                for ln in cluster_list:
                    norm_txt = _norm_label(ln["text"])
                    matched_kw = (
                        (kw in norm_txt)
                        or (kw == "おでかけ" and bool(re.search(r"お.{0,3}け", norm_txt)))
                        or (kw == "相談" and bool(re.search(r"相.{0,2}談", norm_txt)))
                    )
                    if matched_kw and len(norm_txt) <= 18 and int(h * 0.66) < ln["bbox"][1] < int(h * 0.915):
                        bx1, by1, bx2, by2 = ln["bbox"]
                        cx = (bx1 + bx2) // 2
                        xmin = max(10, bx1 - 15)
                        xmax = min(w - 10, bx2 + 15)
                        ymin = max(10, by1 + dy - 20)
                        ymax = min(h - 10, by2 + 15)
                        cy = (ymin + ymax) // 2
                        if _is_dup_tile(cx, cy):
                            continue
                        opt_title = kw
                        opt_desc = desc
                        is_sp = False
                        if is_multi:
                            lesson_by1_map.append(by1)
                            if norm_txt.startswith(("Vo.", "Da.", "Vi.")):
                                attr_pfx = norm_txt[:3]
                                attr_note = {
                                    "Vo.": "Vocal(Vo)属性提升",
                                    "Da.": "Dance(Da)属性提升",
                                    "Vi.": "Visual(Vi)属性提升",
                                }[attr_pfx]
                            else:
                                attr_pfx, attr_note = _classify_tile_attr(cx, by1)
                            is_sp = _is_sp_tile(
                                cx,
                                by1,
                                norm_txt,
                                allow_ocr_fallback=(kw in ("公開レッスン", "レッスン")),
                            )
                            sp_tok = "SP" if is_sp else ""
                            opt_title = f"{attr_pfx}{sp_tok}{kw}"
                            if is_sp:
                                opt_desc = f"【🌟SP発生·最優先必選！】{opt_title}（{attr_note or desc}·SP属性与奖励大幅倍增，无条件优先于普通课！）"
                            elif attr_pfx:
                                opt_desc = f"{opt_title}（{attr_note}）"
                        seen_set.add(kw)
                        options.append({
                            "choice_id": len(options),
                            "name": opt_title,
                            "title": opt_title,
                            **({"is_sp": True} if is_sp else {}),
                            "description": opt_desc,
                            "bbox": BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax).model_dump(),
                        })
                        if not is_multi:
                            break

        def _incomplete_lessons() -> bool:
            lesson_cnt = sum(
                1
                for o in options
                if any(k in o["name"] for k in ("授業", "公開レッスン", "レッスン")) and "追い込み" not in o["name"]
            )
            return 0 < lesson_cnt < 3

        def _needs_soudan_scan() -> bool:
            has_lesson = any(
                any(k in o["name"] for k in ("授業", "公開レッスン", "レッスン"))
                for o in options
            )
            return (not has_lesson) and ("相談" not in seen)

        seen = set()
        _scan_keywords(clusters, seen)
        if (len(options) < 2 or _incomplete_lessons() or _needs_soudan_scan()) and strip.size > 0:
            # White-text mask (2x + pad15) catches white-on-cyan SELECT tiles (e.g. おでかけ) and pale tiles (e.g. 相談)
            from PIL import ImageOps
            wm = (strip[:, :, 0] > 212) & (strip[:, :, 1] > 212) & (strip[:, :, 2] > 212)
            wbw = np.where(wm[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
            w2 = Image.fromarray(wbw).resize((wbw.shape[1] * 2, wbw.shape[0] * 2), Image.Resampling.BICUBIC)
            w_res = self.ocr.recognize_array(np.array(ImageOps.expand(w2, 15, (255, 255, 255))))
            white_clusters: List[Dict[str, Any]] = []
            for sc in self._split_line_clusters(w_res.get("lines", []), max_gap=35):
                bx1, by1, bx2, by2 = sc["bbox"]
                white_clusters.append({
                    "text": sc["text"],
                    "bbox": [
                        sx1 + max(0, (bx1 - 15) // 2),
                        sy1 + max(0, (by1 - 15) // 2),
                        sx1 + max(0, (bx2 - 15) // 2),
                        sy1 + max(0, (by2 - 15) // 2),
                    ],
                })
            _scan_keywords(white_clusters, seen)
        if (len(options) <= 1 or _incomplete_lessons() or _needs_soudan_scan()) and strip.size > 0:
            # Upscaled 2x raw OCR pass catches Day 6 相談 / おでかけ tiles and rainbow SP lesson tiles
            from PIL import ImageOps
            s2 = Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.Resampling.BICUBIC)
            s_res = self.ocr.recognize_array(np.array(ImageOps.expand(s2, 15, (255, 255, 255))))
            up_clusters: List[Dict[str, Any]] = []
            for sc in self._split_line_clusters(s_res.get("lines", []), max_gap=35):
                bx1, by1, bx2, by2 = sc["bbox"]
                up_clusters.append({
                    "text": sc["text"],
                    "bbox": [
                        sx1 + max(0, (bx1 - 15) // 2),
                        sy1 + max(0, (by1 - 15) // 2),
                        sx1 + max(0, (bx2 - 15) // 2),
                        sy1 + max(0, (by2 - 15) // 2),
                    ],
                })
            _scan_keywords(up_clusters, seen)
        if _needs_soudan_scan() and h >= 400 and w >= 300:
            # Focused 3x pass on bottom-center 相談 white pill when full-width strips miss it
            sd_x1, sd_y1, sd_x2, sd_y2 = int(w * 0.34), int(h * 0.845), int(w * 0.66), int(h * 0.912)
            sd_patch = frame[sd_y1:sd_y2, sd_x1:sd_x2]
            if sd_patch.size > 0:
                from PIL import ImageOps
                sd3 = Image.fromarray(sd_patch).resize((sd_patch.shape[1] * 3, sd_patch.shape[0] * 3), Image.Resampling.BICUBIC)
                sd_gray = np.mean(sd_patch[:, :, :3], axis=2)
                sd_dark = np.where((sd_gray < 145)[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                sd_dark3 = Image.fromarray(sd_dark).resize((sd_patch.shape[1] * 3, sd_patch.shape[0] * 3), Image.Resampling.BICUBIC)
                sd_clusters: List[Dict[str, Any]] = []
                for sd_img in (sd3, sd_dark3):
                    sd_res = self.ocr.recognize_array(np.array(ImageOps.expand(sd_img, 18, (255, 255, 255))))
                    for sc in self._split_line_clusters(sd_res.get("lines", []), max_gap=40):
                        bx1, by1, bx2, by2 = sc["bbox"]
                        sd_clusters.append({
                            "text": sc["text"],
                            "bbox": [
                                sd_x1 + max(0, (bx1 - 18) // 3),
                                sd_y1 + max(0, (by1 - 18) // 3),
                                sd_x1 + max(0, (bx2 - 18) // 3),
                                sd_y1 + max(0, (by2 - 18) // 3),
                            ],
                        })
                _scan_keywords(sd_clusters, seen)
        if _incomplete_lessons() and h >= 400 and w >= 300:
            lesson_opts = [
                o
                for o in options
                if any(k in o["name"] for k in ("授業", "公開レッスン", "レッスン")) and "追い込み" not in o["name"]
            ]
            base_kw = next(
                (ck for ck in ("公開レッスン", "授業", "レッスン") if any(ck in o["name"] for o in lesson_opts)),
                None,
            )
            if base_kw and lesson_opts:
                ymin_ref = int(np.median([o["bbox"]["ymin"] for o in lesson_opts]))
                ymax_ref = int(np.median([o["bbox"]["ymax"] for o in lesson_opts]))
                by1_ref = (
                    int(np.median(lesson_by1_map))
                    if lesson_by1_map
                    else max(int(h * 0.78), ymax_ref - int(h * 0.030))
                )
                has_same_row_rest = any(
                    o["name"] == "休む"
                    and o["bbox"]["xmin"] > int(w * 0.68)
                    and o["bbox"]["ymax"] >= ymin_ref + int(h * 0.010)
                    for o in options
                )
                if has_same_row_rest:
                    slot_specs = (
                        ("Vo.", "Vocal(Vo)属性提升", int(w * 0.165)),
                        ("Da.", "Dance(Da)属性提升", int(w * 0.385)),
                        ("Vi.", "Visual(Vi)属性提升", int(w * 0.605)),
                    )
                    slot_hw = int(w * 0.088)
                else:
                    vo_cx, da_cx, vi_cx = int(w * 0.240), int(w * 0.500), int(w * 0.760)
                    detected_cx: Dict[str, int] = {}
                    for lo in lesson_opts:
                        for ap in ("Vo.", "Da.", "Vi."):
                            if lo["name"].startswith(ap):
                                lb = lo["bbox"]
                                detected_cx[ap] = (lb["xmin"] + lb["xmax"]) // 2
                    if "Vo." in detected_cx and "Vi." in detected_cx and "Da." not in detected_cx:
                        da_cx = (detected_cx["Vo."] + detected_cx["Vi."]) // 2
                    elif "Da." in detected_cx and "Vi." in detected_cx and "Vo." not in detected_cx:
                        vo_cx = max(int(w * 0.12), 2 * detected_cx["Da."] - detected_cx["Vi."])
                    elif "Vo." in detected_cx and "Da." in detected_cx and "Vi." not in detected_cx:
                        vi_cx = min(int(w * 0.88), 2 * detected_cx["Da."] - detected_cx["Vo."])
                    elif len(lesson_opts) == 2:
                        cxs = sorted((o["bbox"]["xmin"] + o["bbox"]["xmax"]) // 2 for o in lesson_opts)
                        if (cxs[1] - cxs[0]) >= int(w * 0.36):
                            da_cx = (cxs[0] + cxs[1]) // 2
                    slot_specs = (
                        ("Vo.", "Vocal(Vo)属性提升", vo_cx),
                        ("Da.", "Dance(Da)属性提升", da_cx),
                        ("Vi.", "Visual(Vi)属性提升", vi_cx),
                    )
                    slot_hw = int(w * 0.105)
                cy_ref = (ymin_ref + ymax_ref) // 2
                for attr_pfx, attr_note, slot_cx in slot_specs:
                    if _is_dup_tile(slot_cx, cy_ref) or any(o["name"].startswith(attr_pfx) for o in lesson_opts):
                        continue
                    tile_patch = frame[
                        max(0, by1_ref - int(h * 0.095)) : min(h, by1_ref + int(h * 0.025)),
                        max(0, slot_cx - int(w * 0.075)) : min(w, slot_cx + int(w * 0.075)),
                    ]
                    if tile_patch.size > 0 and float(tile_patch.mean()) >= 145.0:
                        is_sp = _is_sp_tile(
                            slot_cx,
                            by1_ref,
                            "",
                            allow_ocr_fallback=(base_kw in ("公開レッスン", "レッスン")),
                        )
                        opt_title = f"{attr_pfx}{'SP' if is_sp else ''}{base_kw}"
                        opt_desc = (
                            f"【🌟SP発生·最優先必選！】{opt_title}（{attr_note}·SP属性与奖励大幅倍增，无条件优先于普通课！）"
                            if is_sp
                            else f"{opt_title}（{attr_note}）"
                        )
                        new_opt = {
                            "choice_id": len(options),
                            "name": opt_title,
                            "title": opt_title,
                            **({"is_sp": True} if is_sp else {}),
                            "description": opt_desc,
                            "bbox": BoundingBox(
                                xmin=max(10, slot_cx - slot_hw),
                                ymin=ymin_ref,
                                xmax=min(w - 10, slot_cx + slot_hw),
                                ymax=ymax_ref,
                            ).model_dump(),
                        }
                        options.append(new_opt)
                        lesson_opts.append(new_opt)
        if not options:
            for ln in clusters:
                norm_txt = _norm_label(ln["text"])
                if "レッスン" in norm_txt and len(norm_txt) <= 18 and ln["bbox"][1] > int(h * 0.66):
                    bx1, by1, bx2, by2 = ln["bbox"]
                    cx = (bx1 + bx2) // 2
                    ymin = max(10, by1 - 95)
                    ymax = min(h - 10, by2 + 15)
                    if _is_dup_tile(cx, (ymin + ymax) // 2):
                        continue
                    attr_pfx, attr_note = _classify_tile_attr(cx, by1)
                    is_sp = _is_sp_tile(cx, by1, norm_txt, allow_ocr_fallback=True)
                    sp_tok = "SP" if is_sp else ""
                    opt_title = f"{attr_pfx}{sp_tok}{norm_txt}" if (attr_pfx and not norm_txt.startswith(("Vo", "Da", "Vi"))) else norm_txt
                    options.append({
                        "choice_id": len(options),
                        "name": opt_title,
                        "title": opt_title,
                        **({"is_sp": True} if is_sp else {}),
                        "description": (
                            f"【🌟SP発生·最優先必選！】{opt_title}（{attr_note or '课程训练'}·SP属性与奖励大幅倍增，无条件优先于普通课！）"
                            if is_sp
                            else f"课程训练 ({opt_title}{' · ' + attr_note if attr_note else ''})"
                        ),
                        "bbox": BoundingBox(xmin=max(10, bx1 - 20), ymin=ymin, xmax=min(w - 10, bx2 + 20), ymax=ymax).model_dump(),
                    })
        options.sort(key=lambda o: o["bbox"]["xmin"])
        for ck in ("公開レッスン", "授業", "レッスン"):
            matching_lessons = [o for o in options if ck in o["name"] and "追い込み" not in o["name"]]
            if len(matching_lessons) == 3 and any(not o["name"].startswith(("Vo.", "Da.", "Vi.")) for o in matching_lessons):
                for pos_i, o in enumerate(matching_lessons):
                    attr_pfx = ("Vo.", "Da.", "Vi.")[pos_i]
                    attr_note = {
                        "Vo.": "Vocal(Vo)属性提升",
                        "Da.": "Dance(Da)属性提升",
                        "Vi.": "Visual(Vi)属性提升",
                    }[attr_pfx]
                    has_sp = bool(o.get("is_sp") or "SP" in o["name"].upper())
                    new_name = f"{attr_pfx}{'SP' if has_sp else ''}{ck}"
                    o["name"] = new_name
                    o["title"] = new_name
                    if has_sp:
                        o["is_sp"] = True
                        o["description"] = f"【🌟SP発生·最優先必選！】{new_name}（{attr_note}·SP属性与奖励大幅倍增，无条件优先于普通课！）"
                    else:
                        o["description"] = f"{new_name}（{attr_note}）"
                break
        for idx, o in enumerate(options):
            o["choice_id"] = idx
        return options

    def _parse_produce_screen(self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]) -> ScreenObservation:
        h, w = frame.shape[:2]
        options = self._find_keyword_options(frame, lines, w, h)
        scene = "produce_schedule"
        if not options:
            # Never double-tap a fallback screen as produce_schedule
            scene = "popup"
            options = self._extract_bottom_and_list_options(lines, w, h, frame=frame)

        held_drinks = self._recognize_bottom_drinks(frame, plan)
        if scene == "produce_schedule" and held_drinks:
            self._append_banned_drink_discards(
                options,
                held_drinks,
                banned_ids=self.banned_drink_ids,
                banned_names=self._banned_drink_names,
            )

        ctx = compute_week_stats_and_stage_bonus(
            scenario_id=self.scenario_id,
            week=meta["week"],
            plan=plan,
            override_stats=meta["stats"],
        )
        return ScreenObservation(
            scene_type=scene,
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=f"{meta['week_label'] or ctx['week_node'].title} | 持有P点:{meta['p_points']}",
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=meta["p_points"],
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            drinks=held_drinks,
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _classify_option_icon_and_badge(
        self, frame: np.ndarray, line_bbox: List[int], w: int, h: int
    ) -> Tuple[str, str]:
        bx1, by1, bx2, by2 = line_bbox
        cy = (by1 + by2) // 2
        # Left icon patch inside white pill
        ix1, iy1 = max(0, int(w * 0.095)), max(0, cy - 22)
        ix2, iy2 = min(w, int(w * 0.155)), min(h, cy + 22)
        attr_desc = "选项効果"
        if ix2 > ix1 and iy2 > iy1:
            patch = frame[iy1:iy2, ix1:ix2]
            non_white = patch[(patch[:, :, 0] < 235) | (patch[:, :, 1] < 235) | (patch[:, :, 2] < 235)]
            if len(non_white) > 10:
                r, g, b = non_white.mean(axis=0)
                if r > 195 and g < 150 and b > 130:
                    attr_desc = "Vocal(Vo)属性强化"
                elif b > 200 and r < 150:
                    attr_desc = "Dance(Da)属性强化"
                elif r > 200 and g > 165 and b < 150:
                    attr_desc = "Visual(Vi)属性强化"

        # Top-right dark pill stamina/P badge
        tx1, ty1 = max(0, int(w * 0.80)), max(0, by1 - 50)
        tx2, ty2 = min(w, int(w * 0.94)), max(ty1 + 5, by1)
        cost_desc = "无体力消耗 (免费)"
        if tx2 > tx1 and ty2 > ty1:
            bpatch = frame[ty1:ty2, tx1:tx2]
            dark_cnt = int(((bpatch[:, :, 0] < 95) & (bpatch[:, :, 1] < 95) & (bpatch[:, :, 2] < 95)).sum())
            if dark_cnt > 450:
                btxt = self._ocr_crop_upscaled(frame, (tx1, ty1, tx2, ty2), scale=4, use_en=True, mode="white")
                if not btxt:
                    btxt = self._ocr_crop_upscaled(frame, (max(0, tx1 - 15), max(0, ty1 - 8), min(w, tx2 + 15), min(h, ty2 + 8)), scale=4, use_en=False, mode="raw")
                m_cost = re.search(r"-\s*(\d{1,3})", btxt)
                if m_cost:
                    cost_desc = f"消耗体力 -{m_cost.group(1)}"
                else:
                    ja_btxt = self._ocr_crop_upscaled(frame, (max(0, tx1 - 95), max(0, ty1 - 8), min(w, tx2 + 15), min(h, ty2 + 8)), scale=2, use_en=False, mode="white", pad=8)
                    if any(tk in ja_btxt for tk in ("トラブル", "トラプル", "眠気", "ラブル")):
                        cost_desc = "⚠️会加入『トラブル』干扰卡"
                    else:
                        cost_desc = "消耗少量体力/资源"
        if cost_desc == "无体力消耗 (免费)" and by1 > 60:
            tr_patch = frame[max(0, by1 - int(h * 0.045)) : max(1, by1), int(w * 0.66) : min(w, int(w * 0.95))]
            if tr_patch.size > 0 and float(((tr_patch[:, :, 0] < 95) & (tr_patch[:, :, 1] < 95) & (tr_patch[:, :, 2] < 95)).sum()) >= 350:
                tr_txt = self._ocr_crop_upscaled(
                    frame,
                    (int(w * 0.66), max(0, by1 - int(h * 0.045)), min(w, int(w * 0.95)), by1),
                    scale=2,
                    mode="white",
                    pad=8,
                )
                if any(tk in tr_txt for tk in ("トラブル", "トラプル", "眠気", "ラブル")):
                    cost_desc = "⚠️会加入『トラブル』干扰卡 (无扣血)"
        return attr_desc, cost_desc

    def _find_event_choice_pills(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], w: int, h: int
    ) -> List[Dict[str, Any]]:
        """Detect vertical stack of white horizontal choice pills in 授業 / おでかけ / 活動支給 (dark or colored text)."""
        pills: List[Dict[str, Any]] = []
        # Exclude HUD labels and CLEAR/win reward summary rows ("獲得Pポイント"/"獲得アイテム" often OCR as "覆得..."/"得アイテム")
        ignore_tokens = (
            "早送り",
            "ログ",
            "AUTO",
            "オート",
            "メニュー",
            "審査基準",
            "獲得",
            "覆得",
            "得アイテム",
            "親愛度",
            "条件",
            "育成して",
            "重複不可",
            "レッスン中1回",
            "レッスン中",
            "カスタマイズ",
            "強化確認",
            "スキルカード",
            "受け取る",
            "再抽選",
            "トラブル",
            "トチェンジ",
            "セレクトチェンジ",
            "キャンペーン",
            "キャン十ン",
            "お仕事",
            "活動費",
            "イベント",
            "ショップ",
            "プロデュース",
            "プロテ",
            "課題",
            "セール中",
            "プレゼント",
            "ミッション",
            "アチーブ",
            "ホーム",
            "コンテスト",
            "ガシャ",
            "コミュ",
        )
        seen_y: List[int] = []

        def _collect_pills(cand_lines: List[Dict[str, Any]], min_x_frac: float = 0.12) -> None:
            for ln in cand_lines:
                bx1, by1, bx2, by2 = ln["bbox"]
                txt = re.sub(r"^[の◎○〇・)\]\(\[「『.,、。]+\s*", "", ln["text"]).strip()
                cy = (by1 + by2) // 2
                if any(ig in txt for ig in ignore_tokens) or len(txt) < 3 or not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", txt):
                    continue
                if int(h * 0.55) <= cy <= int(h * 0.80) and int(w * min_x_frac) <= bx1 <= int(w * 0.32):
                    bg_patch = frame[max(0, cy - 14) : min(h, cy + 14), int(w * 0.55) : int(w * 0.72)]
                    if bg_patch.size > 0 and float(bg_patch.mean()) < 120.0 and float(bg_patch.max(axis=2).mean()) < 165.0:
                        continue
                    if any(abs(cy - py) < 55 for py in seen_y):
                        continue
                    seen_y.append(cy)
                    pills.append({**ln, "text": txt})

        _collect_pills(lines)
        if len(pills) < 2 and h >= 400 and w >= 300:
            # H.I.F 授業 event pills often render bright orange/amber text on white pills, which full-frame OCR skips
            from PIL import ImageOps
            sy1, sx1 = int(h * 0.58), int(w * 0.165)
            sy2, sx2 = int(h * 0.80), int(w * 0.90)
            roi = frame[sy1:sy2, sx1:sx2]
            if roi.size > 0:
                gray = roi.mean(axis=2)
                dark_or_col = (roi[:, :, 2] < 165) | (gray < 195)
                bw = np.where(dark_or_col[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                b2 = Image.fromarray(bw).resize((bw.shape[1] * 2, bw.shape[0] * 2), Image.Resampling.BICUBIC)
                res = self.ocr.recognize_array(np.array(ImageOps.expand(b2, 15, (255, 255, 255))))
                masked_lines: List[Dict[str, Any]] = []
                for mln in res.get("lines", []):
                    mx1, my1, mx2, my2 = mln["bbox"]
                    masked_lines.append({
                        "text": mln["text"],
                        "bbox": [
                            sx1 + max(0, (mx1 - 15) // 2),
                            sy1 + max(0, (my1 - 15) // 2),
                            sx1 + max(0, (mx2 - 15) // 2),
                            sy1 + max(0, (my2 - 15) // 2),
                        ],
                    })
                _collect_pills(masked_lines, min_x_frac=0.14)
        pills.sort(key=lambda ln: (ln["bbox"][1] + ln["bbox"][3]) // 2)
        return pills

    def _parse_produce_event_screen(
        self,
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
        pill_lines: List[Dict[str, Any]],
        plan: str,
        meta: Dict[str, Any],
    ) -> ScreenObservation:
        h, w = frame.shape[:2]
        hdr = meta.get("header_label") or "培育事件"
        options: List[Dict[str, Any]] = []
        for idx, ln in enumerate(pill_lines):
            bx1, by1, bx2, by2 = ln["bbox"]
            cy = (by1 + by2) // 2
            attr_desc, cost_desc = self._classify_option_icon_and_badge(frame, ln["bbox"], w, h)
            band_trouble = any(
                any(tk in str(oln.get("text") or "") for tk in ("トラブル", "トラプル", "眠気"))
                for oln in lines
                if (by1 - int(h * 0.055)) <= ((oln["bbox"][1] + oln["bbox"][3]) // 2) <= (by2 + int(h * 0.035))
            )
            if band_trouble and ("トラブル" not in cost_desc and "眠気" not in cost_desc):
                cost_desc = f"{cost_desc} | ⚠️会加入『トラブル/眠気』干扰卡"
            options.append({
                "choice_id": idx,
                "name": ln["text"],
                "title": ln["text"],
                "description": f"{attr_desc} | {cost_desc} ({hdr}选项)",
                "bbox": BoundingBox(
                    xmin=max(40, int(w * 0.08)),
                    ymin=max(10, cy - 35),
                    xmax=min(w - 40, int(w * 0.92)),
                    ymax=min(h - 10, cy + 35),
                ).model_dump(),
            })

        ctx = compute_week_stats_and_stage_bonus(
            scenario_id=self.scenario_id,
            week=meta["week"],
            plan=plan,
            override_stats=meta["stats"],
        )
        return ScreenObservation(
            scene_type="produce_event",
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=meta["week_label"] or f"{hdr} ({ctx['week_node'].title})",
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=meta["p_points"],
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _parse_landscape_live_screen(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str
    ) -> ScreenObservation:
        """Handle landscape (w > h) True End Live Concert frames: confirm dialogs, top-right |> skip icon, or tap to reveal HUD."""
        h, w = frame.shape[:2]
        confirm_opt: Optional[Dict[str, Any]] = None
        confirm_toks = ("スキップ", "決定", "次へ", "完了", "はい", "OK")

        for tok in confirm_toks:
            for ln in lines:
                txt = ln["text"].strip()
                if (
                    tok in txt
                    and ln["bbox"][1] > int(h * 0.45)
                    and len(txt) <= 12
                    and not txt.endswith(("しますか", "?", "？"))
                ):
                    bx1, by1, bx2, by2 = ln["bbox"]
                    confirm_opt = {
                        "choice_id": 0,
                        "name": tok,
                        "title": tok,
                        "description": f"横屏Live确认按钮: {txt}",
                        "bbox": BoundingBox(
                            xmin=max(10, bx1 - 35),
                            ymin=max(10, by1 - 15),
                            xmax=min(w - 10, bx2 + 35),
                            ymax=min(h - 10, by2 + 15),
                        ).model_dump(),
                    }
                    break
            if confirm_opt is not None:
                break

        if confirm_opt is None and h >= 300 and w >= 400:
            from PIL import ImageOps

            y_off = int(h * 0.50)
            band = frame[y_off : int(h * 0.92), :]
            if band.size > 0:
                orange = (
                    (band[:, :, 0] > 195)
                    & (band[:, :, 1] > 75)
                    & (band[:, :, 1] < 195)
                    & (band[:, :, 2] < 105)
                    & ((band[:, :, 0].astype(np.int16) - band[:, :, 2].astype(np.int16)) > 115)
                )
                row_spans = self._contiguous_true_spans(np.where(orange.sum(axis=1) >= 40)[0], max_gap=8, min_len=15)
                for oy1, oy2 in row_spans:
                    sub = orange[oy1:oy2, :]
                    col_spans = self._contiguous_true_spans(np.where(sub.sum(axis=0) >= 15)[0], max_gap=12, min_len=40)
                    for ox1, ox2 in col_spans:
                        op = band[oy1:oy2, ox1:ox2]
                        bm = (op[:, :, 0] > 185) & (op[:, :, 1] > 185) & (op[:, :, 2] > 185)
                        bw = np.where(bm[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                        b2 = Image.fromarray(bw).resize((bw.shape[1] * 2, bw.shape[0] * 2), Image.Resampling.BICUBIC)
                        btxt = _clean_ocr_text(
                            self.ocr.recognize_array(np.array(ImageOps.expand(b2, 15, (255, 255, 255)))).get("text", "")
                        )
                        matched_tok = next((t for t in confirm_toks if t in btxt), None)
                        if matched_tok is None and ox1 > w // 2 - 40:
                            matched_tok = "スキップ"
                        if matched_tok is not None:
                            confirm_opt = {
                                "choice_id": 0,
                                "name": matched_tok,
                                "title": matched_tok,
                                "description": f"横屏Live橙色确认按钮: {btxt or matched_tok}",
                                "bbox": BoundingBox(
                                    xmin=ox1, ymin=y_off + oy1, xmax=ox2, ymax=y_off + oy2
                                ).model_dump(),
                            }
                            break
                    if confirm_opt is not None:
                        break

        if confirm_opt is not None:
            opts = [confirm_opt]
        else:
            tr = frame[int(h * 0.07) : int(h * 0.16), int(w * 0.93) : int(w * 0.97)]
            white_px = (
                int(((tr[:, :, 0] > 220) & (tr[:, :, 1] > 220) & (tr[:, :, 2] > 220)).sum())
                if tr.size > 0
                else 0
            )
            if tr.size > 0 and white_px >= 60:
                opts = [{
                    "choice_id": 0,
                    "name": "LIVE_SKIP",
                    "title": "跳过Live演出 (|>)",
                    "description": "点击右上角 |> 按钮跳过横屏Live演出",
                    "bbox": BoundingBox(
                        xmin=int(w * 0.938),
                        ymin=int(h * 0.085),
                        xmax=int(w * 0.965),
                        ymax=int(h * 0.140),
                    ).model_dump(),
                }]
            else:
                opts = [{
                    "choice_id": 0,
                    "name": "唤出Live控制栏",
                    "title": "点击屏幕唤出Live控制栏",
                    "description": "横屏Live点击中央显示右上角跳过按钮",
                    "bbox": BoundingBox(
                        xmin=w // 2 - 80,
                        ymin=h // 2 - 50,
                        xmax=w // 2 + 80,
                        ymax=h // 2 + 50,
                    ).model_dump(),
                }]

        ctx = compute_week_stats_and_stage_bonus(
            scenario_id=self.scenario_id,
            week=self._last_week,
            plan=plan,
            override_stats=self._last_stats,
        )
        return ScreenObservation(
            scene_type="skip",
            plan=plan,
            scenario_id=self.scenario_id,
            week=self._last_week,
            total_weeks=get_scenario_spec(self.scenario_id).total_weeks,
            week_activity=self._last_week_label or ctx["week_node"].title,
            stamina=self._last_stamina[0],
            max_stamina=self._last_stamina[1],
            p_points=self._last_p_points,
            idol_stats=dict(self._last_stats),
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            options=opts,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _detect_bottom_circle_x_close(self, frame: np.ndarray, w: int, h: int) -> Optional[Dict[str, Any]]:
        """Detect bottom-center white circular [✕] close icon (e.g. True End achievement modal)."""
        if h < 400 or w < 300:
            return None
        xc_in = frame[int(h * 0.916) : int(h * 0.952), int(w * 0.455) : int(w * 0.545)]
        xc_core = frame[int(h * 0.924) : int(h * 0.944), int(w * 0.482) : int(w * 0.518)]
        xc_flank_l = frame[int(h * 0.916) : int(h * 0.952), int(w * 0.33) : int(w * 0.41)]
        xc_flank_r = frame[int(h * 0.916) : int(h * 0.952), int(w * 0.59) : int(w * 0.67)]
        if xc_in.size == 0 or xc_core.size == 0 or xc_flank_l.size == 0 or xc_flank_r.size == 0:
            return None
        white_fill = float(((xc_in > 225).all(axis=2)).mean())
        dark_cross = float((xc_core.mean(axis=2) < 165.0).mean())
        flank_l = float(xc_flank_l.mean())
        flank_r = float(xc_flank_r.mean())
        if white_fill >= 0.55 and dark_cross >= 0.08 and flank_l < 155.0 and flank_r < 155.0:
            return {
                "choice_id": 0,
                "name": "閉じる",
                "title": "閉じる",
                "description": "底部圆形关闭按钮 [✕]",
                "bbox": BoundingBox(xmin=w // 2 - 42, ymin=int(h * 0.912), xmax=w // 2 + 42, ymax=int(h * 0.956)).model_dump(),
            }
        return None

    @staticmethod
    def _contiguous_true_spans(indices: np.ndarray, max_gap: int = 8, min_len: int = 12) -> List[Tuple[int, int]]:
        if len(indices) == 0:
            return []
        spans: List[Tuple[int, int]] = []
        start = int(indices[0])
        prev = start
        for v in indices[1:]:
            iv = int(v)
            if iv - prev > max_gap:
                if prev - start + 1 >= min_len:
                    spans.append((start, prev + 1))
                start = iv
            prev = iv
        if prev - start + 1 >= min_len:
            spans.append((start, prev + 1))
        return spans

    def _detect_drink_discard_button(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], w: int, h: int
    ) -> Optional[Dict[str, Any]]:
        """Detect foreground P-drink delete button (`捨てる`/`破棄`/`破棄する`), suppressing in-battle greyed `捨てる`, card `削除`, and abandon prompts."""
        joined = "".join(ln["text"] for ln in lines)
        if any(k in joined for k in ("再開", "続きから", "中断", "リタイア", "あきらめる", "プロデュースを", "データ削除", "タイトルへ")):
            return None

        in_battle_use = self._in_battle_scene or ("使う" in joined) or ("使用する" in joined)
        discard_toks: Tuple[str, ...] = ("破棄する", "破棄") if in_battle_use else ("捨てる", "破棄する", "破棄")

        def _make_discard_opt(tok: str, xmin: int, ymin: int, xmax: int, ymax: int) -> Dict[str, Any]:
            return {
                "choice_id": 0,
                "name": tok,
                "title": f"{tok} (确认删除该旧P饮料)",
                "description": f"点击『{tok}』删除当前查看的旧P饮料，腾出饮料栏空位",
                "bbox": BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax).model_dump(),
            }

        for tok in discard_toks:
            for ln in lines:
                txt = ln["text"].strip()
                if (
                    tok in txt
                    and ln["bbox"][1] > int(h * 0.45)
                    and len(txt) <= 12
                    and not txt.endswith(("しますか", "捨てますか", "?", "？"))
                ):
                    bx1, by1, bx2, by2 = ln["bbox"]
                    return _make_discard_opt(
                        tok,
                        max(10, bx1 - 35),
                        max(10, by1 - 18),
                        min(w - 10, bx2 + 35),
                        min(h - 10, by2 + 18),
                    )

        if h < 400 or w < 300:
            return None

        if (not in_battle_use) and any(
            k in joined for k in ("Pドリンク詳細", "ドリンク詳細", "キャンセル", "受け取らない", "捨てますか")
        ):
            y_sheet = int(h * 0.70)
            sheet_crop = frame[y_sheet : int(h * 0.96), :]
            if sheet_crop.size > 0:
                sheet_res = self.ocr.recognize_array(sheet_crop)
                for sln in sheet_res.get("lines", []):
                    stxt = sln["text"].strip()
                    for tok in discard_toks:
                        if (
                            tok in stxt
                            and len(stxt) <= 12
                            and not stxt.endswith(("しますか", "捨てますか", "?", "？"))
                        ):
                            sx1, sy1, sx2, sy2 = sln["bbox"]
                            return _make_discard_opt(
                                tok,
                                max(10, sx1 - 35),
                                max(10, y_sheet + sy1 - 18),
                                min(w - 10, sx2 + 35),
                                min(h - 10, y_sheet + sy2 + 18),
                            )

        from PIL import ImageOps

        y_off = int(h * 0.48)
        band = frame[y_off : int(h * 0.95), :]
        if band.size == 0:
            return None
        orange_mask = (
            (band[:, :, 0] > 195)
            & (band[:, :, 1] > 75)
            & (band[:, :, 1] < 195)
            & (band[:, :, 2] < 105)
            & ((band[:, :, 0].astype(np.int16) - band[:, :, 2].astype(np.int16)) > 110)
        )
        red_mask = (
            (band[:, :, 0] > 195)
            & (band[:, :, 1] < 115)
            & (band[:, :, 2] < 135)
            & ((band[:, :, 0].astype(np.int16) - band[:, :, 1].astype(np.int16)) > 85)
        )
        for color_mask in (red_mask, orange_mask):
            row_spans = self._contiguous_true_spans(np.where(color_mask.sum(axis=1) >= 25)[0], max_gap=8, min_len=15)
            for oy1, oy2 in row_spans:
                sub = color_mask[oy1:oy2, :]
                col_spans = self._contiguous_true_spans(np.where(sub.sum(axis=0) >= 12)[0], max_gap=12, min_len=35)
                for ox1, ox2 in col_spans:
                    op = band[oy1:oy2, ox1:ox2]
                    bm = (op[:, :, 0] > 185) & (op[:, :, 1] > 185) & (op[:, :, 2] > 185)
                    if int(bm.sum()) < 20:
                        continue
                    bw = np.where(bm[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                    b2 = Image.fromarray(bw).resize((bw.shape[1] * 2, bw.shape[0] * 2), Image.Resampling.BICUBIC)
                    btxt = _clean_ocr_text(self.ocr.recognize_array(np.array(ImageOps.expand(b2, 15, (255, 255, 255)))).get("text", ""))
                    for tok in discard_toks:
                        if tok in btxt:
                            return _make_discard_opt(
                                tok,
                                max(10, ox1),
                                max(10, y_off + oy1),
                                min(w - 10, ox2),
                                min(h - 10, y_off + oy2),
                            )
        return None

    def _ocr_bottom_orange_button(
        self, frame: np.ndarray, w: int, h: int, allow_discard: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Detect primary white-on-orange pill button at bottom (or mid-lower modal fallback)."""
        toks: Tuple[str, ...] = (
            ("再開する", "再開", "続きから", "プロデュース開始", "開始する", "開始", "挑戦", "使用する", "使う", "交換", "強化する", "削除する", "チェンジする", "チェンジ", "カスタマイズする", "更新する", "移動する", "移動", "捨てる", "破棄する", "破棄", "受け取らない", "スキップ", "プロデュース完了", "完了", "ホームへ", "次へ", "決定", "終了", "受け取る", "はい", "確認", "OK")
            if allow_discard
            else ("再開する", "再開", "続きから", "プロデュース開始", "開始する", "開始", "挑戦", "使用する", "使う", "交換", "強化する", "削除する", "チェンジする", "チェンジ", "カスタマイズする", "更新する", "移動する", "移動", "スキップ", "プロデュース完了", "完了", "ホームへ", "次へ", "決定", "終了", "受け取る", "はい", "確認", "OK")
        )
        for y_start_frac, y_end_frac, r_min in ((0.855, 0.965, 225), (0.73, 0.855, 195)):
            y_off = int(h * y_start_frac)
            band = frame[y_off : int(h * y_end_frac), :]
            if band.size == 0:
                continue
            orange = (
                (band[:, :, 0] > r_min)
                & (band[:, :, 1] > 75)
                & (band[:, :, 1] < 195)
                & (band[:, :, 2] < 105)
                & ((band[:, :, 0].astype(np.int16) - band[:, :, 2].astype(np.int16)) > 115)
            )
            xs_mask = orange.sum(axis=0) >= 18
            ys = np.where(orange.sum(axis=1) >= 45)[0]
            if int(xs_mask.sum()) < 40 or len(ys) < 15:
                continue
            # Take the widest contiguous horizontal orange run so stray left/right orange halos never stretch bbox across a gap
            best_run: Tuple[int, int] = (0, 0)
            cur_s: Optional[int] = None
            gap = 0
            for xi in range(len(xs_mask)):
                if xs_mask[xi]:
                    if cur_s is None:
                        cur_s = xi
                    gap = 0
                elif cur_s is not None:
                    gap += 1
                    if gap > 6:
                        run_e = xi - gap
                        if (run_e - cur_s) > (best_run[1] - best_run[0]):
                            best_run = (cur_s, run_e)
                        cur_s = None
                        gap = 0
            if cur_s is not None:
                run_e = len(xs_mask) - gap
                if (run_e - cur_s) > (best_run[1] - best_run[0]):
                    best_run = (cur_s, run_e)
            ox1, ox2 = max(0, best_run[0] + 4), min(w, best_run[1] - 4)
            oy1, oy2 = max(0, int(ys.min()) + 4), min(band.shape[0], int(ys.max()) - 4)
            min_btn_w = max(75, int(w * 0.11))
            if (ox2 - ox1) < min_btn_w or oy2 <= oy1 + 15:
                continue
            op = band[oy1:oy2, ox1:ox2]
            bm = (op[:, :, 0] > 185) & (op[:, :, 1] > 185) & (op[:, :, 2] > 185)
            bw = np.where(bm[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
            from PIL import ImageOps
            b2 = Image.fromarray(bw).resize((bw.shape[1] * 2, bw.shape[0] * 2), Image.Resampling.BICUBIC)
            btxt = _clean_ocr_text(self.ocr.recognize_array(np.array(ImageOps.expand(b2, 15, (255, 255, 255)))).get("text", ""))
            norm_btxt = _normalize_katakana_ocr(btxt)
            if "早送り" in btxt or "所持上限" in btxt or "プロデュース終了" in btxt or ("確認" in btxt and btxt.strip() != "確認"):
                continue
            matched_tok = next((tok for tok in toks if tok in norm_btxt), None)
            if matched_tok is None:
                if y_start_frac < 0.80:
                    continue
                if (ox2 - ox1) < max(110, int(w * 0.16)) or (ox2 - ox1) < int((oy2 - oy1) * 1.8):
                    continue
                matched_tok = "決定"
            return {
                "choice_id": 0,
                "name": matched_tok,
                "title": matched_tok,
                "description": f"底部橙色主按钮: {btxt or matched_tok}",
                "bbox": BoundingBox(xmin=ox1, ymin=y_off + oy1, xmax=ox2, ymax=y_off + oy2).model_dump(),
            }
        return None

    def _parse_popup_screen(
        self,
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
        plan: str,
        meta: Dict[str, Any],
        fg_discard_btn: Optional[Dict[str, Any]] = None,
    ) -> ScreenObservation:
        h, w = frame.shape[:2]
        full_joined = "".join(ln["text"] for ln in lines)
        is_retry_confirm_modal = (
            ("再挑戦確認" in full_joined)
            or ("プロデュース終了を選択した場合は失敗" in full_joined)
            or ("再挑戦が可能ですが" in full_joined)
        )
        if is_retry_confirm_modal:
            has_retry_jewel = ("ジュエル" in full_joined) or ("あと0回" in full_joined)
            retry_options: List[Dict[str, Any]] = []
            if not has_retry_jewel:
                retry_bbox: Optional[BoundingBox] = None
                for ln in lines:
                    txt = ln["text"].strip()
                    bx1, by1, bx2, by2 = ln["bbox"]
                    if (
                        "再挑戦" in txt
                        and "確認" not in txt
                        and "可能" not in txt
                        and "無料" not in txt
                        and len(txt) <= 6
                        and by1 > int(h * 0.82)
                        and bx1 > int(w * 0.45)
                    ):
                        retry_bbox = BoundingBox(
                            xmin=max(int(w * 0.50), bx1 - 65),
                            ymin=max(int(h * 0.875), by1 - 24),
                            xmax=min(int(w * 0.90), bx2 + 65),
                            ymax=min(int(h * 0.955), by2 + 24),
                        )
                        break
                if retry_bbox is None:
                    retry_bbox = BoundingBox(
                        xmin=int(w * 0.52),
                        ymin=int(h * 0.885),
                        xmax=int(w * 0.87),
                        ymax=int(h * 0.945),
                    )
                retry_options.append({
                    "choice_id": 0,
                    "name": "再挑戦",
                    "title": "再挑戦 (使用免费次数重新挑战本轮)",
                    "description": "点击右侧『再挑戦』免费重打本轮冲击更高名次（严禁点左侧プロデュース終了！）",
                    "bbox": retry_bbox.model_dump(),
                })
            else:
                retry_options.append({
                    "choice_id": 0,
                    "name": "ジュエル消耗停止",
                    "title": "ジュエル (免费再挑战次数已用完·等待人工确认)",
                    "description": "当前无免费再挑戦次数，严禁擅自点プロデュース終了或消耗ジュエル",
                    "bbox": BoundingBox(xmin=w // 2 - 40, ymin=int(h * 0.50), xmax=w // 2 + 40, ymax=int(h * 0.56)).model_dump(),
                })
            ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
            return ScreenObservation(
                scene_type="popup",
                plan=plan,
                scenario_id=self.scenario_id,
                week=meta["week"],
                total_weeks=meta["total_weeks"],
                week_activity=f"{meta['week_label'] or ctx['week_node'].title} | 持有P点:{meta['p_points']}",
                stamina=meta["stamina"],
                max_stamina=meta["max_stamina"],
                p_points=meta["p_points"],
                idol_stats=meta["stats"],
                p_items=ctx["p_items"],
                stage_bonuses=ctx["stage_bonuses"],
                options=retry_options,
                raw_text="\n".join(ln["text"] for ln in lines),
            )
        is_resume_prompt = ("再開" in full_joined or "続きから" in full_joined) and not any(
            k in full_joined for k in ("リタイアしますか", "リタイアする", "あきらめる", "破棄", "データ削除")
        )
        avoid_confirm = (not is_resume_prompt) and any(
            k in full_joined for k in ("研修案内", "基礎研修", "一時中断", "リタイア", "あきらめる")
        )
        allow_drink_discard = (not avoid_confirm) and not any(
            k in full_joined for k in ("再開", "続きから", "中断", "リタイア", "あきらめる", "プロデュースを", "データ削除", "タイトル")
        )
        if not avoid_confirm:
            if allow_drink_discard and any(k in full_joined for k in ("捨てる", "捨てますか", "破棄", "ドリンク", "受け取らない")):
                orange_btn = self._ocr_bottom_orange_button(frame, w, h, allow_discard=True)
            else:
                orange_btn = self._ocr_bottom_orange_button(frame, w, h)
        else:
            orange_btn = None

        has_use_action = (
            self._in_battle_scene
            or ("使う" in full_joined)
            or ("使用する" in full_joined)
            or (orange_btn is not None and orange_btn.get("name") in ("使う", "使用する"))
        )
        is_drink_discard_modal = allow_drink_discard and (not has_use_action) and (
            fg_discard_btn is not None or any(k in full_joined for k in ("捨てる", "捨てますか", "破棄", "Pドリンク詳細", "ドリンク詳細", "ンク詳細", "ドリンク詳"))
        )
        if avoid_confirm:
            confirm_tokens: Tuple[str, ...] = ()
        elif is_drink_discard_modal:
            confirm_tokens = ("捨てる", "破棄する", "破棄", "決定", "確認", "はい", "OK", "閉じる")
        elif any(k in full_joined for k in ("Pドリンク詳細", "ドリンク詳細", "ンク詳細", "ドリンク詳")):
            confirm_tokens = ("使用する", "使う", "決定", "確認", "はい", "OK", "閉じる")
        elif allow_drink_discard:
            base_tokens = ("再開する", "再開", "続きから", "プロデュース開始", "開始する", "レギュラー", "使用する", "使う", "交換", "強化する", "強化", "削除する", "削除", "チェンジ", "カスタマイズする", "更新する", "破棄する", "破棄", "受け取らない", "スキップ", "プロデュース完了", "完了", "ホームへ", "閉じる", "決定", "確認", "はい", "OK", "タイトルへ", "受け取る", "次へ")
            confirm_tokens = base_tokens if has_use_action else ("捨てる",) + base_tokens
        else:
            confirm_tokens = ("再開する", "再開", "続きから", "プロデュース開始", "開始する", "レギュラー", "使用する", "使う", "交換", "強化する", "強化", "削除する", "削除", "チェンジ", "カスタマイズする", "更新する", "スキップ", "プロデュース完了", "完了", "ホームへ", "閉じる", "決定", "確認", "はい", "OK", "タイトルへ", "受け取る", "次へ")
        options: List[Dict[str, Any]] = []
        seen_names = set()
        if fg_discard_btn is not None and not has_use_action:
            options.append(dict(fg_discard_btn))
            seen_names.add(fg_discard_btn["name"])
        if orange_btn is not None and orange_btn["name"] not in seen_names:
            options.append(orange_btn)
            seen_names.add(orange_btn["name"])

        for tok in confirm_tokens + ("キャンセル",):
            for ln in lines:
                txt = ln["text"].strip()
                # Ignore modal headers like "交換確認" / "最終確認" / "Pドリンク詳細" or upper body text above button row
                is_header_title = (
                    ("確認" in txt and txt != "確認")
                    or ("しますか" in txt)
                    or ("捨てますか" in txt)
                    or ("案内" in txt)
                    or ("詳細" in txt)
                    or ("スキルカード" in txt)
                    or ("Pポイント" in txt)
                )
                if (
                    tok in txt
                    and not is_header_title
                    and len(txt) <= 12
                    and ln["bbox"][1] > int(h * 0.50)
                    and tok not in seen_names
                ):
                    bx1, by1, bx2, by2 = ln["bbox"]
                    seen_names.add(tok)
                    # Center-align button horizontally if near center ("閉じる" / "タイトルへ")
                    cx = (bx1 + bx2) // 2
                    if abs(cx - w // 2) < int(w * 0.16):
                        xmin, xmax = max(20, w // 2 - 110), min(w - 20, w // 2 + 110)
                    else:
                        xmin, xmax = max(10, bx1 - 35), min(w - 10, bx2 + 35)
                    options.append({
                        "choice_id": len(options),
                        "name": tok,
                        "title": f"{tok} (确认删除该旧P饮料)" if tok in ("捨てる", "破棄", "破棄する") else tok,
                        "description": (
                            "拒绝中断/跳过研修引导，继续当前培育"
                            if (avoid_confirm and tok == "キャンセル")
                            else (
                                "恢复中断的培育对局 (再開)"
                                if tok in ("再開する", "再開", "続きから")
                                else (
                                    f"点击『{tok}』删除当前查看的旧P饮料，腾出饮料栏空位"
                                    if tok in ("捨てる", "破棄", "破棄する")
                                    else f"弹窗按钮: {ln['text']}"
                                )
                            )
                        ),
                        "bbox": BoundingBox(xmin=xmin, ymin=max(10, by1 - 20), xmax=xmax, ymax=min(h - 10, by2 + 20)).model_dump(),
                    })
                    break
        if has_use_action:
            options = [o for o in options if o.get("name") not in ("捨てる", "破棄", "破棄する")]
        if any(o.get("name") in ("捨てる", "破棄", "破棄する") for o in options) and not any(
            o.get("name") in ("閉じる", "キャンセル") for o in options
        ):
            options.append({
                "choice_id": len(options),
                "name": "閉じる",
                "title": "收起饮料详情 (不删除)",
                "description": "点击空白处关闭饮料详情，保留该旧饮料返回选择",
                "bbox": BoundingBox(xmin=int(w * 0.88), ymin=int(h * 0.22), xmax=int(w * 0.98), ymax=int(h * 0.30)).model_dump(),
            })
        for i, o in enumerate(options):
            o["choice_id"] = i
        if not options:
            # Check if dimmed "次へ" has an "おまかせ" auto-fill button on setup screen
            for ln in lines:
                if "おまかせ" in ln["text"] and ln["bbox"][1] > int(h * 0.65):
                    bx1, by1, bx2, by2 = ln["bbox"]
                    options.append({
                        "choice_id": 0,
                        "name": "おまかせ",
                        "title": "おまかせ",
                        "description": "自动编成推荐支援卡/记忆",
                        "bbox": BoundingBox(xmin=max(10, bx1 - 30), ymin=max(10, by1 - 15), xmax=min(w - 10, bx2 + 30), ymax=min(h - 10, by2 + 15)).model_dump(),
                    })
                    break
        if not options:
            circle_x_btn = self._detect_bottom_circle_x_close(frame, w, h)
            if circle_x_btn is not None:
                options.append(circle_x_btn)
        if not options:
            is_inline_tip = any(k in "".join(ln["text"] for ln in lines) for k in ("タップすると", "こちらを", "確認できます"))
            cy_band = (int(h * 0.45), int(h * 0.55)) if is_inline_tip else (int(h * 0.91), int(h * 0.96))
            options.append({
                "choice_id": 0,
                "name": "閉じる",
                "title": "閉じる",
                "description": "关闭弹窗或教程提示",
                "bbox": BoundingBox(xmin=w // 2 - 100, ymin=cy_band[0], xmax=w // 2 + 100, ymax=cy_band[1]).model_dump(),
            })

        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
        rank_score, rank_label = self._parse_rank_result_score(frame, lines, w, h)
        wk_act = f"{meta['week_label'] or ctx['week_node'].title} | 持有P点:{meta['p_points']}"
        if rank_label:
            wk_act = f"{wk_act} | {rank_label}"
        return ScreenObservation(
            scene_type="popup",
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=wk_act,
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            score=rank_score,
            p_points=meta["p_points"],
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _parse_home_screen(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]
    ) -> ScreenObservation:
        """Recognize the Home (ホーム) screen and expose the primary 'プロデュース' button."""
        h, w = frame.shape[:2]
        produce_bbox: Optional[BoundingBox] = None
        for ln in lines:
            txt = ln["text"].strip()
            bx1, by1, bx2, by2 = ln["bbox"]
            cx = (bx1 + bx2) // 2
            if ("プロデュース" in txt or "プロテ" in txt) and "終了" not in txt and by1 > int(h * 0.72) and int(w * 0.18) <= cx <= int(w * 0.82):
                produce_bbox = BoundingBox(
                    xmin=max(10, bx1 - 55),
                    ymin=max(10, by1 - 24),
                    xmax=min(w - 10, bx2 + 55),
                    ymax=min(h - 10, by2 + 24),
                )
                break
        if produce_bbox is None:
            produce_bbox = BoundingBox(
                xmin=int(round(w * (210.0 / 720.0))),
                ymin=int(round(h * (1272.0 / 1600.0))),
                xmax=int(round(w * (510.0 / 720.0))),
                ymax=int(round(h * (1368.0 / 1600.0))),
            )

        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=1, plan=plan, override_stats=meta["stats"])
        return ScreenObservation(
            scene_type="home",
            plan=plan,
            scenario_id=self.scenario_id,
            week=1,
            total_weeks=meta["total_weeks"],
            week_activity="ホーム (主页)",
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=0,
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            options=[
                {
                    "choice_id": 0,
                    "name": "プロデュース",
                    "title": "プロデュース",
                    "description": "从主页进入プロデュース剧本选择",
                    "bbox": produce_bbox.model_dump(),
                }
            ],
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _parse_produce_setup_screen(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]
    ) -> ScreenObservation:
        """Recognize pre-produce setup screens (H.I.F 選抜試験/本戦, 1.アイドル選択, 2.開始確認)."""
        h, w = frame.shape[:2]
        full_joined = "".join(ln["text"] for ln in lines)
        options: List[Dict[str, Any]] = []

        def _find_text_bbox(
            tokens: Tuple[str, ...],
            *,
            y_min_frac: float = 0.0,
            y_max_frac: float = 1.0,
            pad_x: int = 35,
            pad_y: int = 18,
            exclude: Tuple[str, ...] = ("ジュエル", "リタイア", "変更"),
        ) -> Optional[BoundingBox]:
            for ln in lines:
                txt = ln["text"].strip()
                if any(ex in txt for ex in exclude):
                    continue
                bx1, by1, bx2, by2 = ln["bbox"]
                cy = (by1 + by2) // 2
                if not (int(h * y_min_frac) <= cy <= int(h * y_max_frac)):
                    continue
                if any(tok in txt for tok in tokens):
                    return BoundingBox(
                        xmin=max(10, bx1 - pad_x),
                        ymin=max(10, by1 - pad_y),
                        xmax=min(w - 10, bx2 + pad_x),
                        ymax=min(h - 10, by2 + pad_y),
                    )
            return None

        if ("選抜試験" in full_joined and "本戦" in full_joined) or ("Hatsuboshi" in full_joined and "本戦" in full_joined):
            stage_desc = "H.I.F 剧本选择 (選抜試験 / 本戦)"
            honsen_box = _find_text_bbox(("本戦",), y_min_frac=0.68, y_max_frac=0.86, pad_x=65, pad_y=28) or BoundingBox(
                xmin=int(round(w * (405.0 / 720.0))),
                ymin=int(round(h * (1152.0 / 1600.0))),
                xmax=int(round(w * (665.0 / 720.0))),
                ymax=int(round(h * (1298.0 / 1600.0))),
            )
            senbatsu_box = _find_text_bbox(("選抜試験",), y_min_frac=0.68, y_max_frac=0.86, pad_x=50, pad_y=28) or BoundingBox(
                xmin=int(round(w * (60.0 / 720.0))),
                ymin=int(round(h * (1152.0 / 1600.0))),
                xmax=int(round(w * (320.0 / 720.0))),
                ymax=int(round(h * (1298.0 / 1600.0))),
            )
            options.append({
                "choice_id": 0,
                "name": "本戦",
                "title": "本戦 (H.I.F 本戦剧本)",
                "description": "进入 H.I.F 本戦 偶像与編成确认",
                "bbox": honsen_box.model_dump(),
            })
            options.append({
                "choice_id": 1,
                "name": "選抜試験",
                "title": "選抜試験",
                "description": "进入 H.I.F 選抜試験",
                "bbox": senbatsu_box.model_dump(),
            })
        elif any(k in full_joined for k in ("開始確認", "プロデュース開始", "ース開始", "獲得量上昇", "最終確認")):
            stage_desc = "2.開始確認"
            start_box = (
                _find_text_bbox(("プロデュース開始", "開始する"), y_min_frac=0.80, y_max_frac=0.93, pad_x=60, pad_y=24)
                or BoundingBox(
                    xmin=int(round(w * (255.0 / 720.0))),
                    ymin=int(round(h * (1358.0 / 1600.0))),
                    xmax=int(round(w * (575.0 / 720.0))),
                    ymax=int(round(h * (1462.0 / 1600.0))),
                )
            )
            detail_box = _find_text_bbox(("詳細",), y_min_frac=0.50, y_max_frac=0.64, pad_x=38, pad_y=16) or BoundingBox(
                xmin=int(round(w * (450.0 / 720.0))),
                ymin=int(round(h * (872.0 / 1600.0))),
                xmax=int(round(w * (566.0 / 720.0))),
                ymax=int(round(h * (924.0 / 1600.0))),
            )
            options.append({
                "choice_id": 0,
                "name": "プロデュース開始",
                "title": "プロデュース開始",
                "description": "确认編成并开始プロデュース",
                "bbox": start_box.model_dump(),
            })
            options.append({
                "choice_id": 1,
                "name": "詳細",
                "title": "詳細 (扫描初期技能卡与P道具)",
                "is_loadout_target": True,
                "open_via_long_press": False,
                "description": "打开詳細面板读取初期卡组与P道具并存入本局培育库",
                "bbox": detail_box.model_dump(),
            })
        elif (
            any(k in full_joined for k in ("アイドル選択", "選抜試験メモリー", "育成情報", "Pアイドル設定"))
            and "開始確認" not in full_joined
            and "プロデュース開始" not in full_joined
        ):
            stage_desc = "1.アイドル選択"
            next_box = _find_text_bbox(("次へ",), y_min_frac=0.82, y_max_frac=0.93, pad_x=70, pad_y=22) or BoundingBox(
                xmin=int(round(w * (240.0 / 720.0))),
                ymin=int(round(h * (1368.0 / 1600.0))),
                xmax=int(round(w * (480.0 / 720.0))),
                ymax=int(round(h * (1452.0 / 1600.0))),
            )
            options.append({
                "choice_id": 0,
                "name": "次へ",
                "title": "次へ",
                "description": "进入 2.開始確認",
                "bbox": next_box.model_dump(),
            })
            options.append({
                "choice_id": 1,
                "name": "アイドル詳細(長押し)",
                "title": "アイドル詳細(長押し)",
                "is_loadout_target": True,
                "open_via_long_press": True,
                "description": "長押し选中偶像卡片读取初期技能卡与P道具",
                "bbox": BoundingBox(
                    xmin=int(round(w * (170.0 / 720.0))),
                    ymin=int(round(h * (1105.0 / 1600.0))),
                    xmax=int(round(w * (310.0 / 720.0))),
                    ymax=int(round(h * (1265.0 / 1600.0))),
                ).model_dump(),
            })
        else:
            stage_desc = "プロデュース準備"
            orange_btn = self._ocr_bottom_orange_button(frame, w, h)
            if orange_btn is not None and not any(k in str(orange_btn.get("name") or "") for k in ("ジュエル", "リタイア")):
                options.append(orange_btn)
            fwd_box = _find_text_bbox(("プロデュース開始", "開始する", "次へ", "決定"), y_min_frac=0.78, y_max_frac=0.94, pad_x=55, pad_y=20)
            if fwd_box is not None and not options:
                options.append({
                    "choice_id": 0,
                    "name": "次へ",
                    "title": "次へ / 開始",
                    "description": "继续準備流程",
                    "bbox": fwd_box.model_dump(),
                })
            if not options:
                options = self._extract_bottom_and_list_options(lines, w, h, frame=frame)

        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=1, plan=plan, override_stats=meta["stats"])
        return ScreenObservation(
            scene_type="produce_setup",
            plan=plan,
            scenario_id=self.scenario_id,
            week=1,
            total_weeks=meta["total_weeks"],
            week_activity=stage_desc,
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=0,
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _extract_bottom_and_list_options(
        self, lines: List[Dict[str, Any]], w: int, h: int, frame: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """Extract choice buttons (such as 次へ/決定/スキップ) while ignoring HUD toggles and disabled retry."""
        options: List[Dict[str, Any]] = []
        action_tokens = ["再開する", "再開", "続きから", "スキップ", "SKIP", "プロデュース完了", "完了", "ホームへ", "次へ", "決定", "確認", "OK", "受け取る", "終了", "挑戦", "開始", "再挑戦", "閉じる", "はい", "選択"]
        ignore_tokens = ("早送り", "ログ", "オート", "AUTO", "メニュー", "再生成")
        full_joined = "".join(l["text"] for l in lines)
        orange_btn = self._ocr_bottom_orange_button(frame, w, h) if frame is not None else None
        for ln in lines:
            y1, y2 = ln["bbox"][1], ln["bbox"][3]
            txt = ln["text"]
            if any(ig in txt for ig in ignore_tokens):
                continue
            if "再挑戦" in txt:
                if any(k in full_joined for k in ("チケットがありません", "再挑戦チケット", "ジュエル", "あと0", "残り0", "1位", "2位", "3位")):
                    continue
                if orange_btn is not None and orange_btn.get("name") == "次へ":
                    continue
                if frame is not None:
                    bx1_r, _, bx2_r, _ = ln["bbox"]
                    btn_p = frame[max(0, y1 - 8) : min(h, y2 + 8), max(0, (bx1_r + bx2_r) // 2) : min(w, bx2_r + 18)]
                    if btn_p.size and float(np.percentile(btn_p, 85)) < 215.0:
                        continue
                elif "無料" not in full_joined:
                    continue
            if int(h * 0.05) <= y1 <= int(h * 0.96) and 2 <= len(txt) <= 10 and not txt.endswith(("?", "？", "。", "！")):
                if (
                    ("確認" in txt and txt.strip() != "確認")
                    or ("しますか" in txt)
                    or ("詳細" in txt)
                ):
                    continue
                if any(tok in txt for tok in action_tokens):
                    if y1 < int(h * 0.45) and any(
                        tok in txt for tok in ("確認", "決定", "はい", "キャンセル")
                    ):
                        continue
                    bx1, by1, bx2, by2 = ln["bbox"]
                    options.append({
                        "choice_id": len(options),
                        "name": txt,
                        "title": txt,
                        "description": f"点击屏幕按钮: {txt}",
                        "bbox": BoundingBox(
                            xmin=max(10, bx1 - 25),
                            ymin=max(10, by1 - 15),
                            xmax=min(w - 10, bx2 + 25),
                            ymax=min(h - 10, by2 + 15),
                        ).model_dump(),
                    })
        if orange_btn is not None:
            ob = orange_btn["bbox"]
            ocx, ocy = (ob["xmin"] + ob["xmax"]) // 2, (ob["ymin"] + ob["ymax"]) // 2
            already_has_orange = any(
                (o.get("name") == orange_btn["name"])
                or (
                    abs(((o["bbox"]["xmin"] + o["bbox"]["xmax"]) // 2) - ocx) <= 80
                    and abs(((o["bbox"]["ymin"] + o["bbox"]["ymax"]) // 2) - ocy) <= 80
                )
                for o in options
            )
            if not already_has_orange:
                options.insert(0, orange_btn)
                for idx_o, opt_item in enumerate(options):
                    opt_item["choice_id"] = idx_o
        if not options and frame is not None:
            circle_x_btn = self._detect_bottom_circle_x_close(frame, w, h)
            if circle_x_btn is not None:
                options.append(circle_x_btn)
        if not options:
            options.append({
                "choice_id": 0,
                "name": "点击屏幕继续",
                "title": "点击屏幕继续",
                "description": "过场/对话继续点击",
                "bbox": BoundingBox(xmin=w // 2 - 100, ymin=int(h * 0.78), xmax=w // 2 + 100, ymax=int(h * 0.88)).model_dump(),
            })
        return options[:6]

    def _scan_reward_slot_band(
        self, frame: np.ndarray, w: int, h: int, y_top: int, y_bot: int
    ) -> List[Tuple[int, int, int, int, bool]]:
        """Segment square reward card/drink slots inside a horizontal band and split touching glows."""
        x_left, x_right = int(w * 0.11), int(w * 0.89)
        band = frame[y_top:y_bot, x_left:x_right]
        if band.size == 0:
            return []
        min_ch = np.min(band, axis=2)
        is_orange_halo = (
            (band[:, :, 0] > 215)
            & (band[:, :, 2] < 135)
            & ((band[:, :, 0].astype(np.int16) - band[:, :, 2].astype(np.int16)) > 95)
        )
        # Keep orange card artwork (e.g. Sense R silhouettes on white backgrounds) when finding slot spans,
        # and use the non-halo ink profile only when splitting merged/touching segments.
        ink_col_all = (min_ch < 228).mean(axis=0)
        ink_col = ((min_ch < 218) & (~is_orange_halo)).mean(axis=0)
        ink = ink_col_all > 0.12
        raw_segs: List[Tuple[int, int]] = []
        in_seg = False
        s = 0
        for x in range(band.shape[1]):
            if ink[x] and not in_seg:
                in_seg = True
                s = x
            elif not ink[x] and in_seg:
                in_seg = False
                if x - s >= 70:
                    raw_segs.append((s, x))
        if in_seg and band.shape[1] - s >= 70:
            raw_segs.append((s, band.shape[1]))

        max_slot_w = max(145, int(w * 0.25))
        min_slot_w = 70
        split_segs: List[Tuple[int, int]] = []
        stack = list(raw_segs)
        while stack:
            seg_s, seg_e = stack.pop(0)
            if seg_e - seg_s <= max_slot_w:
                split_segs.append((x_left + seg_s, x_left + seg_e))
                continue
            search_lo = seg_s + int(w * 0.13)
            search_hi = seg_e - int(w * 0.13)
            if search_hi <= search_lo:
                split_segs.append((x_left + seg_s, x_left + seg_e))
                continue
            rel_cut = int(np.argmin(ink_col[search_lo:search_hi])) + search_lo
            if rel_cut - seg_s >= min_slot_w and seg_e - rel_cut >= min_slot_w:
                stack.insert(0, (rel_cut + 2, seg_e))
                stack.insert(0, (seg_s, max(seg_s + 1, rel_cut - 2)))
            else:
                split_segs.append((x_left + seg_s, x_left + seg_e))

        if len(split_segs) == 2:
            w0 = split_segs[0][1] - split_segs[0][0]
            w1 = split_segs[1][1] - split_segs[1][0]
            cx0 = (split_segs[0][0] + split_segs[0][1]) / 2.0
            cx1 = (split_segs[1][0] + split_segs[1][1]) / 2.0
            pitch = int(round(cx1 - cx0))
            if int(w * 0.14) <= pitch <= int(w * 0.25) and abs(w0 - w1) <= 35:
                cand_seg: Optional[Tuple[int, int]] = None
                if 0.22 * w <= cx0 <= 0.36 * w and 0.44 * w <= cx1 <= 0.56 * w:
                    cand_seg = (split_segs[1][0] + pitch, split_segs[1][1] + pitch)
                elif 0.44 * w <= cx0 <= 0.56 * w and 0.64 * w <= cx1 <= 0.78 * w:
                    cand_seg = (split_segs[0][0] - pitch, split_segs[0][1] - pitch)
                if cand_seg is not None:
                    cs, ce = cand_seg
                    if cs >= int(w * 0.08) and ce <= int(w * 0.92):
                        patch = frame[y_top:y_bot, cs:ce]
                        if (
                            patch.size > 0
                            and float((np.min(patch, axis=2) < 228).mean()) > 0.25
                            and float(patch.std()) > 18.0
                        ):
                            split_segs.append((cs, ce))
                            split_segs.sort(key=lambda seg: seg[0])

        bracket_orange: List[Tuple[int, int, int, int, int]] = []
        for x1, x2 in split_segs:
            top_br = frame[max(0, y_top - 16) : max(0, y_top - 2), max(0, x1 + 4) : min(w, max(x1 + 5, x2 - 4))]
            bot_br = frame[min(h, y_bot + 2) : min(h, y_bot + 16), max(0, x1 + 4) : min(w, max(x1 + 5, x2 - 4))]
            left_br = frame[max(0, y_top - 4) : min(h, y_bot + 4), max(0, x1 - 14) : max(0, x1 - 2)]
            right_br = frame[max(0, y_top - 4) : min(h, y_bot + 4), min(w, x2 + 2) : min(w, x2 + 14)]
            br_cnt = 0
            for br in (top_br, bot_br, left_br, right_br):
                if br.size:
                    br_cnt += int(
                        ((br[:, :, 0] > 230) & (br[:, :, 1] > 115) & (br[:, :, 1] < 185) & (br[:, :, 2] < 85)).sum()
                    )
            bracket_orange.append((x1, y_top, x2, y_bot, br_cnt))
        max_orange = max((c for *_, c in bracket_orange), default=0)
        best_idx = -1
        if max_orange >= 80:
            best_idx = max(range(len(bracket_orange)), key=lambda idx: bracket_orange[idx][4])
        return [(x1, y1, x2, y2, idx == best_idx) for idx, (x1, y1, x2, y2, _) in enumerate(bracket_orange)]

    def _scan_deck_grid_slots(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        *,
        allow_trouble_dark: bool = False,
        bottom_first: bool = False,
        grid_mode: str = "default",
    ) -> List[Tuple[int, int, int, int, bool, bool]]:
        """Detect 4-column scrollable deck-grid card slots for チェンジ/削除/強化/カスタマイズ."""
        cols = (
            (0.112, 0.278),
            (0.322, 0.488),
            (0.532, 0.698),
            (0.742, 0.908),
        )
        rows_px: List[Tuple[int, int]]
        if grid_mode == "upgrade":
            rows_px = [
                (int(h * y1_f), int(h * y2_f))
                for y1_f, y2_f in (
                    (0.438, 0.514),
                    (0.532, 0.608),
                    (0.626, 0.702),
                    (0.720, 0.796),
                )
            ]
        elif grid_mode == "move_to_hand":
            rows_px = [
                (int(h * y1_f), int(h * y2_f))
                for y1_f, y2_f in (
                    (0.248, 0.324),
                    (0.341, 0.417),
                    (0.433, 0.509),
                    (0.525, 0.601),
                    (0.618, 0.694),
                    (0.711, 0.786),
                )
            ]
        elif grid_mode == "shop_delete":
            rows_px = [
                (int(h * y1_f), int(h * y2_f))
                for y1_f, y2_f in (
                    (0.711, 0.786),
                    (0.618, 0.694),
                    (0.525, 0.601),
                    (0.433, 0.509),
                    (0.341, 0.417),
                    (0.248, 0.324),
                )
            ]
        elif grid_mode == "customize":
            rows_px = [
                (int(h * y1_f), int(h * y2_f))
                for y1_f, y2_f in (
                    (0.353, 0.425),
                    (0.447, 0.519),
                    (0.541, 0.609),
                    (0.634, 0.703),
                    (0.728, 0.797),
                )
            ]
        else:
            y_scan_top = int(h * 0.390)
            y_scan_bot = int(h * 0.895)
            dyn_bands: List[Tuple[int, int]] = []
            if y_scan_bot > y_scan_top:
                col_p90s = []
                for x1_f, x2_f in cols:
                    x1, x2 = int(w * x1_f), int(w * x2_f)
                    strip = frame[y_scan_top:y_scan_bot, x1:x2]
                    if strip.size > 0:
                        col_p90s.append(np.percentile(strip, 90, axis=(1, 2)))
                if col_p90s:
                    row_p90 = np.maximum.reduce(col_p90s)
                    active_y = np.where(row_p90 >= 150.0)[0] + y_scan_top
                    gap_merge = max(4, int(h * 0.004))
                    raw_runs: List[List[int]] = []
                    for yy in active_y:
                        yi = int(yy)
                        if not raw_runs or yi > raw_runs[-1][1] + gap_merge:
                            raw_runs.append([yi, yi])
                        else:
                            raw_runs[-1][1] = yi
                    min_bh = int(h * 0.055)
                    max_bh = int(h * 0.125)
                    dyn_bands = [(y1, y2) for y1, y2 in raw_runs if min_bh <= (y2 - y1) <= max_bh]
            if dyn_bands and allow_trouble_dark:
                if len(dyn_bands) >= 2:
                    deltas = sorted(dyn_bands[i + 1][0] - dyn_bands[i][0] for i in range(len(dyn_bands) - 1))
                    pitch = deltas[len(deltas) // 2]
                else:
                    pitch = int(round(h * 0.092))
                ext_y1 = dyn_bands[-1][0] + pitch
                ext_y2 = dyn_bands[-1][1] + pitch
                if ext_y2 <= int(h * 0.905):
                    dyn_bands.append((ext_y1, ext_y2))
            if dyn_bands:
                rows_px = list(reversed(dyn_bands))
            else:
                rows_px = [
                    (int(h * y1_f), int(h * y2_f))
                    for y1_f, y2_f in (
                        (0.852, 0.892),
                        (0.750, 0.835),
                        (0.660, 0.745),
                        (0.570, 0.655),
                        (0.480, 0.565),
                        (0.390, 0.475),
                    )
                ]
        raw_cells: List[Tuple[int, int, int, int, int, bool]] = []
        for y1, y2 in rows_px:
            for x1_f, x2_f in cols:
                x1, x2 = int(w * x1_f), int(w * x2_f)
                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                mean_v = float(patch.mean())
                if mean_v < 75.0:
                    continue
                sat = float((patch.max(axis=2).astype(np.int16) - patch.min(axis=2).astype(np.int16)).mean())
                p95 = float(np.percentile(patch, 95))
                if grid_mode == "shop_delete":
                    patch_std = float(patch.std())
                    if mean_v < 130.0 and sat < 40.0 and (patch_std < 22.0 or mean_v < 85.0):
                        # Blurred empty trailing slot in 6x4 shop delete grid (preserve crisp dark-gray T/眠気 cards)
                        continue
                elif grid_mode == "move_to_hand":
                    # In #22 移動 modal, populated cards sit brightly above the blurred stage background (mean>=138 & p95>=185)
                    if mean_v < 138.0 or p95 < 185.0:
                        continue
                elif (not allow_trouble_dark) and (p95 < 175.0 or (mean_v < 122.0 and sat < 45.0)):
                    # Dimmed / darkened unselectable card (e.g. 眠気 / locked card in チェンジ)
                    continue
                top_y2 = min(h, y1 + (14 if (y2 - y1) > int(h * 0.082) else 6))
                top_br = frame[max(0, y1 - 14) : top_y2, max(0, x1 - 10) : min(w, x2 + 10)]
                bot_br = frame[max(0, y2 - 6) : min(h, y2 + 14), max(0, x1 - 10) : min(w, x2 + 10)]
                left_outer = frame[y1:y2, max(0, x1 - 11) : max(0, x1 - 1)]
                right_outer = frame[y1:y2, min(w, x2 + 1) : min(w, x2 + 11)]
                o_cnt = 0
                for br in (top_br, bot_br):
                    if br.size > 0:
                        o_cnt += int(
                            ((br[:, :, 0] > 230) & (br[:, :, 1] > 115) & (br[:, :, 1] < 190) & (br[:, :, 2] < 90)).sum()
                        )
                outer_o_cnt = 0
                for obr in (left_outer, right_outer):
                    if obr.size > 0:
                        outer_o_cnt += int(
                            ((obr[:, :, 0] > 225) & (obr[:, :, 1] > 115) & (obr[:, :, 1] < 195) & (obr[:, :, 2] < 95)).sum()
                        )
                # Protruding left/right orange L-corner arms uniquely distinguish true selection from gold SSR borders
                o_score = o_cnt + outer_o_cnt * 3
                is_basic_gray = (
                    (sat < 48.0 and (mean_v >= 130.0 or float(patch.std()) >= 22.0))
                    if grid_mode == "shop_delete"
                    else ((mean_v < 142.0) and (p95 >= 175.0 or allow_trouble_dark))
                )
                raw_cells.append((x1, y1, x2, y2, o_score, is_basic_gray))
        if not raw_cells:
            return []
        max_o = max(c[4] for c in raw_cells)
        best_idx = max(range(len(raw_cells)), key=lambda i: raw_cells[i][4]) if max_o >= 120 else -1
        # Keep spatial slot ordering invariant across clicks so grid slot_idx never shifts mid-scan
        ordered = sorted(
            [
                (i, x1, y1, x2, y2, i == best_idx, is_gray)
                for i, (x1, y1, x2, y2, _, is_gray) in enumerate(raw_cells)
            ],
            key=(
                (lambda t: (-t[2], t[1]))
                if bottom_first
                else (lambda t: (t[2], t[1]))
            ),
        )
        return [(x1, y1, x2, y2, is_sel, is_gray) for _, x1, y1, x2, y2, is_sel, is_gray in ordered]

    def _find_reward_icon_slots(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        include_deck_grid: bool = False,
        *,
        allow_trouble_dark: bool = False,
        bottom_first: bool = False,
        grid_mode: str = "default",
    ) -> List[Tuple[int, int, int, int, bool]]:
        """Detect horizontal square card/drink icon thumbnails across normal & expanded preview layouts."""
        if include_deck_grid:
            grid_slots = self._scan_deck_grid_slots(
                frame,
                w,
                h,
                allow_trouble_dark=allow_trouble_dark,
                bottom_first=bottom_first,
                grid_mode=grid_mode,
            )
            if grid_slots:
                return [(x1, y1, x2, y2, is_sel) for x1, y1, x2, y2, is_sel, _ in grid_slots]
        best: List[Tuple[int, int, int, int, bool]] = []
        for y1_f, y2_f in ((0.625, 0.718), (0.648, 0.752), (0.600, 0.695)):
            cand = self._scan_reward_slot_band(frame, w, h, int(h * y1_f), int(h * y2_f))
            if len(cand) > len(best):
                best = cand
            if len(best) >= 3:
                break
        return best

    def _parse_card_reward_screen(self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]) -> ScreenObservation:
        h, w = frame.shape[:2]
        full_joined = "".join(ln["text"] for ln in lines)
        norm_joined = _normalize_katakana_ocr(full_joined)
        held_drinks = self._recognize_bottom_drinks(frame, plan)
        options: List[Dict[str, Any]] = []

        want_deck_grid = (
            ("スキルカードが選択されていません" in full_joined)
            or any(
                k in norm_joined
                for k in (
                    "強化するスキルカード",
                    "削除するスキルカード",
                    "チェンジするスキルカード",
                    "手放すスキルカード",
                    "カスタマイズするスキルカード",
                    "カスタマイズ可能スキルカード",
                )
            )
            or ("チェンジする" in norm_joined and "チェンジで獲得" not in norm_joined)
            or (
                "チェンジ" in norm_joined
                and "チェンジで獲得" not in norm_joined
                and any(k in norm_joined for k in ("スキルカードを選択", "を選択してください", "カスタマイズ確認"))
            )
        )
        is_customize_flow = any(k in norm_joined for k in ("カスタマイズ", "特別指導")) and any(
            k in norm_joined
            for k in ("カスタマイズを選", "カスタマイズ内容", "強化内容", "カスタマイズする", "カスタマイズ可能スキルカード")
        )
        has_delete_header = (
            ("削除するスキルカード" in norm_joined)
            or any(
                ("削除" in ln["text"])
                and len(ln["text"].strip()) <= 6
                and ln["bbox"][1] < int(h * 0.12)
                and ln["bbox"][0] < int(w * 0.35)
                for ln in lines
            )
        )
        has_change_or_delete_cue = (
            has_delete_header
            or ("チェンジするスキルカード" in norm_joined)
            or ("手放すスキルカード" in norm_joined)
            or ("チェンジする" in norm_joined and "チェンジで獲得" not in norm_joined)
            or ("削除する" in norm_joined)
            or any(
                ("チェンジ" in _normalize_katakana_ocr(ln["text"]) or "削除" in ln["text"])
                and len(ln["text"].strip()) <= 8
                and (
                    (ln["bbox"][1] < int(h * 0.12) and ln["bbox"][0] < int(w * 0.40))
                    or ln["bbox"][1] > int(h * 0.74)
                )
                for ln in lines
            )
        )
        has_upgrade_header = (
            ("強化するスキルカード" in norm_joined)
            or any(
                ("強化" in ln["text"]) and len(ln["text"].strip()) <= 6 and ln["bbox"][1] < int(h * 0.12)
                for ln in lines
            )
        )
        has_reward_receive_cue = (
            any(
                k in norm_joined
                for k in (
                    "受け取る",
                    "受け取らない",
                    "再抽選",
                    "Pドリンク",
                    "ドリンクを選",
                    "ドリンクを選択",
                    "ドリンク獲得",
                    "チェンジで獲得",
                )
            )
            or any(
                ("ドリンク" in ln["text"]) and ln["bbox"][1] < int(h * 0.58)
                for ln in lines
            )
        )
        # Never let card/drink effect description text (e.g. パワフル漢方ドリンク / 初星黒酢 / 天賦の才+
        # containing '山札か捨札...手札に移動') hijack produce rewards or チェンジ/削除/強化/カスタマイズ
        # into the #22 move_to_hand battle grid!
        is_move_to_hand_flow = (
            not is_customize_flow
            and not has_change_or_delete_cue
            and not has_upgrade_header
            and not has_reward_receive_cue
            and (
                ("移動するスキルカード" in norm_joined)
                or any(
                    ("移動" in ln["text"])
                    and len(ln["text"].strip()) <= 6
                    and ln["bbox"][1] < int(h * 0.12)
                    and ln["bbox"][0] < int(w * 0.35)
                    for ln in lines
                )
                or any(
                    ln["text"].strip().lstrip("ノ◆◇✦✓✔・") in ("移動", "移動する")
                    and ln["bbox"][1] > int(h * 0.74)
                    for ln in lines
                )
                or (
                    self._in_battle_scene
                    and (
                        ("手札に移動" in norm_joined)
                        or ("山札か捨札" in norm_joined)
                        or ("が選択されていません" in norm_joined)
                    )
                )
            )
            and ("移動中" not in norm_joined)
        )
        self._in_battle_scene = bool(is_move_to_hand_flow)
        is_shop_delete_flow = (
            not is_customize_flow
            and not is_move_to_hand_flow
            and has_delete_header
        )
        is_upgrade_flow = (
            not is_customize_flow
            and not is_move_to_hand_flow
            and not has_change_or_delete_cue
            and (has_upgrade_header or ("強化確認" in norm_joined))
        )
        want_deck_grid = want_deck_grid or is_upgrade_flow or is_shop_delete_flow or is_move_to_hand_flow
        is_deck_trade_or_delete = (
            ("削除" in norm_joined)
            or ("手放す" in norm_joined)
            or ("チェンジ" in norm_joined and "チェンジで獲得" not in norm_joined)
        ) and (not is_upgrade_flow) and (not is_customize_flow) and (not is_move_to_hand_flow)

        # 1. Inspect confirm button ("受け取る" / "決定" / "チェンジ" / "カスタマイズする"), skip pill, reroll, cancel, and optional "受け取らない" checkbox
        receive_opt: Optional[Dict[str, Any]] = None
        receive_active = False
        skip_opt: Optional[Dict[str, Any]] = None
        reroll_opt: Optional[Dict[str, Any]] = None
        cancel_opt: Optional[Dict[str, Any]] = None
        dont_take_ln: Optional[Dict[str, Any]] = None
        for ln in lines:
            txt = ln["text"].strip()
            norm_txt = _normalize_katakana_ocr(txt)
            if "受け取らない" in txt and len(txt) <= 10 and int(h * 0.65) <= ln["bbox"][1] <= int(h * 0.77):
                dont_take_ln = ln
            elif (
                (
                    "受け取る" in norm_txt
                    or "強化する" in norm_txt
                    or (is_upgrade_flow and "強化" in norm_txt and "強化確認" not in norm_txt)
                    or "削除する" in norm_txt
                    or (is_shop_delete_flow and "削除" in norm_txt and "削除確認" not in norm_txt)
                    or (is_move_to_hand_flow and re.sub(r"^[ノレvV✓✔◆◇✦\W]+", "", norm_txt) in ("移動", "移動する"))
                    or "チェンジ" in norm_txt
                    or "カスタマイズする" in norm_txt
                    or ("カスタマイズ" in norm_txt and "可能" not in norm_txt and "メニュー" not in norm_txt and "確認" not in norm_txt)
                    or "決定" in norm_txt
                    or "次へ" in norm_txt
                )
                and "受け取らない" not in txt
                and len(txt) <= 10
                and ln["bbox"][1] > int(h * 0.74)
            ):
                bx1, by1, bx2, by2 = ln["bbox"]
                btn_patch = frame[max(0, by1 - 15) : min(h, by2 + 15), max(0, bx1 - 45) : min(w, bx2 + 45)]
                mean_r = float(btn_patch[:, :, 0].mean()) if btn_patch.size else 0.0
                receive_active = mean_r > (205.0 if (is_shop_delete_flow or is_move_to_hand_flow) else 218.0)
                if is_shop_delete_flow and "削除" in norm_txt:
                    opt_name = "削除する"
                elif is_move_to_hand_flow and re.sub(r"^[ノレvV✓✔◆◇✦\W]+", "", norm_txt) in ("移動", "移動する"):
                    opt_name = "移動する"
                else:
                    opt_name = txt
                receive_opt = {
                    "name": opt_name,
                    "title": opt_name,
                    "bbox": BoundingBox(xmin=max(10, bx1 - 45), ymin=max(10, by1 - 18), xmax=min(w - 10, bx2 + 45), ymax=min(h - 10, by2 + 18)).model_dump(),
                }
            elif ("キャンセル" in txt or "戻る" in txt or "もどる" in txt) and len(txt) <= 8 and ln["bbox"][1] > int(h * 0.78):
                bx1, by1, bx2, by2 = ln["bbox"]
                c_name = "キャンセル" if "キャンセル" in txt else ("もどる" if "もどる" in txt else "戻る")
                cancel_opt = {
                    "name": c_name,
                    "title": f"{c_name} (取消本次操作)",
                    "bbox": BoundingBox(xmin=max(10, bx1 - 35), ymin=max(10, by1 - 15), xmax=min(w - 10, bx2 + 35), ymax=min(h - 10, by2 + 15)).model_dump(),
                }
            elif "再抽選" in txt and len(txt) <= 14 and ln["bbox"][1] > int(h * 0.72) and "ジュエル" not in full_joined and "あと0回" not in full_joined:
                bx1, by1, bx2, by2 = ln["bbox"]
                rp_patch = frame[max(0, by1 - 10) : min(h, by2 + 10), max(0, bx1 - 15) : min(w, bx2 + 15)]
                if rp_patch.size == 0 or float(rp_patch.mean()) >= 165.0:
                    reroll_opt = {
                        "name": "再抽選",
                        "title": "再抽選 (免费重抽候选卡)",
                        "bbox": BoundingBox(xmin=max(10, bx1 - 20), ymin=max(10, by1 - 15), xmax=min(w - 10, bx2 + 20), ymax=min(h - 10, by2 + 15)).model_dump(),
                    }
            elif "スキップ" in txt and len(txt) <= 8 and ln["bbox"][1] > int(h * 0.72):
                bx1, by1, bx2, by2 = ln["bbox"]
                skip_opt = {
                    "name": "スキップ",
                    "title": "スキップ",
                    "bbox": BoundingBox(xmin=max(10, bx1 - 30), ymin=max(10, by1 - 15), xmax=min(w - 10, bx2 + 30), ymax=min(h - 10, by2 + 15)).model_dump(),
                }

        cap_band_txt = ""
        if dont_take_ln is not None and not receive_active and h >= 400 and w >= 300:
            cap_band_txt = self._ocr_crop_upscaled(
                frame, (int(w * 0.28), int(h * 0.785), int(w * 0.72), int(h * 0.84)), scale=2, mode="white", pad=12
            )
        has_cap_text = ("所持上限" in full_joined) or ("所持上限" in cap_band_txt)

        if (receive_opt is None or not receive_active) and not has_cap_text:
            orange_confirm = self._ocr_bottom_orange_button(frame, w, h)
            if orange_confirm is not None and orange_confirm["bbox"]["ymin"] >= int(h * 0.74):
                oc_name = re.sub(r"^[ノレvV✓✔◆◇✦\W]+", "", str(orange_confirm["name"] or "").strip())
                if oc_name in ("決定", "受け取る", "強化する", "強化", "削除する", "削除", "移動する", "移動", "チェンジする", "チェンジ", "カスタマイズする", "カスタマイズ", "次へ", "確認", "はい", "OK", "受け取らない"):
                    if is_shop_delete_flow and oc_name in ("削除", "決定"):
                        receive_opt = dict(orange_confirm, name="削除する", title="削除する")
                    elif is_move_to_hand_flow and oc_name in ("移動", "移動する", "決定"):
                        receive_opt = dict(orange_confirm, name="移動する", title="移動する")
                    else:
                        receive_opt = orange_confirm
                    receive_active = True

        dont_take_bbox: Optional[Dict[str, int]] = None
        dont_take_checked = False
        if dont_take_ln is not None:
            bx1, by1, bx2, by2 = dont_take_ln["bbox"]
            dont_take_bbox = BoundingBox(
                xmin=max(10, bx1 - 75),
                ymin=max(10, by1 - 15),
                xmax=min(w - 10, bx2 + 35),
                ymax=min(h - 10, by2 + 15),
            ).model_dump()
            cb_patch = frame[max(0, by1 - 8) : min(h, by2 + 8), max(0, bx1 - 62) : max(1, bx1 - 4)]
            cb_orange = (
                int(
                    (
                        (cb_patch[:, :, 0] > 215)
                        & (cb_patch[:, :, 1] > 100)
                        & (cb_patch[:, :, 1] < 190)
                        & (cb_patch[:, :, 2] < 95)
                    ).sum()
                )
                if cb_patch.size
                else 0
            )
            dont_take_checked = cb_orange >= 20

        # 2. Read upper modal preview panel (upgrade uses y=0.065..0.30; deck-grid チェンジ/削除 uses y=0.155..0.345; regular rewards use y=0.37..0.59)
        if is_upgrade_flow:
            pv_patch = frame[int(h * 0.065) : int(h * 0.30), int(w * 0.14) : int(w * 0.92)]
        elif is_shop_delete_flow or is_move_to_hand_flow:
            pv_patch = frame[int(h * 0.060) : int(h * 0.255), int(w * 0.14) : int(w * 0.92)]
        elif want_deck_grid:
            pv_patch = frame[int(h * 0.155) : int(h * 0.345), int(w * 0.26) : int(w * 0.92)]
        else:
            pv_patch = frame[int(h * 0.37) : int(h * 0.59), int(w * 0.08) : int(w * 0.92)]
        pv_res = self.ocr.recognize_array(pv_patch)
        pv_lines = [
            ln["text"].strip()
            for ln in pv_res.get("lines", [])
            if ln["text"].strip() and "カスタマイズ" not in ln["text"] and "選択されていません" not in ln["text"]
        ]
        preview_card_name = ""
        preview_effect = ""
        preview_card_id: Optional[int] = None
        preview_drink_id: Optional[int] = None
        pv_joined = "".join(pv_lines)
        if pv_lines and "選んでください" not in pv_joined and "上限" not in pv_joined:
            preview_card_name = pv_lines[0]
            matched_pv = self.match_card_by_name(preview_card_name, plan=plan)
            if not matched_pv:
                for cand_pv in pv_lines[:4]:
                    cleaned_pv = re.sub(r"^[\d\W_]+", "", cand_pv).strip()
                    if len(cleaned_pv) >= 2:
                        matched_pv = self.match_card_by_name(cleaned_pv, plan=plan)
                        if matched_pv:
                            break
        else:
            matched_pv = None
        if is_move_to_hand_flow and not matched_pv:
            pv2_patch = frame[int(h * 0.22) : int(h * 0.46), int(w * 0.14) : int(w * 0.92)]
            pv2_res = self.ocr.recognize_array(pv2_patch)
            pv2_lines = [
                ln["text"].strip()
                for ln in pv2_res.get("lines", [])
                if ln["text"].strip() and "カスタマイズ" not in ln["text"] and "選択されていません" not in ln["text"]
            ]
            if pv2_lines and "選んでください" not in "".join(pv2_lines):
                pv_lines = pv2_lines
                preview_card_name = pv_lines[0]
                matched_pv = self.match_card_by_name(preview_card_name, plan=plan)
                if not matched_pv:
                    for cand_pv in pv_lines[:4]:
                        cleaned_pv = re.sub(r"^[\d\W_]+", "", cand_pv).strip()
                        if len(cleaned_pv) >= 2:
                            matched_pv = self.match_card_by_name(cleaned_pv, plan=plan)
                            if matched_pv:
                                break
        if pv_lines and "選んでください" not in "".join(pv_lines) and "上限" not in "".join(pv_lines):
            matched_dr_pv = self.match_drink_by_name(preview_card_name, plan=plan) if not matched_pv else None
            if matched_pv and dont_take_ln is None and "ドリンク" not in full_joined:
                preview_card_id = matched_pv["id"]
                preview_card_name = matched_pv["name"]
                preview_effect = matched_pv.get("description") or " / ".join(pv_lines[1:4])
            elif matched_dr_pv or dont_take_ln is not None or "ドリンク" in full_joined:
                m_dr = matched_dr_pv or self.match_drink_by_name(preview_card_name, plan=plan)
                if m_dr:
                    preview_drink_id = int(m_dr["id"])
                    preview_card_name = m_dr["name"]
                    preview_effect = m_dr.get("description") or " / ".join(pv_lines[1:4])
                elif matched_pv:
                    preview_card_id = matched_pv["id"]
                    preview_card_name = matched_pv["name"]
                    preview_effect = matched_pv.get("description") or " / ".join(pv_lines[1:4])
                else:
                    preview_effect = " / ".join(pv_lines[1:4])
            else:
                preview_effect = " / ".join(pv_lines[1:4])
        if not preview_card_name and not want_deck_grid:
            # Single-card / single-drink acquisition popup places banner lower (y ~ 55%..72%)
            for ln in lines:
                cy = (ln["bbox"][1] + ln["bbox"][3]) // 2
                if int(h * 0.55) <= cy <= int(h * 0.72):
                    matched_pv = self.match_card_by_name(ln["text"], plan=plan) if dont_take_ln is None else None
                    if matched_pv:
                        preview_card_id = matched_pv["id"]
                        preview_card_name = matched_pv["name"]
                        preview_effect = matched_pv.get("description") or ""
                        break
                    m_dr = self.match_drink_by_name(ln["text"], plan=plan)
                    if m_dr:
                        preview_drink_id = int(m_dr["id"])
                        preview_card_name = m_dr["name"]
                        preview_effect = m_dr.get("description") or ""
                        break

        # 3. Check modal square card/drink slots and P-Drink cap state (3 in Hajime, 4 in H.I.F) via visual/OCR cues
        cap_count = 4 if self.scenario_id in ("hif_honsen", "hif") else 3
        allow_trouble = ("削除" in norm_joined) and (not is_move_to_hand_flow)
        grid_mode = (
            "customize"
            if is_customize_flow
            else (
                "upgrade"
                if is_upgrade_flow
                else (
                    "move_to_hand"
                    if is_move_to_hand_flow
                    else ("shop_delete" if is_shop_delete_flow else "default")
                )
            )
        )
        grid_slots = (
            self._scan_deck_grid_slots(
                frame,
                w,
                h,
                allow_trouble_dark=allow_trouble,
                bottom_first=is_deck_trade_or_delete,
                grid_mode=grid_mode,
            )
            if want_deck_grid
            else []
        )
        icon_slots = (
            [(x1, y1, x2, y2, is_sel) for x1, y1, x2, y2, is_sel, _ in grid_slots]
            if grid_slots
            else self._find_reward_icon_slots(
                frame,
                w,
                h,
                include_deck_grid=want_deck_grid,
                allow_trouble_dark=allow_trouble,
                bottom_first=is_deck_trade_or_delete,
                grid_mode=grid_mode,
            )
        )
        if want_deck_grid and not receive_active:
            icon_slots = [(x1, y1, x2, y2, False) for x1, y1, x2, y2, _ in icon_slots]
        grid_gray_flags = [is_gray for *_, is_gray in grid_slots] if grid_slots else []
        grid_custom_flags = (
            [self._has_card_customization_badge(frame[y1:y2, x1:x2]) for x1, y1, x2, y2, _ in icon_slots]
            if (want_deck_grid and icon_slots)
            else []
        )
        is_explicit_drink_modal = (
            ("ドリンク" in full_joined)
            or (dont_take_ln is not None)
            or (preview_drink_id is not None and preview_card_id is None)
        )
        slot_matches: List[Tuple[Optional[Dict[str, Any]], float]] = (
            [self._match_square_drink_slot_with_score(frame, x1, y1, x2, y2) for x1, y1, x2, y2, _ in icon_slots]
            if icon_slots
            and (is_explicit_drink_modal or ("スキルカード" not in full_joined))
            and (not want_deck_grid)
            and (not is_move_to_hand_flow)
            and (not is_upgrade_flow)
            and (not is_customize_flow)
            and (not is_shop_delete_flow)
            else []
        )
        visual_drink_hit = any(m is not None and sc >= 0.58 for m, sc in slot_matches)
        is_drink_reward = (
            (not want_deck_grid)
            and (not is_move_to_hand_flow)
            and (not is_upgrade_flow)
            and (not is_customize_flow)
            and (not is_shop_delete_flow)
            and (
                ("ドリンク" in full_joined)
                or (dont_take_ln is not None)
                or (preview_drink_id is not None and preview_card_id is None)
                or (visual_drink_hit and preview_card_id is None)
            )
        )
        has_drink_cap = has_cap_text or (
            dont_take_ln is not None and len(held_drinks) >= cap_count and not receive_active
        )
        slot_drinks: List[Optional[Dict[str, Any]]] = []
        if is_drink_reward and icon_slots:
            for idx, (x1, y1, x2, y2, is_sel) in enumerate(icon_slots):
                if is_sel and preview_drink_id is not None:
                    cd = self.db.get_drink_by_id(preview_drink_id)
                elif idx < len(slot_matches):
                    cd = slot_matches[idx][0]
                else:
                    cd = self._match_square_drink_slot(frame, x1, y1, x2, y2)
                slot_drinks.append(cd)
                if has_drink_cap and is_sel:
                    self._reward_visited_slots.add(idx)

        if dont_take_ln is not None and dont_take_checked and dont_take_bbox is not None:
            confirm_bbox = (
                receive_opt["bbox"]
                if receive_opt is not None
                else BoundingBox(xmin=int(w * 0.30), ymin=int(h * 0.755), xmax=int(w * 0.70), ymax=int(h * 0.81)).model_dump()
            )
            options.append({
                "choice_id": 0,
                "name": "決定",
                "title": "确认不拿新饮料并继续 (已勾选『受け取らない』)",
                "description": f"点击底部橙色确认按钮跳过本次P饮料领取，保留手上现有{cap_count}瓶饮料",
                "bbox": confirm_bbox,
            })
            options.append({
                "choice_id": 1,
                "name": "取消受け取らない",
                "title": "取消勾选『受け取らない』(返回重新选择或删旧换新)",
                "description": "取消『受け取らない』勾选状态",
                "bbox": dont_take_bbox,
            })
        elif has_drink_cap and not receive_active:
            cand_parts: List[str] = []
            for idx in range(len(icon_slots)):
                cd = slot_drinks[idx] if idx < len(slot_drinks) else None
                if cd:
                    c_eff = (cd.get("description") or "").replace("\n", " ")[:24]
                    cand_parts.append(f"#{idx + 1}《{cd['name']}》({c_eff})")
                elif icon_slots[idx][4] and preview_card_name:
                    cand_parts.append(f"#{idx + 1}《{preview_card_name}》({preview_effect.replace(chr(10), ' ')[:24]})")
                else:
                    cand_parts.append(f"#{idx + 1}候选饮料")
            cand_summary = " / ".join(cand_parts) or (preview_card_name or "候选P饮料")

            for di, d in enumerate(held_drinks, 1):
                options.append({
                    "choice_id": len(options),
                    "drink_id": d.drink_id,
                    "name": f"破棄候補_{d.drink_id}",
                    "title": f"点击饮料栏旧饮料{di}《{d.name}》打开详情删除(捨てる/破棄)",
                    "description": (
                        f"当前{cap_count}/{cap_count}满槽！先点击左下角饮料栏第{di}瓶《{d.name}》({d.dsl_summary})唤出删除键『捨てる/破棄』，"
                        f"删掉旧饮料腾位后再领新饮料（候选: {cand_summary}）"
                    ),
                    "bbox": d.bbox.model_dump(),
                })

            dt_bbox = (
                dont_take_bbox
                if dont_take_bbox is not None
                else BoundingBox(xmin=int(w * 0.28), ymin=int(h * 0.715), xmax=int(w * 0.72), ymax=int(h * 0.748)).model_dump()
            )
            options.append({
                "choice_id": len(options),
                "name": "受け取らない",
                "title": f"勾选『受け取らない』直接不拿新饮料 (保留当前{cap_count}瓶旧饮料)",
                "description": f"不删除旧饮料，勾选『受け取らない』后确认跳过领取（候选: {cand_summary}）",
                "bbox": dt_bbox,
            })

            for idx, (x1, y1, x2, y2, is_sel) in enumerate(icon_slots):
                if not is_sel and idx not in self._reward_visited_slots:
                    cd = slot_drinks[idx] if idx < len(slot_drinks) else None
                    c_title = f"切换选中第{idx + 1}瓶候选饮料《{cd['name']}》" if cd else f"切换选中第{idx + 1}瓶候选饮料"
                    c_desc = (
                        f"切换查看第{idx + 1}瓶候选饮料《{cd['name']}》: {(cd.get('description') or '').replace(chr(10), ' / ')}"
                        if cd
                        else f"切换查看第{idx + 1}瓶候选饮料名称与效果"
                    )
                    options.append({
                        "choice_id": len(options),
                        **({"drink_id": int(cd["id"])} if cd else {}),
                        "name": f"预览候选饮料{idx + 1}",
                        "title": c_title,
                        "description": c_desc,
                        "bbox": BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2).model_dump(),
                    })
        elif receive_opt is not None and receive_active and (preview_card_name or len(icon_slots) <= 1):
            # Active selected card/drink ready to accept!
            sel_label = preview_card_name or receive_opt["title"]
            item_word = "饮料" if is_drink_reward else "卡"
            id_extra: Dict[str, Any] = {}
            if preview_card_id is not None:
                id_extra["card_id"] = preview_card_id
            elif preview_drink_id is not None:
                id_extra["drink_id"] = preview_drink_id
            if is_move_to_hand_flow:
                act_name = "移動する"
                act_title = f"确认移动当前选中卡至手牌: {sel_label}"
            elif is_upgrade_flow:
                act_name = "強化する"
                act_title = f"确认強化当前选中卡: {sel_label}"
            elif is_deck_trade_or_delete or is_customize_flow:
                act_name = (
                    "チェンジする"
                    if "チェンジ" in norm_joined
                    else ("削除する" if "削除" in norm_joined else "カスタマイズする")
                )
                act_title = f"确认换出/删除/自定义当前选中卡: {sel_label}"
            else:
                act_name = "受け取る"
                act_title = (
                    f"确认领取当前选中{item_word}: {sel_label}"
                    if preview_card_name
                    else receive_opt["title"]
                )
            sel_slot_idx = next((i for i, (_, _, _, _, is_sel) in enumerate(icon_slots) if is_sel), None) if want_deck_grid else None
            preview_is_customized = False
            if is_deck_trade_or_delete and not is_drink_reward and not is_customize_flow:
                if h >= 400 and w >= 300:
                    # Prefer un-occluded large detail-box icon (lp) & top 'before' icon (tp) for exact selected card
                    lp = frame[int(h * 0.165) : int(h * 0.265), int(w * 0.045) : int(w * 0.255)]
                    tp = frame[int(h * 0.055) : int(h * 0.155), int(w * 0.230) : int(w * 0.440)] if "チェンジ" in norm_joined else None
                    if lp.size > 0 and float(lp.std()) >= 15.0:
                        if self._has_card_customization_badge(lp) or (tp is not None and self._has_card_customization_badge(tp)):
                            preview_is_customized = True
                    elif sel_slot_idx is not None and sel_slot_idx < len(grid_custom_flags) and grid_custom_flags[sel_slot_idx]:
                        preview_is_customized = True
                elif sel_slot_idx is not None and sel_slot_idx < len(grid_custom_flags) and grid_custom_flags[sel_slot_idx]:
                    preview_is_customized = True
            cust_tag = " 【已加点强化(カスタマイズ済)·换牌不继承!】" if preview_is_customized else ""
            options.append({
                "choice_id": len(options),
                **id_extra,
                **({"slot_idx": sel_slot_idx} if sel_slot_idx is not None else {}),
                **({"is_customized": True} if preview_is_customized else {}),
                "name": act_name,
                "title": act_title,
                "description": (
                    f"{preview_effect.replace(chr(10), ' / ')}{cust_tag} (点击{receive_opt['title']}确认)"
                    if preview_effect
                    else f"奖励界面确认按钮: {receive_opt['title']}{cust_tag}"
                ),
                "bbox": receive_opt["bbox"],
            })
            # Also allow switching to another thumbnail if multiple candidates exist
            if len(icon_slots) > 1:
                if want_deck_grid:
                    # Keep full stable slot list so agent can scan cleanly and return to any best_grid_slot
                    unselected_idxs = list(range(len(icon_slots)))[:24]
                else:
                    unselected_idxs = [i for i, (_, _, _, _, is_sel) in enumerate(icon_slots) if not is_sel]
                for idx in unselected_idxs:
                    x1, y1, x2, y2, _is_sel = icon_slots[idx]
                    cd = slot_drinks[idx] if idx < len(slot_drinks) else None
                    is_slot_cust = bool(idx < len(grid_custom_flags) and grid_custom_flags[idx])
                    if is_drink_reward:
                        sw_name = f"预览候选饮料{idx + 1}"
                        sw_title = f"切换选中第{idx + 1}瓶候选饮料《{cd['name']}》" if cd else f"预览第{idx + 1}瓶候选饮料"
                        sw_desc = (
                            f"切换查看并选中第{idx + 1}瓶候选饮料《{cd['name']}》: {(cd.get('description') or '').replace(chr(10), ' / ')}"
                            if cd
                            else f"切换查看并选中第{idx + 1}瓶候选P饮料"
                        )
                    elif want_deck_grid:
                        if is_upgrade_flow or is_customize_flow or is_move_to_hand_flow:
                            sw_name = f"预览第{idx + 1}张待选卡"
                            sw_title = f"切换选中卡库第{idx + 1}张待选卡"
                            sw_desc = (
                                "切换查看并选中该卡库技能卡加入手牌"
                                if is_move_to_hand_flow
                                else "切换查看并选中该卡库技能卡进行強化/カスタマイズ"
                            )
                        else:
                            is_gray_low = idx < len(grid_gray_flags) and grid_gray_flags[idx]
                            sw_name = f"预览第{idx + 1}张待选卡"
                            if is_slot_cust:
                                sw_title = f"切换选中卡库第{idx + 1}张卡 【已加点强化·勿换出】"
                                sw_desc = "已加点强化(カスタマイズ済)卡牌，换牌/删除不会继承加点，请勿换出"
                            else:
                                sw_title = (
                                    f"改选底部暗灰低稀有度卡 #{idx + 1} (疑似基本卡·推荐换出)"
                                    if is_gray_low
                                    else f"切换选中卡库第{idx + 1}张卡"
                                )
                                sw_desc = (
                                    "位于卡库最底部的暗灰N/基本卡，换出几乎无损(推荐先点看确认再点チェンジ)"
                                    if is_gray_low
                                    else "切换查看并选中该卡库卡牌"
                                )
                    else:
                        sw_name = f"预览第{idx + 1}张候选卡"
                        sw_title = sw_name
                        sw_desc = f"切换查看并选中第{idx + 1}张候选技能卡"
                    options.append({
                        "choice_id": len(options),
                        "slot_idx": idx,
                        **({"drink_id": int(cd["id"])} if cd else {}),
                        **({"is_customized": True} if is_slot_cust else {}),
                        "name": sw_name,
                        "title": sw_title,
                        "description": sw_desc,
                        "bbox": BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2).model_dump(),
                    })
            if (want_deck_grid or is_deck_trade_or_delete) and cancel_opt:
                options.append({
                    "choice_id": len(options),
                    "name": cancel_opt["name"],
                    "title": cancel_opt["title"],
                    "description": "若无可换出的多余卡，放弃チェンジ并返回",
                    "bbox": cancel_opt["bbox"],
                })
            if dont_take_ln is not None and dont_take_bbox is not None:
                options.append({
                    "choice_id": len(options),
                    "name": "受け取らない",
                    "title": "勾选『受け取らない』放弃领取新饮料",
                    "description": "不领取任何新饮料，保留当前饮料并继续",
                    "bbox": dont_take_bbox,
                })
        elif icon_slots:
            # No card/drink selected yet (or current slot disabled) -> pick an unselected slot
            has_unselected = any(not s[4] for s in icon_slots)
            iter_slots = list(enumerate(icon_slots))
            if want_deck_grid:
                iter_slots = iter_slots[:24]
            for idx, (x1, y1, x2, y2, is_sel) in iter_slots:
                if is_sel and has_unselected:
                    continue
                cd = slot_drinks[idx] if idx < len(slot_drinks) else None
                if is_drink_reward and cd:
                    options.append({
                        "choice_id": len(options),
                        "slot_idx": idx,
                        "drink_id": int(cd["id"]),
                        "name": f"候选饮料{idx + 1}",
                        "title": f"候选饮料 {idx + 1}: 《{cd['name']}》",
                        "description": (cd.get("description") or "").replace("\n", " / ") or "点击预览并选中该候选P饮料",
                        "bbox": BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2).model_dump(),
                    })
                elif want_deck_grid:
                    if is_upgrade_flow or is_customize_flow or is_move_to_hand_flow:
                        options.append({
                            "choice_id": len(options),
                            "slot_idx": idx,
                            "name": f"待选卡{idx + 1}",
                            "title": f"待选卡 {idx + 1} (点击预览卡牌名称与效果)",
                            "description": (
                                "点击预览并选中该卡库技能卡加入手牌"
                                if is_move_to_hand_flow
                                else "点击预览并选中该卡库技能卡进行強化/カスタマイズ"
                            ),
                            "bbox": BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2).model_dump(),
                        })
                    else:
                        is_gray_low = idx < len(grid_gray_flags) and grid_gray_flags[idx]
                        is_slot_cust = bool(idx < len(grid_custom_flags) and grid_custom_flags[idx])
                        options.append({
                            "choice_id": len(options),
                            "slot_idx": idx,
                            **({"is_customized": True} if is_slot_cust else {}),
                            "name": f"换出候选卡{idx + 1}",
                            "title": (
                                f"待选卡 {idx + 1} 【已加点强化·勿换出】"
                                if is_slot_cust
                                else (
                                    f"待换出卡 {idx + 1} (底部暗灰基本卡区域·优先选看!)"
                                    if is_gray_low
                                    else f"待选卡 {idx + 1} (点击预览卡牌名称与效果)"
                                )
                            ),
                            "description": (
                                "已加点强化(カスタマイズ済)卡牌，换牌/删除不会继承加点，请勿换出"
                                if is_slot_cust
                                else (
                                    "卡库底部低稀有度/基本卡槽位，点击预览名称确认后点チェンジ换出"
                                    if is_gray_low
                                    else "点击预览并选中该卡库技能卡"
                                )
                            ),
                            "bbox": BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2).model_dump(),
                        })
                else:
                    slot_num = self._ocr_crop_upscaled(frame, (x1, y2 - 42, x2, y2), scale=2, use_en=True, mode="white", pad=10)
                    hint = f"体力/数值角标 {slot_num}" if slot_num else "点击预览名称与效果并选中"
                    item_lbl = "候选饮料" if is_drink_reward else "候选卡"
                    options.append({
                        "choice_id": len(options),
                        "name": f"{item_lbl}{idx + 1}",
                        "title": f"{item_lbl} {idx + 1}",
                        "description": hint,
                        "bbox": BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2).model_dump(),
                    })
            if (want_deck_grid or is_deck_trade_or_delete) and cancel_opt:
                options.append({
                    "choice_id": len(options),
                    "name": cancel_opt["name"],
                    "title": cancel_opt["title"],
                    "description": "取消操作并返回",
                    "bbox": cancel_opt["bbox"],
                })
        else:
            # Fallback: Match customization perks OR candidate card titles in upper-middle band
            seen_ids = set()
            if is_customize_flow:
                perk_kws = ("パラメータ", "好印象", "やる気", "元気", "コスト", "使用数", "消費", "追加", "開始時", "スキルカード", "レッスン中1回", "+", "＋")
                for ln in lines:
                    cy = (ln["bbox"][1] + ln["bbox"][3]) // 2
                    txt = ln["text"].strip()
                    if (
                        int(h * 0.25) <= cy <= int(h * 0.78)
                        and any(k in txt for k in perk_kws)
                        and not any(bad in txt for bad in ("カスタマイズを", "カスタマイズする", "Pポイント", "所持", "閉じる", "キャンセル", "戻る", "選んで"))
                        and 2 <= len(txt) <= 32
                    ):
                        bx1, by1, bx2, by2 = ln["bbox"]
                        options.append({
                            "choice_id": len(options),
                            "name": f"カスタム_{txt[:16]}",
                            "title": f"自定义词条: {txt}",
                            "description": f"选中该自定义强化词条 ({txt})",
                            "bbox": BoundingBox(
                                xmin=max(40, int(w * 0.10)),
                                ymin=max(10, by1 - 22),
                                xmax=min(w - 40, int(w * 0.90)),
                                ymax=min(h - 10, by2 + 22),
                            ).model_dump(),
                        })
                if receive_opt and receive_active:
                    options.append({
                        "choice_id": len(options),
                        "name": receive_opt["name"],
                        "title": f"确认应用当前自定义 ({receive_opt['title']})",
                        "description": f"点击底部确认按钮应用自定义改造 ({receive_opt['title']})",
                        "bbox": receive_opt["bbox"],
                    })
                if cancel_opt:
                    options.append({
                        "choice_id": len(options),
                        "name": cancel_opt["name"],
                        "title": cancel_opt["title"],
                        "description": "取消当前自定义改造并返回",
                        "bbox": cancel_opt["bbox"],
                    })
            if not options:
                for ln in lines:
                    cy = (ln["bbox"][1] + ln["bbox"][3]) // 2
                    if int(h * 0.15) <= cy <= int(h * 0.75):
                        matched = self.match_card_by_name(ln["text"], plan=plan)
                        if matched and matched["id"] not in seen_ids:
                            seen_ids.add(matched["id"])
                            bx1, by1, bx2, by2 = ln["bbox"]
                            options.append({
                                "choice_id": len(options),
                                "card_id": matched["id"],
                                "name": matched["name"],
                                "title": matched["name"],
                                "description": matched.get("description") or "",
                                "bbox": BoundingBox(
                                    xmin=max(10, bx1 - 30),
                                    ymin=max(10, by1 - 20),
                                    xmax=min(w - 10, bx2 + 30),
                                    ymax=min(h - 10, by2 + 35),
                                ).model_dump(),
                            })
                if receive_opt and receive_active:
                    options.append({
                        "choice_id": len(options),
                        "name": receive_opt["name"],
                        "title": receive_opt["title"],
                        "description": f"奖励界面按钮: {receive_opt['title']}",
                        "bbox": receive_opt["bbox"],
                    })
                if dont_take_ln is not None and dont_take_bbox is not None:
                    options.append({
                        "choice_id": len(options),
                        "name": "受け取らない",
                        "title": "勾选『受け取らない』放弃领取",
                        "description": "不领取新饮料，保留当前饮料并继续",
                        "bbox": dont_take_bbox,
                    })

        if reroll_opt:
            options.append({
                "choice_id": len(options),
                "name": reroll_opt["name"],
                "title": reroll_opt["title"],
                "description": "使用免费次数重抽当前候选卡牌 (不消耗ジュエル)",
                "bbox": reroll_opt["bbox"],
            })
        if skip_opt:
            options.append({
                "choice_id": len(options),
                "name": skip_opt["name"],
                "title": skip_opt["title"],
                "description": "跳过领取 (スキップ)",
                "bbox": skip_opt["bbox"],
            })
        if not options:
            circle_x_btn = self._detect_bottom_circle_x_close(frame, w, h)
            if circle_x_btn is not None:
                options.append(circle_x_btn)
        if not options:
            for ln in lines:
                txt = ln["text"].strip()
                if any(k in txt for k in ("キャンセル", "戻る")) and len(txt) <= 8 and ln["bbox"][1] > int(h * 0.72):
                    bx1, by1, bx2, by2 = ln["bbox"]
                    options.append({
                        "choice_id": 0,
                        "name": txt,
                        "title": txt,
                        "description": f"取消并返回 ({txt})",
                        "bbox": BoundingBox(xmin=max(10, bx1 - 25), ymin=max(10, by1 - 15), xmax=min(w - 10, bx2 + 25), ymax=min(h - 10, by2 + 15)).model_dump(),
                    })
                    break
        if not options:
            options = self._extract_bottom_and_list_options(lines, w, h, frame=frame)
        if held_drinks and not (dont_take_ln is not None and dont_take_checked):
            self._append_banned_drink_discards(
                options,
                held_drinks,
                banned_ids=self.banned_drink_ids,
                banned_names=self._banned_drink_names,
            )

        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
        return ScreenObservation(
            scene_type="card_reward",
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            turns_remaining=int(self._last_turns_remaining) if (is_move_to_hand_flow and self._last_turns_remaining is not None) else 10,
            week_activity=f"{meta['week_label'] or ctx['week_node'].title} | 持有P点:{meta['p_points']}",
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=meta["p_points"],
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            drinks=held_drinks,
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    @staticmethod
    def _collect_shop_feature_lines(
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
        ocr_obj: Any,
        *,
        fy1_ratio: float,
        fy2_ratio: float,
        fx1_ratio: float,
        fx2_ratio: float,
    ) -> List[Dict[str, Any]]:
        """Run multi-pass OCR (raw, dark-on-light, white-on-color) over the shop feature button band."""
        h, w = frame.shape[:2]
        feat_lines: List[Dict[str, Any]] = [
            ln for ln in lines if int(h * 0.56) <= ((ln["bbox"][1] + ln["bbox"][3]) // 2) <= int(h * 0.83)
        ]
        if h >= 400 and w >= 300:
            fy1, fx1 = int(h * fy1_ratio), int(w * fx1_ratio)
            fy2, fx2 = int(h * fy2_ratio), int(w * fx2_ratio)
            feat_crop = frame[fy1:fy2, fx1:fx2]
            if feat_crop.size > 0:
                from PIL import ImageOps
                fgray = feat_crop.mean(axis=2)
                sat_bg = (feat_crop.max(axis=2).astype(np.int16) - feat_crop.min(axis=2).astype(np.int16)) >= 30
                dil_bg = cv2.dilate(sat_bg.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
                white_on_color = (
                    (feat_crop[:, :, 0] > 205)
                    & (feat_crop[:, :, 1] > 205)
                    & (feat_crop[:, :, 2] > 205)
                    & dil_bg
                )
                pass_imgs = [
                    Image.fromarray(feat_crop).resize((feat_crop.shape[1] * 2, feat_crop.shape[0] * 2), Image.Resampling.BICUBIC),
                    Image.fromarray(
                        np.where((fgray < 165)[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                    ).resize((feat_crop.shape[1] * 2, feat_crop.shape[0] * 2), Image.Resampling.BICUBIC),
                ]
                if int(white_on_color.sum()) >= 25:
                    pass_imgs.append(
                        Image.fromarray(
                            np.where(white_on_color[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                        ).resize((feat_crop.shape[1] * 2, feat_crop.shape[0] * 2), Image.Resampling.BICUBIC)
                    )
                for fb2 in pass_imgs:
                    fres = ocr_obj.recognize_array(np.array(ImageOps.expand(fb2, 12, (255, 255, 255))))
                    for fln in fres.get("lines", []):
                        bx1, by1, bx2, by2 = fln["bbox"]
                        feat_lines.append({
                            "text": fln["text"],
                            "bbox": [
                                fx1 + max(0, (bx1 - 12) // 2),
                                fy1 + max(0, (by1 - 12) // 2),
                                fx1 + max(0, (bx2 - 12) // 2),
                                fy1 + max(0, (by2 - 12) // 2),
                            ],
                        })
        return feat_lines

    def _get_shop_card_icon_templates(self, plan: str) -> List[Tuple[int, np.ndarray]]:
        """Lazy-load normalized 32x32 inner-art templates for plan+free skill cards (used on shop shelves)."""
        key = (plan or "all").lower()
        cached = self._shop_card_icon_cache.get(key)
        if cached is not None:
            return cached
        from pathlib import Path

        allowed_plans = {key, "free", "all", ""} if key not in ("all", "") else None
        out: List[Tuple[int, np.ndarray]] = []
        cards_root = Path(__file__).resolve().parents[2] / "cards"
        if not cards_root.exists():
            cards_root = Path("cards")
        if cards_root.exists():
            for p in cards_root.rglob("icons/*.png"):
                try:
                    cid = int(p.stem)
                except ValueError:
                    continue
                row = self.db.get_card_by_id(cid)
                if not row or str(row.get("type") or "").lower() == "trouble":
                    continue
                pl = str(row.get("plan") or "").lower()
                if allowed_plans is not None and pl not in allowed_plans:
                    continue
                try:
                    im = np.array(Image.open(p).convert("RGB"))
                except Exception:
                    continue
                ih, iw = im.shape[:2]
                if ih < 20 or iw < 20:
                    continue
                crop = cv2.resize(
                    im[int(ih * 0.14) : int(ih * 0.76), int(iw * 0.16) : int(iw * 0.84)],
                    (32, 32),
                ).astype(np.float32)
                std = float(crop.std())
                if std < 1e-4:
                    continue
                vec = (crop - float(crop.mean())) / (std + 1e-6)
                out.append((cid, vec))
        self._shop_card_icon_cache[key] = out
        return out

    def _match_square_card_slot(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        plan: str = "all",
        min_score: float = 0.28,
    ) -> Optional[Dict[str, Any]]:
        """Visually identify a skill card icon on a shop shelf (returns DB card row when confident)."""
        patch = frame[y1:y2, x1:x2]
        if patch is None or patch.size == 0:
            return None
        ph, pw = patch.shape[:2]
        if ph < 24 or pw < 24:
            return None
        templates = self._get_shop_card_icon_templates(plan)
        if not templates:
            return None
        crop = cv2.resize(
            patch[int(ph * 0.14) : int(ph * 0.76), int(pw * 0.16) : int(pw * 0.84)],
            (32, 32),
        ).astype(np.float32)
        std = float(crop.std())
        if std < 1e-4:
            return None
        vec = (crop - float(crop.mean())) / (std + 1e-6)
        best_cid = 0
        best_sc = -1.0
        for cid, tvec in templates:
            sc = float((vec * tvec).mean())
            if sc > best_sc:
                best_sc = sc
                best_cid = cid
        if best_cid > 0 and best_sc >= min_score:
            return self.db.get_card_by_id(best_cid)
        return None

    def _scan_shop_grid_and_confirm(
        self,
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
        plan: str,
        meta: Dict[str, Any],
        *,
        cols: List[Tuple[float, float]],
        rows: List[Tuple[str, int, int, int, int]],
        row2_split_cards_drinks: bool,
        sold_out_p90_max: float,
    ) -> Tuple[List[Dict[str, Any]], int, List[RecognizedDrink], Optional[Dict[str, Any]], str]:
        """Scan shop grid slots, prices, selection rim, active 交換する preview, and Exit control."""
        h, w = frame.shape[:2]
        p_pts = int(meta.get("p_points") or 0)
        options: List[Dict[str, Any]] = []
        full_joined_shop = "".join(ln["text"] for ln in lines)

        exchange_opt: Optional[Dict[str, Any]] = None
        exchange_active = False
        exit_opt: Optional[Dict[str, Any]] = None
        for ln in lines:
            txt = ln["text"]
            bx1, by1, bx2, by2 = ln["bbox"]
            if "交換する" in txt and by1 > int(h * 0.72):
                btn_p = frame[max(0, by1 - 15) : min(h, by2 + 15), max(0, bx1 - 45) : min(w, bx2 + 45)]
                exchange_active = (float(btn_p[:, :, 0].mean()) > 218.0) if btn_p.size else False
                exchange_opt = {
                    "name": "交換する",
                    "title": "確認交換 (購買已選商品)",
                    "bbox": BoundingBox(xmin=max(10, bx1 - 45), ymin=max(10, by1 - 15), xmax=min(w - 10, bx2 + 45), ymax=min(h - 10, by2 + 15)).model_dump(),
                }
            elif ("終了" in txt or "次へ" in txt) and len(txt) <= 8 and by1 > int(h * 0.72):
                exit_name = "次へ" if "次へ" in txt else "終了"
                exit_opt = {
                    "name": exit_name,
                    "title": f"{exit_name} (离开商店/进入下一阶段)",
                    "bbox": BoundingBox(xmin=max(10, bx1 - 30), ymin=max(10, by1 - 20), xmax=min(w - 5, max(bx2 + 30, int(w * 0.96))), ymax=min(h - 10, by2 + 20)).model_dump(),
                }

        px_scale = (w / 720.0) * (h / 1600.0)
        x_price_off = int(round(w * 0.035))
        rim_thresh = int(round(900 * px_scale))
        grid_entries: List[Tuple[str, int, int, int, int, int, int, int, bool]] = []
        for row_idx, (row_kind, iy1, iy2, py1, py2) in enumerate(rows):
            for ci, (cx1f, cx2f) in enumerate(cols):
                if row2_split_cards_drinks and row_idx == 1:
                    if ci in (0, 1):
                        kind, slot_idx = "スキルカード", 4 + ci
                    else:
                        kind, slot_idx = "Pドリンク", ci
                else:
                    kind, slot_idx = row_kind, ci
                x1, x2 = int(w * cx1f), int(w * cx2f)
                slot_interior = frame[iy1 + 8 : max(iy1 + 9, iy2 - 8), x1 + 8 : max(x1 + 9, x2 - 8)]
                if slot_interior.size > 0:
                    slot_mean = float(slot_interior.mean())
                    slot_p90 = float(np.percentile(slot_interior, 90))
                    if slot_mean < 182.0 and slot_p90 <= sold_out_p90_max:
                        continue
                ptxt = self._ocr_crop_upscaled(frame, (x1 + x_price_off, py1, x2, py2), scale=2, use_en=True, mode="raw", pad=10)
                nums = [int(n) for n in re.findall(r"\d{2,3}", ptxt) if 10 <= int(n) <= 500]
                if not nums:
                    ptxt2 = self._ocr_crop_upscaled(
                        frame, (x1 + x_price_off, py1, x2, min(h, py2 + int(h * 0.006))), scale=3, use_en=True, mode="dark_on_light", pad=12
                    )
                    nums = [int(n) for n in re.findall(r"\d{2,3}", ptxt2) if 10 <= int(n) <= 500]
                if not nums:
                    for fln in lines:
                        fbx1, fby1, fbx2, fby2 = fln["bbox"]
                        fcx, fcy = (fbx1 + fbx2) // 2, (fby1 + fby2) // 2
                        if (x1 - 16) <= fcx <= (x2 + 16) and (py1 - 14) <= fcy <= (py2 + 14):
                            nums.extend(
                                int(n) for n in re.findall(r"\d{2,3}", fln["text"]) if 10 <= int(n) <= 500
                            )
                price = nums[-1] if nums else 999
                is_red_unaffordable = False
                if price < 999:
                    sale_band = frame[max(0, py1 - int(h * 0.018)) : py1 + 2, x1:x2]
                    pink_sale = bool(
                        sale_band.size > 0
                        and int(((sale_band[:, :, 0] > 210) & (sale_band[:, :, 1] < 110) & (sale_band[:, :, 2] > 90)).sum()) >= int(round(25 * px_scale))
                    )
                    p_patch = frame[py1 + 2 : max(py1 + 3, py2 - 2), x1 + int((x2 - x1) * 0.50) : max(x1 + 10, x2 - 4)]
                    if p_patch.size > 0 and not pink_sale:
                        red_px = int(((p_patch[:, :, 0] > 200) & (p_patch[:, :, 1] < 115) & (p_patch[:, :, 2] < 115)).sum())
                        dark_px = int(((p_patch[:, :, 0] < 110) & (p_patch[:, :, 1] < 110) & (p_patch[:, :, 2] < 110)).sum())
                        if red_px >= int(round(25 * px_scale)) and dark_px <= int(round(5 * px_scale)):
                            is_red_unaffordable = True
                rim_mask = np.zeros((h, w), dtype=bool)
                rim_mask[max(0, iy1 - 8) : min(h, iy2 + 8), max(0, x1 - 8) : min(w, x2 + 8)] = True
                rim_mask[iy1 + 10 : max(iy1 + 10, iy2 - 10), x1 + 10 : max(x1 + 10, x2 - 10)] = False
                rim_orange = int(
                    ((frame[:, :, 0] > 230) & (frame[:, :, 1] > 120) & (frame[:, :, 1] < 180) & (frame[:, :, 2] < 80) & rim_mask).sum()
                )
                grid_entries.append((kind, slot_idx, x1, iy1, x2, iy2, price, rim_orange, is_red_unaffordable))

        max_rim = max((g[7] for g in grid_entries), default=0)
        selected_key: Optional[Tuple[str, int]] = None
        selected_price: Optional[int] = None
        selected_coords: Optional[Tuple[int, int, int, int]] = None
        if max_rim >= rim_thresh:
            for kind, ci, x1, iy1, x2, iy2, price, rim_orange, _is_red in grid_entries:
                if rim_orange == max_rim:
                    selected_key = (kind, ci)
                    selected_price = price
                    selected_coords = (x1, iy1, x2, iy2)
                    self._shop_visited_slots.add(f"{kind}:{ci}")
                    break
        elif exchange_opt is not None and self._shop_last_clicked_slot and ":" in self._shop_last_clicked_slot:
            l_kind, l_ci_str = self._shop_last_clicked_slot.split(":", 1)
            if l_ci_str.isdigit():
                l_ci = int(l_ci_str)
                for kind, ci, x1, iy1, x2, iy2, price, _rim_orange, _is_red in grid_entries:
                    if kind == l_kind and ci == l_ci:
                        selected_key = (kind, ci)
                        selected_price = price
                        selected_coords = (x1, iy1, x2, iy2)
                        self._shop_visited_slots.add(f"{kind}:{ci}")
                        break

        cap_count = 4 if self.scenario_id in ("hif_honsen", "hif") else 3
        held_drinks = self._recognize_bottom_drinks(frame, plan)
        # Any red unaffordable price on the shop grid proves p_pts < price!
        min_red_price = min((g[6] for g in grid_entries if g[8] and 10 <= g[6] < 999), default=None)
        if min_red_price is not None and p_pts >= min_red_price:
            p_pts = max(0, min_red_price - 1)
        if (
            selected_key is not None
            and exchange_opt is not None
            and not exchange_active
            and selected_price is not None
            and 25 <= selected_price < 999
        ):
            if selected_key[0] == "スキルカード" or len(held_drinks) < cap_count:
                p_pts = min(p_pts, max(0, selected_price - 1))
        self._last_p_points = p_pts

        if exchange_opt and exchange_active and (selected_price is None or p_pts >= selected_price):
            pv_res = self.ocr.recognize_array(frame[int(h * 0.20) : int(h * 0.39), int(w * 0.08) : int(w * 0.92)])
            pv_lines = [l["text"].strip() for l in pv_res.get("lines", []) if l["text"].strip()]
            pv_txt = " | ".join(pv_lines)
            pv_name = pv_lines[0] if pv_lines else ""
            ex_extra: Dict[str, Any] = {}
            matched_card = None
            if selected_key is None or selected_key[0] == "スキルカード":
                for cand_card_txt in ([pv_name] + pv_lines[:3]):
                    if cand_card_txt:
                        matched_card = self.match_card_by_name(cand_card_txt, plan=plan)
                        if matched_card:
                            break
            if matched_card:
                ex_extra["card_id"] = matched_card["id"]
                pv_name = matched_card["name"]
            else:
                matched_dr = None
                for cand_dr_txt in ([pv_name] + pv_lines[:3]):
                    if cand_dr_txt:
                        matched_dr = self.match_drink_by_name(cand_dr_txt, plan=plan)
                        if matched_dr is not None:
                            break
                if matched_dr is None and selected_coords is not None and selected_key is not None and selected_key[0] == "Pドリンク":
                    sx1, siy1, sx2, siy2 = selected_coords
                    matched_dr = self._match_square_drink_slot(frame, sx1, siy1, sx2, siy2)
                if matched_dr is not None:
                    ex_extra["drink_id"] = int(matched_dr["id"])
                    pv_name = matched_dr["name"]
            if selected_price is not None and selected_price < 999:
                ex_extra["price"] = selected_price
            price_tag = f" (-{selected_price}P)" if (selected_price is not None and selected_price < 999) else ""
            options.append({
                "choice_id": len(options),
                **ex_extra,
                "name": "交換する",
                "title": f"【确认购买当前已选中商品!】交換する《{pv_name or '已选商品'}》{price_tag}",
                "description": f"确认付款兑换当前已选中的《{pv_name or '商品'}》{price_tag}: {pv_txt[:120] or '已选商品'} (注意: 点下方其他栏位仅切换预览，唯有此项会真实购买!)",
                "bbox": exchange_opt["bbox"],
            })
        elif (
            len(held_drinks) >= cap_count
            and selected_key is not None
            and selected_key[0] == "Pドリンク"
            and not exchange_active
            and selected_price is not None
            and p_pts >= selected_price
        ):
            pv_res = self.ocr.recognize_array(frame[int(h * 0.20) : int(h * 0.39), int(w * 0.08) : int(w * 0.92)])
            pv_lines = [l["text"].strip() for l in pv_res.get("lines", []) if l["text"].strip()]
            pv_txt = " | ".join(pv_lines)
            pv_name = pv_lines[0] if pv_lines else "商店P饮料"
            matched_dr = self.match_drink_by_name(pv_name, plan=plan)
            if matched_dr is not None:
                pv_name = matched_dr["name"]
            for di, d in enumerate(held_drinks, 1):
                options.append({
                    "choice_id": len(options),
                    "drink_id": d.drink_id,
                    "name": f"破棄候補_{d.drink_id}",
                    "title": f"点击饮料栏旧饮料{di}《{d.name}》打开详情删除(捨てる/破棄)",
                    "description": (
                        f"当前{cap_count}/{cap_count}满槽无法直接购买商店饮料《{pv_name}》(-{selected_price}P: {pv_txt[:60]})！"
                        f"可先点击左下角旧饮料《{d.name}》({d.dsl_summary})点『捨てる/破棄』删除腾位再购买，或选其他商品/終了"
                    ),
                    "bbox": d.bbox.model_dump(),
                })

        for kind, ci, x1, iy1, x2, iy2, price, _rim, is_red_unaff in grid_entries:
            slot_key = f"{kind}:{ci}"
            if selected_key == (kind, ci) or slot_key in self._shop_visited_slots:
                continue
            if (not is_red_unaff) and p_pts >= price:
                slot_dr = self._match_square_drink_slot(frame, x1, iy1, x2, iy2) if kind == "Pドリンク" else None
                slot_card = (
                    self._match_square_card_slot(frame, x1, iy1, x2, iy2, plan=plan)
                    if kind == "スキルカード"
                    else None
                )
                item_label = (
                    f"《{slot_dr['name']}》 "
                    if slot_dr
                    else (f"《{slot_card['name']}》 " if slot_card else "")
                )
                options.append({
                    "choice_id": len(options),
                    "shop_slot_key": slot_key,
                    **({"drink_id": int(slot_dr["id"])} if slot_dr else {}),
                    **({"card_id": int(slot_card["id"])} if slot_card else {}),
                    "name": f"{kind}{ci + 1}",
                    "title": f"切换预览 {kind} #{ci + 1} {item_label}({price}P)".replace("  ", " "),
                    "price": price,
                    "description": f"{kind} 栏位#{ci + 1} {item_label}| 价格 {price}P <= 当前{p_pts}P (点击选中并预览详情，不直接扣P)",
                    "bbox": BoundingBox(xmin=x1, ymin=iy1, xmax=x2, ymax=iy2).model_dump(),
                })

        return options, p_pts, held_drinks, exit_opt, full_joined_shop

    def _finalize_shop_observation(
        self,
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
        plan: str,
        meta: Dict[str, Any],
        options: List[Dict[str, Any]],
        p_pts: int,
        held_drinks: List[RecognizedDrink],
        exit_opt: Optional[Dict[str, Any]],
        shop_label: str,
    ) -> ScreenObservation:
        h, w = frame.shape[:2]
        if held_drinks:
            self._append_banned_drink_discards(
                options,
                held_drinks,
                banned_ids=self.banned_drink_ids,
                banned_names=self._banned_drink_names,
            )

        exit_bbox = (
            exit_opt["bbox"]
            if exit_opt
            else BoundingBox(xmin=int(w * 0.74), ymin=int(h * 0.79), xmax=int(w * 0.96), ymax=int(h * 0.85)).model_dump()
        )
        if not options:
            options = [{
                "choice_id": 0,
                "name": exit_opt["name"] if exit_opt else "終了",
                "title": exit_opt["title"] if exit_opt else f"終了 (离开{shop_label})",
                "description": f"持有{p_pts}P不足或已无可买项，直接结束离开{shop_label}",
                "bbox": exit_bbox,
            }]
        elif exit_opt or any(str(o.get("name") or "").startswith("破棄候補_") for o in options):
            options.append({
                "choice_id": len(options),
                "name": exit_opt["name"] if exit_opt else "終了",
                "title": exit_opt["title"] if exit_opt else f"終了 (离开{shop_label})",
                "description": f"离开{shop_label} (当前持有 {p_pts}P)",
                "bbox": exit_bbox,
            })
        for idx, o in enumerate(options):
            o["choice_id"] = idx

        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
        return ScreenObservation(
            scene_type="shop",
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=f"{meta['week_label'] or ctx['week_node'].title} | 持有P点:{p_pts}",
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=p_pts,
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            drinks=held_drinks,
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _parse_consultation_shop_screen(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]
    ) -> ScreenObservation:
        """Parse W6 / standard 相談 shop with its own 4-card + 4-drink grid and feature buttons."""
        h, w = frame.shape[:2]
        cols = [(0.08, 0.27), (0.29, 0.48), (0.505, 0.695), (0.715, 0.91)]
        rows = [
            ("スキルカード", int(h * 0.395), int(h * 0.482), int(h * 0.485), int(h * 0.510)),
            ("Pドリンク", int(h * 0.512), int(h * 0.598), int(h * 0.599), int(h * 0.638)),
        ]
        options, p_pts, held_drinks, exit_opt, _ = self._scan_shop_grid_and_confirm(
            frame,
            lines,
            plan,
            meta,
            cols=cols,
            rows=rows,
            row2_split_cards_drinks=False,
            sold_out_p90_max=229.9,
        )

        feat_lines = self._collect_shop_feature_lines(
            frame,
            lines,
            self.ocr,
            fy1_ratio=0.60,
            fy2_ratio=0.79,
            fx1_ratio=0.04,
            fx2_ratio=0.96,
        )
        seen_feats: Set[str] = set()

        def _is_bright_box(box: Tuple[int, int, int, int], fallback: bool) -> bool:
            patch = frame[box[1] : box[3], box[0] : box[2]]
            if not patch.size:
                return fallback
            return float(patch.mean()) >= 150.0 or float(np.percentile(patch, 85)) >= 195.0

        def _lookup_side_price(by1: int, by2: int, x_min: int, x_max: int) -> Optional[int]:
            cy_this = (by1 + by2) // 2
            for oln in feat_lines:
                otxt = oln["text"].strip()
                obx1, oby1, obx2, oby2 = oln["bbox"]
                ocx = (obx1 + obx2) // 2
                ocy = (oby1 + oby2) // 2
                if abs(ocy - cy_this) <= 32 and x_min <= ocx <= x_max:
                    m_adj = re.search(r"^\s*P?\s*(\d{2,3})\s*P?\s*$", otxt)
                    if not m_adj and len(otxt) <= 10 and not any(k in otxt for k in ("強化", "削除", "カード")):
                        m_adj = re.search(r"(\d{2,3})", otxt)
                    if m_adj and 10 <= int(m_adj.group(1)) <= 500:
                        return int(m_adj.group(1))
            return None

        for ln in feat_lines:
            txt = ln["text"].strip()
            norm_t = txt.replace("チェンツ", "チェンジ").replace("カスタマイス", "カスタマイズ")
            # Skip the white helper tooltip banner ("持っているスキルカードの強化や削除ができます。") above the real buttons!
            if (
                "持っている" in norm_t
                or "強化や削除" in norm_t
                or "ができます" in norm_t
                or "スキルカードの強化" in norm_t
            ):
                continue
            has_both_upg_del = ("強化" in norm_t and "削除" in norm_t)
            if len(norm_t) > (34 if has_both_upg_del else 18):
                continue
            if "交換" in norm_t or "終了" in norm_t or "次へ" in norm_t or "チェンジで獲得" in norm_t:
                continue
            bx1, by1, bx2, by2 = ln["bbox"]
            btn_box = (max(10, bx1 - 30), max(int(h * 0.56), by1 - 16), min(w - 10, bx2 + 30), min(int(h * 0.82), by2 + 16))
            btn_bright = _is_bright_box(btn_box, True)
            m_price = re.search(r"(\d{2,3})\s*P?", norm_t)
            feat_price = int(m_price.group(1)) if m_price and 10 <= int(m_price.group(1)) <= 500 else None
            if feat_price is None:
                cy_this = (by1 + by2) // 2
                for oln in feat_lines:
                    otxt = oln["text"].strip()
                    obx1, oby1, _obx2, oby2 = oln["bbox"]
                    ocy_other = (oby1 + oby2) // 2
                    if abs(ocy_other - cy_this) <= 24 and -10 <= obx1 - bx1 <= int(w * 0.35):
                        m_adj = re.search(r"^\s*P?\s*(\d{2,3})\s*P?\s*$", otxt)
                        if m_adj and 10 <= int(m_adj.group(1)) <= 500:
                            feat_price = int(m_adj.group(1))
                            break

            if ("セレクトチェンジ" in norm_t or "チェンジ" in norm_t) and "チェンジ" not in seen_feats and "セレクトチェンジ" not in self._shop_visited_slots and "チェンジ" not in self._shop_visited_slots:
                if btn_bright and (feat_price is None or p_pts >= feat_price):
                    seen_feats.add("チェンジ")
                    options.append({
                        "choice_id": len(options),
                        "shop_slot_key": "セレクトチェンジ",
                        "name": "セレクトチェンジ",
                        "title": "セレクトチェンジ (1换1换牌·不增加牌库张数)",
                        "description": "走换牌功能1换1替换手头卡，不增加牌库总数，适合压缩循环构建",
                        **({"price": feat_price} if feat_price is not None else {}),
                        "bbox": BoundingBox(xmin=btn_box[0], ymin=btn_box[1], xmax=btn_box[2], ymax=btn_box[3]).model_dump(),
                    })
                continue

            if (
                "カスタマイズ" in norm_t
                and "確認" not in norm_t
                and "カスタマイズ" not in seen_feats
                and "カスタマイズ" not in self._shop_visited_slots
            ):
                if btn_bright and (feat_price is None or p_pts >= feat_price):
                    seen_feats.add("カスタマイズ")
                    options.append({
                        "choice_id": len(options),
                        "shop_slot_key": "カスタマイズ",
                        "name": "カスタマイズ",
                        "title": "スキルカードカスタマイズ (用余钱自定义强化主力卡牌!)",
                        "description": "用富裕P点进入自定义改造，强化主力卡牌数值/减耗/使用数",
                        **({"price": feat_price} if feat_price is not None else {}),
                        "bbox": BoundingBox(xmin=btn_box[0], ymin=btn_box[1], xmax=btn_box[2], ymax=btn_box[3]).model_dump(),
                    })
                continue

            if (
                "強化" in norm_t
                and "強化済" not in norm_t
                and "強化前" not in norm_t
                and "強化" not in seen_feats
                and "強化" not in self._shop_visited_slots
            ):
                if has_both_upg_del or bx2 > int(w * 0.50):
                    u_box = (
                        max(10, min(bx1 - 24, int(w * 0.08))),
                        max(int(h * 0.56), by1 - 16),
                        int(w * 0.48),
                        min(int(h * 0.82), by2 + 16),
                    )
                    m_u = re.search(r"強化\D{0,6}(\d{2,3})", norm_t)
                    u_price = (
                        int(m_u.group(1))
                        if m_u and 10 <= int(m_u.group(1)) <= 500
                        else _lookup_side_price(by1, by2, 0, int(w * 0.49))
                    )
                else:
                    u_box = btn_box
                    u_price = feat_price
                if _is_bright_box(u_box, btn_bright) and (u_price is None or p_pts >= u_price):
                    seen_feats.add("強化")
                    options.append({
                        "choice_id": len(options),
                        "shop_slot_key": "強化",
                        "name": "強化",
                        "title": (
                            f"スキルカード強化 ({u_price}P·强化手头未+卡牌)"
                            if u_price is not None
                            else "スキルカード強化 (强化手头未+卡牌)"
                        ),
                        **({"price": u_price} if u_price is not None else {}),
                        "description": f"花费{u_price}P强化一张技能卡(+)" if u_price is not None else "强化一张技能卡(+)",
                        "bbox": BoundingBox(xmin=u_box[0], ymin=u_box[1], xmax=u_box[2], ymax=u_box[3]).model_dump(),
                    })

            if (
                "削除" in norm_t
                and "削除済" not in norm_t
                and "削除" not in seen_feats
                and "削除" not in self._shop_visited_slots
            ):
                if has_both_upg_del or bx1 < int(w * 0.49):
                    d_box = (
                        int(w * 0.52),
                        max(int(h * 0.56), by1 - 16),
                        min(w - 10, max(bx2 + 24, int(w * 0.92))),
                        min(int(h * 0.82), by2 + 16),
                    )
                    m_d = re.search(r"削除\D{0,6}(\d{2,3})", norm_t)
                    d_price = (
                        int(m_d.group(1))
                        if m_d and 10 <= int(m_d.group(1)) <= 500
                        else _lookup_side_price(by1, by2, int(w * 0.50), w)
                    )
                else:
                    d_box = btn_box
                    d_price = feat_price
                if _is_bright_box(d_box, btn_bright) and (d_price is None or p_pts >= d_price):
                    seen_feats.add("削除")
                    options.append({
                        "choice_id": len(options),
                        "shop_slot_key": "削除",
                        "name": "削除",
                        "title": "スキルカード削除 (删除冗余废卡)",
                        "description": "删除多余基本/废卡精简牌库",
                        **({"price": d_price} if d_price is not None else {}),
                        "bbox": BoundingBox(xmin=d_box[0], ymin=d_box[1], xmax=d_box[2], ymax=d_box[3]).model_dump(),
                    })

        # Visual fallback for W6 相談's vivid pink-orange 強化 (left) and purple-blue 削除 (right) buttons
        # at y=0.681..0.730: WinRT OCR often misses white kanji on the colored gradient buttons.
        btn_y1, btn_y2 = int(h * 0.681), int(h * 0.730)
        if "強化" not in seen_feats and "強化" not in self._shop_visited_slots:
            u_box = (int(w * 0.070), btn_y1, int(w * 0.490), btn_y2)
            u_patch = frame[u_box[1] : u_box[3], u_box[0] : u_box[2]]
            if u_patch.size > 0:
                u_sat = float((u_patch.max(axis=2).astype(np.int16) - u_patch.min(axis=2).astype(np.int16)).mean())
                u_r = float(u_patch[:, :, 0].mean())
                u_b = float(u_patch[:, :, 2].mean())
                if u_sat >= 50.0 and (u_r - u_b) >= 25.0:
                    u_price = _lookup_side_price(btn_y1, btn_y2, 0, int(w * 0.49))
                    if u_price is None or p_pts >= u_price:
                        seen_feats.add("強化")
                        options.append({
                            "choice_id": len(options),
                            "shop_slot_key": "強化",
                            "name": "強化",
                            "title": (
                                f"スキルカード強化 ({u_price}P·强化手头未+卡牌)"
                                if u_price is not None
                                else "スキルカード強化 (强化手头未+卡牌)"
                            ),
                            **({"price": u_price} if u_price is not None else {}),
                            "description": f"花费{u_price}P强化一张技能卡(+)" if u_price is not None else "强化一张技能卡(+)",
                            "bbox": BoundingBox(xmin=u_box[0], ymin=u_box[1], xmax=u_box[2], ymax=u_box[3]).model_dump(),
                        })

        if "削除" not in seen_feats and "削除" not in self._shop_visited_slots:
            d_box = (int(w * 0.508), btn_y1, int(w * 0.930), btn_y2)
            d_patch = frame[d_box[1] : d_box[3], d_box[0] : d_box[2]]
            if d_patch.size > 0:
                d_sat = float((d_patch.max(axis=2).astype(np.int16) - d_patch.min(axis=2).astype(np.int16)).mean())
                d_r = float(d_patch[:, :, 0].mean())
                d_b = float(d_patch[:, :, 2].mean())
                if d_sat >= 45.0 and (d_b - d_r) >= 20.0:
                    d_price = _lookup_side_price(btn_y1, btn_y2, int(w * 0.50), w)
                    if d_price is None or p_pts >= d_price:
                        seen_feats.add("削除")
                        options.append({
                            "choice_id": len(options),
                            "shop_slot_key": "削除",
                            "name": "削除",
                            "title": (
                                f"スキルカード削除 ({d_price}P·删除冗余废卡)"
                                if d_price is not None
                                else "スキルカード削除 (删除冗余废卡)"
                            ),
                            "description": f"花费{d_price}P删除一张废卡/干扰卡" if d_price is not None else "删除多余基本/废卡精简牌库",
                            **({"price": d_price} if d_price is not None else {}),
                            "bbox": BoundingBox(xmin=d_box[0], ymin=d_box[1], xmax=d_box[2], ymax=d_box[3]).model_dump(),
                        })

        return self._finalize_shop_observation(
            frame,
            lines,
            plan,
            meta,
            options,
            p_pts,
            held_drinks,
            exit_opt,
            shop_label="相談商店",
        )

    def _parse_interval_shop_screen(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]
    ) -> ScreenObservation:
        """Parse W8 H.I.F インターバル shop with dedicated grid, 強化/カスタマイズ, and aligned リフレッシュ."""
        h, w = frame.shape[:2]
        cols = [(0.0750, 0.2486), (0.2972, 0.4722), (0.5194, 0.6944), (0.7417, 0.9167)]
        rows = [
            ("スキルカード", int(h * 0.4119), int(h * 0.4950), int(h * 0.4963), int(h * 0.5180)),
            ("Pドリンク", int(h * 0.5325), int(h * 0.6156), int(h * 0.6163), int(h * 0.6400)),
        ]
        options, p_pts, held_drinks, exit_opt, full_joined_shop = self._scan_shop_grid_and_confirm(
            frame,
            lines,
            plan,
            meta,
            cols=cols,
            rows=rows,
            row2_split_cards_drinks=True,
            sold_out_p90_max=232.0,
        )

        feat_lines = self._collect_shop_feature_lines(
            frame,
            lines,
            self.ocr,
            fy1_ratio=0.64,
            fy2_ratio=0.81,
            fx1_ratio=0.02,
            fx2_ratio=0.95,
        )
        seen_feats: Set[str] = set()
        norm_shop_all = full_joined_shop.replace("カスタマイス", "カスタマイズ")
        upg_zero_rem = bool(re.search(r"強化可能.{0,6}あと\s*0", norm_shop_all))
        cust_zero_rem = bool(re.search(r"カスタマイズ可能.{0,6}あと\s*0", norm_shop_all))

        for ln in feat_lines:
            txt = ln["text"].strip()
            norm_t = txt.replace("チェンツ", "チェンジ").replace("カスタマイス", "カスタマイズ")
            if len(norm_t) > 18 or "交換" in norm_t or "終了" in norm_t or "次へ" in norm_t or "チェンジで獲得" in norm_t:
                continue
            bx1, by1, bx2, by2 = ln["bbox"]
            btn_box = (max(10, bx1 - 30), max(int(h * 0.56), by1 - 16), min(w - 10, bx2 + 30), min(int(h * 0.82), by2 + 16))
            btn_patch = frame[btn_box[1] : btn_box[3], btn_box[0] : btn_box[2]]
            btn_bright = (
                (float(btn_patch.mean()) >= 150.0 or float(np.percentile(btn_patch, 85)) >= 195.0)
                if btn_patch.size
                else True
            )
            m_price = re.search(r"(\d{2,3})\s*P?", norm_t)
            feat_price = int(m_price.group(1)) if m_price and 10 <= int(m_price.group(1)) <= 500 else None
            if feat_price is None:
                cy_this = (by1 + by2) // 2
                for oln in feat_lines:
                    otxt = oln["text"].strip()
                    obx1, oby1, _obx2, oby2 = oln["bbox"]
                    ocy_other = (oby1 + oby2) // 2
                    if abs(ocy_other - cy_this) <= 24 and -10 <= obx1 - bx1 <= int(w * 0.35):
                        m_adj = re.search(r"^\s*P?\s*(\d{2,3})\s*P?\s*$", otxt)
                        if m_adj and 10 <= int(m_adj.group(1)) <= 500:
                            feat_price = int(m_adj.group(1))
                            break

            if (
                ("カスタマイズ" in norm_t or "カスタマイズ可能" in norm_t)
                and not cust_zero_rem
                and "確認" not in norm_t
                and "カスタマイズ" not in seen_feats
                and "カスタマイズ" not in self._shop_visited_slots
            ):
                c_box = (int(w * 0.444), int(h * 0.6575), int(w * 0.778), int(h * 0.7013))
                c_patch = frame[c_box[1] : c_box[3], c_box[0] : c_box[2]]
                c_bright = (
                    (float(c_patch.mean()) >= 150.0 or float(np.percentile(c_patch, 85)) >= 195.0)
                    if c_patch.size
                    else btn_bright
                )
                if c_bright and (feat_price is None or p_pts >= feat_price):
                    seen_feats.add("カスタマイズ")
                    options.append({
                        "choice_id": len(options),
                        "shop_slot_key": "カスタマイズ",
                        "name": "カスタマイズ",
                        "title": "スキルカードカスタマイズ (用余钱自定义强化主力卡牌!)",
                        "description": "用富裕P点进入自定义改造，强化主力卡牌数值/减耗/使用数",
                        **({"price": feat_price} if feat_price is not None else {}),
                        "bbox": BoundingBox(xmin=c_box[0], ymin=c_box[1], xmax=c_box[2], ymax=c_box[3]).model_dump(),
                    })
            elif (
                ("強化" in norm_t or "強化可能" in norm_t)
                and not upg_zero_rem
                and "強化済" not in norm_t
                and "強化前" not in norm_t
                and "強化" not in seen_feats
                and "強化" not in self._shop_visited_slots
            ):
                u_box = (int(w * 0.078), int(h * 0.6569), int(w * 0.412), int(h * 0.7013))
                u_patch = frame[u_box[1] : u_box[3], u_box[0] : u_box[2]]
                u_bright = (
                    (float(u_patch.mean()) >= 150.0 or float(np.percentile(u_patch, 85)) >= 195.0)
                    if u_patch.size
                    else btn_bright
                )
                u_nums: List[int] = []
                for oln in lines + feat_lines:
                    obx1, oby1, obx2, oby2 = oln["bbox"]
                    ocx, ocy = (obx1 + obx2) // 2, (oby1 + oby2) // 2
                    if int(w * 0.05) <= ocx <= int(w * 0.44) and int(h * 0.65) <= ocy <= int(h * 0.715):
                        u_nums.extend(
                            int(n) for n in re.findall(r"\d{2,3}", oln["text"]) if 10 <= int(n) <= 500
                        )
                req_p = u_nums[-1] if u_nums else (feat_price if feat_price is not None else 100)
                if u_bright and p_pts >= req_p:
                    seen_feats.add("強化")
                    options.append({
                        "choice_id": len(options),
                        "shop_slot_key": "強化",
                        "name": "強化",
                        "title": f"スキルカード強化 ({req_p}P·强化手头未+卡牌)",
                        "price": req_p,
                        "description": f"花费{req_p}P强化一张技能卡(+)",
                        "bbox": BoundingBox(xmin=u_box[0], ymin=u_box[1], xmax=u_box[2], ymax=u_box[3]).model_dump(),
                    })
            elif (
                ("リフレッシュ" in norm_t or "リフレッシ" in norm_t)
                and by1 >= int(h * 0.68)
                and bx1 <= int(w * 0.35)
                and "リフレッシュ" not in seen_feats
            ):
                # Click strictly inside the upper white [リフレッシュ] pill (~y=1207..1246 on 1600p),
                # never down into the gap / non-clickable lower `P 10` price pill (~y=1255..1283).
                if int(h * 0.745) <= by1 <= int(h * 0.782):
                    r_box = (int(w * 0.026), int(h * 0.7544), int(w * 0.276), int(h * 0.7788))
                else:
                    r_box = (
                        max(10, bx1 - 18),
                        max(int(h * 0.68), by1 - 8),
                        min(w - 10, max(bx2 + 18, int(w * 0.276))),
                        min(int(h * 0.80), by2 + 8),
                    )
                r_patch = frame[r_box[1] : r_box[3], r_box[0] : r_box[2]]
                r_bright = (
                    (float(r_patch.mean()) >= 150.0 or float(np.percentile(r_patch, 85)) >= 195.0)
                    if r_patch.size
                    else btn_bright
                )
                r_nums: List[int] = []
                for oln in lines + feat_lines:
                    obx1, oby1, _obx2, _oby2 = oln["bbox"]
                    if obx1 <= int(w * 0.30) and by1 <= oby1 <= by2 + int(h * 0.055):
                        r_nums.extend(
                            int(n) for n in re.findall(r"\d{1,3}", oln["text"]) if 5 <= int(n) <= 300
                        )
                ref_price = r_nums[-1] if r_nums else 10
                ref_price_band = frame[
                    min(h - 1, r_box[3]) : min(h, r_box[3] + int(h * 0.035)),
                    max(0, r_box[0]) : min(w, r_box[2]),
                ]
                ref_red_px = (
                    int(
                        (
                            (ref_price_band[:, :, 0] > 185)
                            & (ref_price_band[:, :, 1] < 105)
                            & (ref_price_band[:, :, 2] < 115)
                        ).sum()
                    )
                    if ref_price_band.size > 0
                    else 0
                )
                ref_price_is_red = ref_red_px >= max(12, int(round(25 * (w * h) / (720.0 * 1600.0))))
                if ref_price_is_red and ref_price >= 10:
                    p_pts = min(p_pts, max(0, ref_price - 1))
                    self._last_p_points = p_pts
                if r_bright and (not ref_price_is_red) and p_pts >= ref_price:
                    seen_feats.add("リフレッシュ")
                    options.append({
                        "choice_id": len(options),
                        "shop_slot_key": "リフレッシュ",
                        "name": "リフレッシュ",
                        "title": f"リフレッシュ ({ref_price}P·刷新商店P饮料与強化/カスタマイズ次数)",
                        "price": ref_price,
                        "description": f"花费{ref_price}P刷新商品行，重新获取P饮料并重置強化/カスタマイズ次数",
                        "bbox": BoundingBox(xmin=r_box[0], ymin=r_box[1], xmax=r_box[2], ymax=r_box[3]).model_dump(),
                    })

        return self._finalize_shop_observation(
            frame,
            lines,
            plan,
            meta,
            options,
            p_pts,
            held_drinks,
            exit_opt,
            shop_label="インターバル商店",
        )

    def _parse_shop_screen(self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]) -> ScreenObservation:
        h = frame.shape[0]
        full_joined_shop = "".join(ln["text"] for ln in lines)
        hdr_shop = str(meta.get("header_label") or "")
        has_interval_cue = (
            ("インターバル" in full_joined_shop)
            or ("インターバル" in hdr_shop)
            or ("リフレッシュ" in full_joined_shop)
            or ("カスタマイズ可能" in full_joined_shop)
            or ("強化可能" in full_joined_shop)
        )
        has_explicit_soudan_cue = (
            any("相談" in str(ln.get("text") or "") and ln["bbox"][1] < int(h * 0.25) for ln in lines)
            or ("セレクトチェンジ" in full_joined_shop)
            or ("チェンジで獲得" in full_joined_shop)
        )
        if has_interval_cue or (
            self.scenario_id in ("hif_honsen", "hif")
            and int(meta.get("week") or 0) == 8
            and not has_explicit_soudan_cue
        ):
            return self._parse_interval_shop_screen(frame, lines, plan, meta)
        return self._parse_consultation_shop_screen(frame, lines, plan, meta)

    def _parse_customize_menu_screen(
        self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]
    ) -> ScreenObservation:
        """Parse the floating カスタマイズメニュー option picker and execute/back/exit controls."""
        h, w = frame.shape[:2]
        options: List[Dict[str, Any]] = []
        exec_opt: Optional[Dict[str, Any]] = None
        back_opt: Optional[Dict[str, Any]] = None
        close_opt: Optional[Dict[str, Any]] = None
        tile_opts: List[Dict[str, Any]] = []

        for ln in lines:
            txt = ln["text"].strip()
            norm_t = txt.replace("カスタマイス", "カスタマイズ")
            bx1, by1, bx2, by2 = ln["bbox"]
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            if "実行する" in norm_t and by1 >= int(h * 0.85):
                btn_p = frame[max(0, by1 - 16) : min(h, by2 + 16), max(0, bx1 - 45) : min(w, bx2 + 45)]
                mean_r = float(btn_p[:, :, 0].mean()) if btn_p.size else 0.0
                mean_b = float(btn_p[:, :, 2].mean()) if btn_p.size else 255.0
                if mean_r > 218.0 and mean_b < 115.0:
                    exec_opt = {
                        "name": "カスタマイズ実行",
                        "title": "実行する (确认执行当前カスタマイズ)",
                        "description": "已选中改造方案，点击実行する完成本次カスタマイズ",
                        "bbox": BoundingBox(
                            xmin=max(10, bx1 - 45),
                            ymin=max(10, by1 - 18),
                            xmax=min(w - 10, bx2 + 45),
                            ymax=min(h - 10, by2 + 18),
                        ).model_dump(),
                    }
            elif "一覧に戻る" in norm_t and by1 >= int(h * 0.85):
                back_opt = {
                    "name": "一覧に戻る",
                    "title": "一覧に戻る (返回卡牌列表)",
                    "description": "返回カスタマイズ卡牌选择列表",
                    "bbox": BoundingBox(
                        xmin=max(10, bx1 - 45),
                        ymin=max(10, by1 - 18),
                        xmax=min(w - 10, bx2 + 45),
                        ymax=min(h - 10, by2 + 18),
                    ).model_dump(),
                }
            elif "終了" in norm_t and len(norm_t) <= 6 and by1 >= int(h * 0.78) and cx >= int(w * 0.70):
                close_opt = {
                    "name": "終了",
                    "title": "✕ 終了 (关闭カスタマイズ并返回商店)",
                    "description": "关闭カスタマイズ面板返回商店主界面",
                    "bbox": BoundingBox(
                        xmin=max(10, bx1 - 30),
                        ymin=max(10, by1 - 16),
                        xmax=min(w - 5, bx2 + 25),
                        ymax=min(h - 10, by2 + 16),
                    ).model_dump(),
                }
            elif (
                int(h * 0.73) <= cy <= int(h * 0.84)
                and cx <= int(w * 0.90)
                and "終了" not in norm_t
                and "カスタマイズメニュー" not in norm_t
                and "あと" not in norm_t
                and not re.match(r"^[⑨◎○P()\d\s+＋-]+$", norm_t)
            ):
                clean_t = re.sub(r"^[\]\[\)（(「『|｜ー\-_・\s]+", "", norm_t).strip()
                if not (2 <= len(clean_t) <= 12):
                    continue
                tile_p = frame[max(0, by1 - 24) : min(h, by2 + 34), max(0, bx1 - 28) : min(w, bx2 + 28)]
                mean_lum = float(tile_p.mean()) if tile_p.size else 200.0
                has_zero_rem = any(
                    ("あと0" in str(oln.get("text") or "") or "上限" in str(oln.get("text") or ""))
                    and abs(((oln["bbox"][0] + oln["bbox"][2]) // 2) - cx) <= int(w * 0.14)
                    and (by1 - int(h * 0.055)) <= oln["bbox"][1] <= (by2 + int(h * 0.055))
                    for oln in lines
                )
                tile_price: Optional[int] = None
                for oln in lines:
                    ocx = (oln["bbox"][0] + oln["bbox"][2]) // 2
                    if abs(ocx - cx) <= int(w * 0.14) and by2 <= oln["bbox"][1] <= (by2 + int(h * 0.05)):
                        m_tp = re.findall(r"\d{2,3}", str(oln.get("text") or ""))
                        p_cands = [int(n) for n in m_tp if 10 <= int(n) <= 500]
                        if p_cands:
                            tile_price = p_cands[-1]
                            break
                p_avail = int(meta.get("p_points") or 0)
                if tile_price is not None and p_avail > 0 and tile_price > p_avail:
                    continue
                if mean_lum >= 95.0 and not has_zero_rem and not any(t["customize_tile_key"] == clean_t for t in tile_opts):
                    tile_opts.append({
                        "name": clean_t,
                        "title": f"選択メニュー: {clean_t}",
                        "description": f"尝试点击カスタマイズ方案『{clean_t}』",
                        "is_customize_tile": True,
                        "customize_tile_key": clean_t,
                        "bbox": BoundingBox(
                            xmin=max(10, bx1 - 26),
                            ymin=max(10, by1 - 20),
                            xmax=min(w - 10, bx2 + 26),
                            ymax=min(h - 10, by2 + 26),
                        ).model_dump(),
                    })

        if exec_opt is not None:
            options.append(exec_opt)
        options.extend(tile_opts)
        if close_opt is not None:
            options.append(close_opt)
        if back_opt is not None:
            options.append(back_opt)
        if not options:
            options.append({
                "name": "終了",
                "title": "終了 (离开カスタマイズ菜单)",
                "description": "退出当前カスタマイズ菜单",
                "bbox": BoundingBox(
                    xmin=int(w * 0.80),
                    ymin=int(h * 0.83),
                    xmax=int(w * 0.96),
                    ymax=int(h * 0.88),
                ).model_dump(),
            })
        for idx, opt in enumerate(options):
            opt["choice_id"] = idx

        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
        return ScreenObservation(
            scene_type="card_reward",
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=f"{meta['week_label'] or ctx['week_node'].title} | カスタマイズ",
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=meta["p_points"],
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            drinks=self._recognize_bottom_drinks(frame, plan),
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _parse_option_screen(self, scene_type: str, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]) -> ScreenObservation:
        h, w = frame.shape[:2]
        options = self._extract_bottom_and_list_options(lines, w, h, frame=frame)
        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
        return ScreenObservation(
            scene_type=scene_type,
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=f"{meta['week_label'] or ctx['week_node'].title} | 持有P点:{meta['p_points']}",
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            p_points=meta["p_points"],
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )

    def _load_drink_templates(self) -> List[Tuple[int, str, np.ndarray, np.ndarray]]:
        if self._drink_templates_cache is not None:
            return self._drink_templates_cache
        from pathlib import Path
        vecs: List[Tuple[int, str, np.ndarray, np.ndarray]] = []
        seen = set()
        # Inner circular mask focuses on the drink artwork and ignores gold/silver/blue badge rims
        yy, xx = np.ogrid[:24, :24]
        center = 11.5
        inner_2d = ((yy - center) ** 2 + (xx - center) ** 2) <= (24 * 0.38) ** 2
        inner_mask = np.repeat(inner_2d[:, :, None], 3, axis=2).flatten()
        for p_name in ("logic", "sense", "anomaly"):
            for d in self.db.get_drinks_for_plan(p_name):
                did = int(d["id"])
                if did in seen or not d.get("icon_path"):
                    continue
                seen.add(did)
                p = Path(d["icon_path"])
                if not p.exists():
                    continue
                arr_d = np.array(Image.open(p).convert("RGBA"))
                alpha = arr_d[:, :, 3:4].astype(np.float32) / 255.0
                comp = (arr_d[:, :, :3].astype(np.float32) * alpha + 245.0 * (1.0 - alpha)).astype(np.uint8)
                pil = Image.fromarray(comp).resize((24, 24), Image.Resampling.BILINEAR)
                rgb = np.array(pil, dtype=np.float32).flatten()[inner_mask]
                rgb = (rgb - rgb.mean()) / (rgb.std() + 1e-6)
                vecs.append((did, d["name"], inner_mask, rgb))
        self._drink_templates_cache = vecs
        return vecs

    def _recognize_bottom_drinks(self, frame: np.ndarray, plan: str) -> List[RecognizedDrink]:
        h, w = frame.shape[:2]
        if h < 400 or w < 300:
            return []
        templates = self._load_drink_templates()
        if not templates:
            return []
        y1, y2 = int(h * 0.925), min(h, int(h * 0.985))
        ref_mat = np.stack([ref for _, _, _, ref in templates], axis=0)
        mask = templates[0][2]
        rad_live = max(20, int(round(w * 0.0458)))
        rad_compact = max(18, int(round(w * 0.039)))
        radii = (rad_live,) if abs(rad_live - rad_compact) <= 2 else (rad_live, rad_compact)
        cy_candidates = (int(round(h * 0.9506)), int(round(h * 0.9545)), int(round(h * 0.9581)))
        drinks: List[RecognizedDrink] = []
        seen_ids = set()
        for slot_idx, x_frac in enumerate((0.0833, 0.2042, 0.3222, 0.4403)):
            cx_est = int(round(w * x_frac))
            hx1, hx2 = max(0, cx_est - int(w * 0.055)), min(w, cx_est + int(w * 0.055))
            strip = frame[y1:y2, hx1:hx2]
            if strip.size == 0:
                continue
            gold = (strip[:, :, 0] > 210) & (strip[:, :, 1] > 155) & (strip[:, :, 2] < 115)
            bright_box = frame[y1:y2, max(0, cx_est - rad_live) : min(w, cx_est + rad_live)]
            bright_fill = (
                (bright_box[:, :, 0] > 175) & (bright_box[:, :, 1] > 175) & (bright_box[:, :, 2] > 175)
                if bright_box.size
                else np.zeros((1, 1), dtype=bool)
            )
            sat_box = (
                (bright_box.max(axis=2).astype(np.int16) - bright_box.min(axis=2).astype(np.int16)) >= 35
                if bright_box.size
                else np.zeros((1, 1), dtype=bool)
            )
            if int(gold.sum()) < 100 and int(bright_fill.sum()) < 350 and int(sat_box.sum()) < 350:
                continue
            chk = frame[
                max(0, cy_candidates[1] - rad_live) : min(h, cy_candidates[1] + rad_live),
                max(0, cx_est - rad_live) : min(w, cx_est + rad_live),
            ]
            if chk.shape[0] >= 16 and chk.shape[1] >= 16:
                ch_h, ch_w = chk.shape[:2]
                yy_c, xx_c = np.ogrid[:ch_h, :ch_w]
                r_chk = min(ch_h, ch_w) * 0.5
                disc_m = ((yy_c - (ch_h - 1) * 0.5) ** 2 + (xx_c - (ch_w - 1) * 0.5) ** 2) <= (r_chk * 0.78) ** 2
                in_px = chk[disc_m].astype(np.float32)
                if in_px.shape[0] >= 16:
                    in_std = float(in_px.std(axis=0).mean())
                    in_cv = in_std / max(1.0, float(in_px.mean()))
                    if in_std < 12.0 or in_cv < 0.125:
                        continue
            best_sc = -1.0
            best_match: Optional[Tuple[int, str, int, int, int]] = None
            for cy in cy_candidates:
                for cx in (cx_est - 3, cx_est, cx_est + 3):
                    for rad in radii:
                        crop = frame[max(0, cy - rad) : min(h, cy + rad), max(0, cx - rad) : min(w, cx + rad)]
                        if crop.size == 0:
                            continue
                        sm = np.array(
                            Image.fromarray(crop).resize((24, 24), Image.Resampling.BILINEAR),
                            dtype=np.float32,
                        ).flatten()
                        v = sm[mask]
                        v = (v - v.mean()) / (v.std() + 1e-6)
                        scs = (ref_mat @ v) / len(v)
                        idx = int(np.argmax(scs))
                        sc = float(scs[idx])
                        if sc > best_sc:
                            best_sc = sc
                            did, dname, _, _ = templates[idx]
                            best_match = (did, dname, cx, cy, rad)
            if best_match and best_sc >= 0.62:
                did, dname, cx, cy, rad = best_match
                d_row = self.db.get_drink_by_id(did)
                drinks.append(
                    RecognizedDrink(
                        drink_id=did if did not in seen_ids else did,
                        name=dname,
                        playable=True,
                        bbox=BoundingBox(
                            xmin=max(0, cx - rad),
                            ymin=max(0, cy - rad),
                            xmax=min(w, cx + rad),
                            ymax=min(h, cy + rad),
                        ),
                        dsl_summary=(d_row.get("description") if d_row else "") or "",
                    )
                )
                seen_ids.add(did)
        return drinks

    @staticmethod
    def _has_card_customization_badge(card_patch: np.ndarray) -> bool:
        """Detect the bright-green circular customization-count badge (1)/(2)/(3) above the A/M tab of a skill card icon."""
        if card_patch is None or card_patch.size == 0:
            return False
        ph, pw = card_patch.shape[:2]
        if ph < 24 or pw < 24:
            return False
        roi = card_patch[int(ph * 0.64) : int(ph * 0.88), int(pw * 0.45) : int(pw * 0.65)]
        if roi.size == 0:
            return False
        r = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        b = roi[:, :, 2].astype(np.int16)
        gm = (g >= 145) & ((g - r) >= 45) & ((g - b) >= 25) & (r < 115)
        cnt = int(gm.sum())
        area = max(1, roi.shape[0] * roi.shape[1])
        return cnt >= max(24, int(area * 0.045))

    def _detect_hand_card_customization(self, card_patch: np.ndarray) -> Tuple[bool, int]:
        """Detect circular green 专精强化 badge on a hand card and OCR its level (1..3)."""
        if card_patch is None or card_patch.size == 0:
            return False, 0
        ph, pw = card_patch.shape[:2]
        if ph < 24 or pw < 24:
            return False, 0
        roi = card_patch[int(ph * 0.66) : int(ph * 0.84), int(pw * 0.38) : int(pw * 0.62)]
        if roi.size == 0:
            return False, 0
        r = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        b = roi[:, :, 2].astype(np.int16)
        strict_gm = (g >= 115) & ((g - r) >= 65) & ((g - b) >= 55) & (r < 95) & (b < 105)
        cnt = int(strict_gm.sum())
        area = max(1, roi.shape[0] * roi.shape[1])
        if cnt < max(24, int(area * 0.012)):
            return False, 0
        ys, xs = np.where(strict_gm)
        sub = roi[
            max(0, int(ys.min()) + 1) : min(roi.shape[0], int(ys.max()) + 1),
            max(0, int(xs.min()) + 1) : min(roi.shape[1], int(xs.max()) + 1),
        ]
        if sub.size >= 9 and self.en_ocr is not None:
            wm = (sub[:, :, 0] > 185) & (sub[:, :, 1] > 185) & (sub[:, :, 2] > 185)
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
                txt = self.en_ocr.recognize_array(canvas).get("text", "")
                m_c = re.search(r"10\+\s*([1-3])\s*\+10", txt)
                if m_c:
                    return True, int(m_c.group(1))
        return True, 1

    def _detect_hand_card_slots(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, bool]]:
        """Detect card column slots via light card title strip brightness projection + non-overlapping Trouble (眠気) scan."""
        h, w = frame.shape[:2]
        foot = frame[int(h * 0.865) : int(h * 0.905), :]
        col_bright = (np.max(foot, axis=2) > 195).mean(axis=0) > 0.32
        min_slot_w = max(55, int(round(w * 0.135)))
        max_single_w = max(150, int(round(w * 0.275)))
        raw_spans: List[Tuple[int, int]] = []
        in_seg = False
        s = 0
        for x in range(w):
            if col_bright[x] and not in_seg:
                in_seg = True
                s = x
            elif not col_bright[x] and in_seg:
                in_seg = False
                if x - s >= min_slot_w:
                    raw_spans.append((s, x))
        if in_seg and w - s >= min_slot_w:
            raw_spans.append((s, w))

        slots: List[Tuple[int, int, int, int, bool]] = []
        for sx, ex in raw_spans:
            seg_w = ex - sx
            if seg_w > max_single_w:
                # 4- or 5-card hand with touching title strips: split evenly by expected card width (~0.19w)
                n_cards = max(2, min(5, int(round(seg_w / max(1.0, w * 0.19)))))
                step = seg_w / float(n_cards)
                for k in range(n_cards):
                    cx1 = int(round(sx + k * step))
                    cx2 = int(round(sx + (k + 1) * step))
                    if cx2 - cx1 >= max(45, int(round(w * 0.11))):
                        slots.append((cx1, int(h * 0.74), cx2, int(h * 0.91), False))
            else:
                slots.append((sx, int(h * 0.74), ex, int(h * 0.91), False))

        # Separate non-overlapping scan for dark-charcoal Trouble (#23 眠気) cards (silver body + dark neutral footer)
        if h >= 400 and w >= 300:
            dark_strip = frame[int(h * 0.885) : int(h * 0.902), :]
            f_r = dark_strip[:, :, 0].astype(np.int16)
            f_g = dark_strip[:, :, 1].astype(np.int16)
            f_b = dark_strip[:, :, 2].astype(np.int16)
            f_spread = np.maximum(np.maximum(f_r, f_g), f_b) - np.minimum(np.minimum(f_r, f_g), f_b)
            f_mean = (f_r + f_g + f_b) / 3.0
            col_dark = ((f_mean >= 42) & (f_mean <= 112) & (f_spread <= 24)).mean(axis=0) > 0.50
            body = frame[int(h * 0.775) : int(h * 0.865), :]
            b_r = body[:, :, 0].astype(np.int16)
            b_g = body[:, :, 1].astype(np.int16)
            b_b = body[:, :, 2].astype(np.int16)
            b_spread = np.maximum(np.maximum(b_r, b_g), b_b) - np.minimum(np.minimum(b_r, b_g), b_b)
            b_mean = (b_r + b_g + b_b) / 3.0
            col_silver = ((b_mean >= 170) & (b_mean <= 245) & (b_spread <= 45)).mean(axis=0) > 0.35
            col_nemuki = col_dark & col_silver & (~col_bright)
            for bx1, _, bx2, _, _ in slots:
                col_nemuki[max(0, bx1 - 4) : min(w, bx2 + 4)] = False
            for nx1, nx2 in self._contiguous_true_spans(
                np.where(col_nemuki)[0],
                max_gap=max(24, int(round(w * 0.06))),
                min_len=max(55, int(round(w * 0.115))),
            ):
                slots.append((int(nx1), int(h * 0.74), int(nx2), int(h * 0.91), False))
            slots.sort(key=lambda t: t[0])

        enriched: List[Tuple[int, int, int, int, bool]] = []
        for x1, y1, x2, y2, _ in slots:
            rib = frame[int(h * 0.906) : int(h * 0.930), x1:x2]
            rib_orange = int(((rib[:, :, 0] > 225) & (rib[:, :, 1] > 140) & (rib[:, :, 1] < 215) & (rib[:, :, 2] < 95)).sum())
            sel_p = frame[int(h * 0.893) : int(h * 0.928), x1:x2]
            yel_cnt = int(((sel_p[:, :, 0] > 215) & (sel_p[:, :, 1] > 150) & (sel_p[:, :, 2] < 105)).sum())
            sel_band = frame[int(h * 0.902) : int(h * 0.934), x1:x2]
            badge_cnt = int(((sel_band[:, :, 0] > 215) & (sel_band[:, :, 1] > 150) & (sel_band[:, :, 2] < 95)).sum())
            is_selected = rib_orange > 450 or yel_cnt > 700 or badge_cnt >= 220
            enriched.append((x1, y1 - (15 if is_selected else 0), x2, y2 - (15 if is_selected else 0), is_selected))
        return enriched

    def _ocr_bw_banner_number(
        self,
        fg_mask: np.ndarray,
        prefix: str = "HP:",
        max_val: int = 999,
        scale: int = 3,
        min_val: int = 0,
        max_digits: int = 3,
        en_first: bool = False,
    ) -> Optional[int]:
        """OCR a foreground digit mask with a tight ASCII prefix so WinRT never drops 1-digit numbers."""
        ys, xs = np.where(fg_mask)
        if len(xs) < 8:
            return None
        crop = fg_mask[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 2:
            return None
        from PIL import ImageDraw, ImageFont, ImageOps
        tgt_h = max(32, min(40, int(crop.shape[0] * scale)))
        tgt_w = max(8, int(round(crop.shape[1] * tgt_h / max(1, crop.shape[0]))))
        bw = np.where(crop[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
        bw_r = np.array(Image.fromarray(bw).resize((tgt_w, tgt_h), Image.Resampling.BICUBIC))
        try:
            font = ImageFont.truetype("arialbd.ttf", max(18, int(tgt_h * 0.88)))
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", max(18, int(tgt_h * 0.88)))
            except Exception:
                font = ImageFont.load_default(size=max(18, int(tgt_h * 0.88)))
        probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        tb = probe.textbbox((0, 0), prefix, font=font)
        p_w = max(36, (tb[2] - tb[0]) + 14)
        p_img = Image.new("RGB", (p_w, tgt_h), (255, 255, 255))
        ImageDraw.Draw(p_img).text((4, max(0, (tgt_h - (tb[3] - tb[1])) // 2 - tb[1])), prefix, fill=(0, 0, 0), font=font)
        combo = np.concatenate([np.array(p_img), bw_r], axis=1)
        padded = np.array(ImageOps.expand(Image.fromarray(combo), 16, (255, 255, 255)))
        pfx_alpha = re.sub(r"[^A-Za-z0-9]", "", prefix).upper()
        pat = rf"(\d[\d,]{{0,{max(1, int(max_digits))}}})"
        engines = (self.en_ocr, self.ocr) if en_first else (self.ocr, self.en_ocr)
        for engine in engines:
            raw = engine.recognize_array(padded).get("text", "")
            tail = raw
            if ":" in tail:
                tail = tail.split(":", 1)[1]
            elif pfx_alpha and pfx_alpha in tail.upper():
                idx = tail.upper().find(pfx_alpha)
                tail = tail[idx + len(pfx_alpha) :]
            cleaned_tail = _clean_ocr_text(tail)
            if min_val == 0 and cleaned_tail in ("O", "o", "Q"):
                cleaned_tail = "0"
            else:
                cleaned_tail = re.sub(r"(?<=\d)[OoQ]|[OoQ](?=\d)", "0", cleaned_tail)
            m_num = re.search(pat, cleaned_tail)
            if m_num:
                val = int(m_num.group(1).replace(",", ""))
                if min_val <= val <= max_val:
                    return val
        return None

    @staticmethod
    def _extract_enclosed_white_digits(
        patch: np.ndarray,
        white_min: int = 225,
        outline_max: int = 205,
        exclude_blue: bool = False,
    ) -> np.ndarray:
        """Extract bright white digits enclosed by a darker outline (handles stage spotlight bloom)."""
        if patch.size == 0:
            return np.zeros((0, 0), dtype=bool)
        wm = (patch[:, :, 0] >= white_min) & (patch[:, :, 1] >= white_min) & (patch[:, :, 2] >= white_min)
        passable = (patch.mean(axis=2) > outline_max).astype(np.uint8)
        if exclude_blue:
            is_blue = (patch[:, :, 2] > 175) & ((patch[:, :, 2].astype(np.int16) - patch[:, :, 0].astype(np.int16)) >= 45)
            passable[is_blue] = 2
        _, labels = cv2.connectedComponents((passable == 1).astype(np.uint8), connectivity=4)
        border_lbls = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
        if exclude_blue:
            blue_dil = cv2.dilate((passable == 2).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            border_lbls |= set(labels[blue_dil])
        border_lbls.discard(0)
        return wm & (~np.isin(labels, list(border_lbls))) & (labels > 0)

    @staticmethod
    def _deshear_mask(mask: np.ndarray, shear: float = -0.20) -> np.ndarray:
        """Horizontally un-slant italic game UI digits for cleaner WinRT OCR."""
        if mask.size == 0:
            return mask
        mh, mw = mask.shape[:2]
        m_mat = np.float32([[1.0, shear, -shear * mh * 0.5], [0.0, 1.0, 0.0]])
        out = cv2.warpAffine(
            mask.astype(np.uint8) * 255,
            m_mat,
            (mw + int(abs(shear) * mh) + 4, mh),
            flags=cv2.INTER_LINEAR,
        )
        return out > 127

    @staticmethod
    def _extract_outlined_white_core(
        dig: np.ndarray,
        radius: int = 3,
        white_min: int = 205,
        dark_max: int = 135,
    ) -> np.ndarray:
        """Isolate bright white digit cores bordered by a dark outline (rejects sky/glare)."""
        if dig.size == 0:
            return np.zeros((0, 0), dtype=bool)
        dark = (dig[:, :, 0] < dark_max) & (dig[:, :, 1] < dark_max) & (dig[:, :, 2] < dark_max)
        pad = np.pad(dark, radius, mode="constant", constant_values=False)
        near_dark = np.zeros_like(dark)
        span = 2 * radius + 1
        for dy in range(span):
            for dx in range(span):
                near_dark |= pad[dy : dy + dig.shape[0], dx : dx + dig.shape[1]]
        return (dig[:, :, 0] > white_min) & (dig[:, :, 1] > white_min) & (dig[:, :, 2] > white_min) & near_dark

    def _ocr_heart_anchored_hp(
        self,
        frame: np.ndarray,
        roi_fracs: Tuple[float, float, float, float],
        max_hp: int,
    ) -> Optional[int]:
        """Locate the green heart icon and OCR the adjacent dark-outlined white HP digits cleanly."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(w * roi_fracs[0]), int(h * roi_fracs[1]), int(w * roi_fracs[2]), int(h * roi_fracs[3])
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        green_heart = (roi[:, :, 1] > 175) & ((roi[:, :, 1].astype(np.int16) - roi[:, :, 0].astype(np.int16)) > 35) & (roi[:, :, 2] < 145)
        left_lim = max(12, int(roi.shape[1] * 0.55))
        gh_left = green_heart[:, :left_lim]
        ys, xs = np.where(gh_left)
        if len(xs) < 12:
            gh_left = green_heart
            ys, xs = np.where(gh_left)
        if len(xs) < 12:
            return None
        n_g, g_lbl, g_stats, _ = cv2.connectedComponentsWithStats(gh_left.astype(np.uint8), connectivity=8)
        best_gcid = 0
        best_score_g = -1
        for cid in range(1, n_g):
            area_g = int(g_stats[cid, cv2.CC_STAT_AREA])
            if area_g < 12:
                continue
            ch_g = int(g_stats[cid, cv2.CC_STAT_HEIGHT])
            cw_g = int(g_stats[cid, cv2.CC_STAT_WIDTH])
            # Prefer compact square-ish heart blob over tall vertical stage glow
            score_g = area_g + (1000 if (ch_g <= int(h * 0.022) and cw_g >= 8) else 0)
            if score_g > best_score_g:
                best_score_g = score_g
                best_gcid = cid
        if best_gcid > 0:
            ys, xs = np.where(g_lbl == best_gcid)
        hx = int(xs.max())
        hy_c = (int(ys.min()) + int(ys.max())) // 2
        dig = roi[max(0, hy_c - 18) : min(roi.shape[0], hy_c + 18), max(0, hx + 1) : min(roi.shape[1], hx + 68)]
        enc = self._extract_enclosed_white_digits(dig, white_min=225, outline_max=205)
        for cand_m in (self._deshear_mask(enc, shear=-0.22), enc):
            for pfx_enc in ("HP:", "V:", "LV:"):
                val = self._ocr_bw_banner_number(cand_m, prefix=pfx_enc, max_val=max(35, max_hp), scale=3, en_first=True)
                if val is not None:
                    return val
        if dig.size > 0:
            wm_hp = (dig[:, :, 0] > 215) & (dig[:, :, 1] > 215) & (dig[:, :, 2] > 215)
            dk_hp = (dig[:, :, 0] < 120) & (dig[:, :, 1] < 120) & (dig[:, :, 2] < 120)
            dk_dil = cv2.dilate(dk_hp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            cand_hp = wm_hp & dk_dil
            n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(cand_hp.astype(np.uint8), connectivity=8)
            filt_hp = np.zeros_like(cand_hp)
            for cc_i in range(1, n_cc):
                if 10 <= int(cc_stats[cc_i, cv2.CC_STAT_HEIGHT]) <= 32 and int(cc_stats[cc_i, cv2.CC_STAT_AREA]) >= 18:
                    filt_hp[cc_lbl == cc_i] = True
            for pfx_cc, en_first_cc in (("HP:", True), ("体力", False)):
                val_cc = self._ocr_bw_banner_number(filt_hp, prefix=pfx_cc, max_val=max(35, max_hp), scale=3, en_first=en_first_cc)
                if val_cc is not None:
                    return val_cc
        dig_fb = roi[max(0, hy_c - 16) : min(roi.shape[0], hy_c + 18), min(roi.shape[1] - 5, max(0, hx)) : min(roi.shape[1], hx + 62)]
        core = self._extract_outlined_white_core(dig_fb, radius=4, white_min=205, dark_max=125)
        return self._ocr_bw_banner_number(core, prefix="HP:", max_val=max(35, max_hp), scale=3, en_first=True)

    @staticmethod
    def _estimate_exam_stamina_from_bar(frame: np.ndarray, max_hp: int) -> Optional[int]:
        """Estimate exam HUD stamina directly from the top-right stamina bar fill ratio."""
        h, w = frame.shape[:2]
        if h < 400 or w < 300 or max_hp <= 0:
            return None
        bar = frame[int(h * 0.114) : int(h * 0.123), int(w * 0.685) : int(w * 0.916)]
        if bar.size == 0:
            return None
        hsv = cv2.cvtColor(bar, cv2.COLOR_RGB2HSV)
        filled_m = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 140) & (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 95)
        empty_m = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] < 130)
        col_fill = filled_m.mean(axis=0) > 0.30
        col_empty = empty_m.mean(axis=0) > 0.30
        if float((col_fill | col_empty).mean()) < 0.50:
            return None
        filled_idxs = np.where(col_fill)[0]
        if len(filled_idxs) == 0:
            return 0 if float(col_empty.mean()) >= 0.70 else None
        right_fill = float(int(filled_idxs.max()) + 1) / float(bar.shape[1])
        if right_fill >= 0.96:
            return int(max_hp)
        return max(1, min(int(max_hp), int(round(right_fill * max_hp))))

    def _ocr_shield_anchored_genki(
        self,
        frame: np.ndarray,
        roi_fracs: Tuple[float, float, float, float],
    ) -> Optional[int]:
        """Locate the blue Genki shield icon and OCR the adjacent/overlapping white digits."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = int(w * roi_fracs[0]), int(h * roi_fracs[1]), int(w * roi_fracs[2]), int(h * roi_fracs[3])
        if x2 <= x1 or y2 <= y1:
            return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        blue_shield = (
            (roi[:, :, 2] > 165)
            & (roi[:, :, 0] < 135)
            & ((roi[:, :, 2].astype(np.int16) - roi[:, :, 0].astype(np.int16)) > 35)
        )
        ys, xs = np.where(blue_shield)
        if len(xs) < 16:
            return None
        sx_start = max(0, int(np.percentile(xs, 35)) - 2)
        sy_c = (int(ys.min()) + int(ys.max())) // 2
        dig = roi[max(0, sy_c - 18) : min(roi.shape[0], sy_c + 18), sx_start : roi.shape[1]]
        if dig.size == 0:
            return None
        core = self._extract_outlined_white_core(dig, radius=3, white_min=200, dark_max=135)
        val = self._ocr_bw_banner_number(core, prefix="GK:", max_val=999, scale=3)
        if val is not None:
            return val
        bright_white = (dig[:, :, 0] > 200) & (dig[:, :, 1] > 200) & (dig[:, :, 2] > 200)
        return self._ocr_bw_banner_number(bright_white, prefix="GK:", max_val=999, scale=3)

    def _build_recognized_card(
        self,
        matched: Dict[str, Any],
        raw_title: str,
        right_title_patch: np.ndarray,
        bbox: BoundingBox,
        selected: bool,
        playable: bool,
        visual_playable: Optional[bool] = None,
        customized: bool = False,
        customization_count: int = 0,
    ) -> RecognizedCard:
        cyan_plus = bool(
            right_title_patch.size > 0
            and ((right_title_patch[:, :, 2] > 195) & (right_title_patch[:, :, 1] > 135) & (right_title_patch[:, :, 0] < 115)).sum() >= 15
        )
        is_up = bool(matched.get("upgraded", 0) == 1 or "+" in raw_title or "＋" in raw_title or cyan_plus)
        if is_up and matched.get("upgraded", 0) == 0 and matched.get("upgraded_id"):
            up_row = self.db.get_card_by_id(matched["upgraded_id"])
            if up_row:
                matched = up_row
        eff_cust_cnt = int(customization_count or (1 if customized else 0))
        return RecognizedCard(
            card_id=matched["id"],
            name=matched["name"],
            upgraded=is_up,
            playable=playable if visual_playable is None else bool(visual_playable and playable),
            visual_playable=visual_playable,
            selected=selected,
            bbox=bbox,
            dsl_summary=matched.get("description") or "",
            type=matched.get("type") or "active",
            rarity=matched.get("rarity") or "N",
            customized=bool(customized or eff_cust_cnt > 0),
            customization_count=eff_cust_cnt,
        )

    def _parse_right_p_item_counters(self, frame: np.ndarray) -> Dict[int, int]:
        """Read remaining trigger counter ('3回' / '2回' / '1回') under the top-right stage P-item slot 0."""
        h, w = frame.shape[:2]
        if h < 400 or w < 240:
            return {}
        dig_crop = frame[int(h * 0.160) : int(h * 0.188), int(w * 0.820) : int(w * 0.855)]
        if dig_crop.size > 0 and self.en_ocr is not None:
            core = self._extract_outlined_white_core(dig_crop, radius=2, white_min=200, dark_max=155)
            core_px = int((core > 0).sum())
            if 25 <= core_px <= 350:
                ch, cw = core.shape
                n_l, labels, stats_cc, _ = cv2.connectedComponentsWithStats((core > 0).astype(np.uint8))
                clean = np.zeros((ch, cw), dtype=np.uint8)
                for i in range(1, n_l):
                    area = int(stats_cc[i, cv2.CC_STAT_AREA])
                    bw_cc = int(stats_cc[i, cv2.CC_STAT_WIDTH])
                    bh_cc = int(stats_cc[i, cv2.CC_STAT_HEIGHT])
                    if area >= 6 and int(ch * 0.25) <= bh_cc <= int(ch * 0.92) and bw_cc <= int(cw * 0.80):
                        clean[labels == i] = 255
                ys, xs = np.where(clean > 0)
                if len(xs) > 0:
                    sub = clean[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                    up = cv2.resize(sub, (max(1, sub.shape[1] * 4), max(1, sub.shape[0] * 4)), interpolation=cv2.INTER_CUBIC)
                    _, up = cv2.threshold(up, 115, 255, cv2.THRESH_BINARY)
                    inv = 255 - up
                    dh, dw = inv.shape[:2]
                    pad_y = 28
                    fscale = max(0.75, dh / 24.0)
                    (tw_p, _), _ = cv2.getTextSize("10+", cv2.FONT_HERSHEY_SIMPLEX, fscale, 4)
                    (sw_p, _), _ = cv2.getTextSize("+10", cv2.FONT_HERSHEY_SIMPLEX, fscale, 4)
                    x0 = 24
                    dx = x0 + tw_p + 10
                    sx = dx + dw + 10
                    canvas = np.full((dh + pad_y * 2, sx + sw_p + 30, 3), 255, dtype=np.uint8)
                    cv2.putText(canvas, "10+", (x0, pad_y + dh - 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), 4, cv2.LINE_AA)
                    canvas[pad_y : pad_y + dh, dx : dx + dw] = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
                    cv2.putText(canvas, "+10", (sx, pad_y + dh - 2), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0), 4, cv2.LINE_AA)
                    txt = self.en_ocr.recognize_array(canvas).get("text", "")
                    m = re.search(r"10\+\s*([1-9])\s*\+10", txt)
                    if m:
                        return {0: int(m.group(1))}
        if self.ocr is not None:
            jp_txt = self._ocr_crop_upscaled(
                frame,
                (int(w * 0.815), int(h * 0.155), int(w * 0.915), int(h * 0.195)),
                scale=3,
                mode="raw",
                pad=12,
            )
            m_jp = re.search(r"([1-9])\s*[回団囹]", jp_txt)
            if m_jp:
                return {0: int(m_jp.group(1))}
        return {}

    def _parse_left_buff_diamonds(self, frame: np.ndarray, *, plan: str = "logic") -> Dict[str, int]:
        """Detect left-HUD blue buff diamonds for Logic (好印象/やる気) and Sense (好調/集中/絶好調)."""
        self._last_buff_hud_occluded = False
        h, w = frame.shape[:2]
        if h < 400 or w < 300:
            return {}
        is_sense_plan = (plan or "logic").lower() == "sense"

        def _classify_glyph(wm: np.ndarray) -> Optional[str]:
            g_ys, g_xs = np.where(wm)
            if len(g_ys) < 12:
                return None
            gy1, gy2 = int(g_ys.min()), int(g_ys.max())
            gx1, gx2 = int(g_xs.min()), int(g_xs.max())
            gh, gw = gy2 - gy1 + 1, gx2 - gx1 + 1
            if gh < 6 or gw < 6:
                return None
            glyph = wm[gy1 : gy2 + 1, gx1 : gx2 + 1]
            mid_r = max(1, gh // 2)
            top_half = glyph[:mid_r, :]
            bot_half = glyph[mid_r:, :]
            top_area = float(top_half.mean()) if top_half.size else 0.0
            bot_area = float(bot_half.mean()) if bot_half.size else 0.0
            upper_rows = glyph[: max(1, int(round(gh * 0.55))), :]
            max_runs = 0
            multi_run_rows = 0
            for r_mask in upper_rows:
                runs = int(r_mask[0]) + int((~r_mask[:-1] & r_mask[1:]).sum()) if r_mask.size else 0
                if runs > max_runs:
                    max_runs = runs
                if runs >= 3:
                    multi_run_rows += 1
            if max_runs >= 4 and multi_run_rows >= 5 and bot_area >= 0.45 and top_area >= 0.38:
                return "motivation"
            if max_runs <= 2 and multi_run_rows == 0 and bot_area >= 0.65 and top_area >= 0.30:
                return "goodImpressionTurns"
            return None

        def _filter_digit_mask(m: np.ndarray, dp_h: int) -> np.ndarray:
            n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            cl = np.zeros_like(m)
            for cid in range(1, n_cc):
                cy_t = int(cc_stats[cid, cv2.CC_STAT_TOP])
                ch_c = int(cc_stats[cid, cv2.CC_STAT_HEIGHT])
                area_c = int(cc_stats[cid, cv2.CC_STAT_AREA])
                if int(h * 0.008) <= ch_c <= int(h * 0.022) and area_c >= 12 and (cy_t + ch_c) >= int(dp_h * 0.45):
                    cl[cc_lbl == cid] = True
            return cl

        found: Dict[str, int] = {}

        # Primary real-HUD fixed-grid diamond probe (k=0..10): avoids merged clusters & keeps 3-digit leftmost digit
        ja_right_grid = int(w * 0.42)
        dx2_grid = min(w, int(w * 0.24))
        for k in range(11):
            y1 = int(round(h * (0.1469 + k * 0.03825)))
            y2 = min(h, y1 + int(round(h * 0.0332)))
            x1 = int(round(w * 0.0236))
            x2 = min(w, int(round(w * 0.0958)))
            if y2 <= y1 + 8 or x2 <= x1 + 8:
                continue
            patch = frame[y1:y2, x1:x2]
            ph, pw = patch.shape[:2]
            yy, xx = np.ogrid[:ph, :pw]
            cy, cx = (ph - 1) / 2.0, (pw - 1) / 2.0
            norm_dist = np.abs(yy - cy) / (ph / 2.0) + np.abs(xx - cx) / (pw / 2.0)
            in_dia = norm_dist <= 0.88
            p_blue = (
                (patch[:, :, 2] > 180)
                & (patch[:, :, 1] > 110)
                & (patch[:, :, 0] < 135)
                & ((patch[:, :, 2].astype(np.int16) - patch[:, :, 0].astype(np.int16)) >= 55)
            )
            if float((p_blue & in_dia).sum()) / max(1, int(in_dia.sum())) < 0.25:
                continue
            # Exclude outer white diamond rim & bottom-right overlapping stack-digit corner
            in_inner = (norm_dist <= 0.80) & ~((yy > cy + 0.35 * ph / 2.0) & (xx > cx + 0.35 * pw / 2.0))
            wm = in_inner & (patch[:, :, 0] > 205) & (patch[:, :, 1] > 205) & (patch[:, :, 2] > 205)
            buff_key = None if is_sense_plan else _classify_glyph(wm)
            if (not is_sense_plan) and buff_key is None and k > 1:
                continue

            ja_lbl = self._ocr_crop_upscaled(
                frame,
                (
                    (x1 + int(pw * 0.56), y1 + int(ph * 0.35), int(w * 0.32), min(h, y2 + 10))
                    if is_sense_plan
                    else (x2 + 1, max(0, y1 - 6), ja_right_grid, min(h, y2 + 8))
                ),
                scale=(3 if is_sense_plan else 2),
                use_en=False,
                mode="white",
                pad=(12 if is_sense_plan else 10),
            )
            if any(tok in ja_lbl for tok in self._LEFT_BUFF_TOAST_TOKENS):
                self._last_buff_hud_occluded = True
                continue
            if is_sense_plan:
                if any(tok in ja_lbl for tok in ("回", "%", "％")):
                    continue
                has_turn_lbl = any(tok in ja_lbl for tok in ("ターン", "フーン", "タ-ン", "ター"))
                if k == 0 and (has_turn_lbl or not ja_lbl.strip()):
                    buff_key = "goodConditionTurns"
                elif k == 1 and not has_turn_lbl:
                    buff_key = "concentration"
                elif k >= 2 and has_turn_lbl and "perfectConditionTurns" not in found:
                    # 絶好調 (flexing arm + burst lines) always appears above 消費体力減少 (heart)
                    buff_key = "perfectConditionTurns"
                else:
                    continue
            else:
                if buff_key is None or any(tok in ja_lbl for tok in ("ターン", "回", "%", "％")):
                    continue

            dy1, dy2 = (y1 + int(ph * 0.35), min(h, y2 + 10)) if is_sense_plan else (y1 + int(ph * 0.48), min(h, y2 + 10))
            dx1 = x1 + int(pw * (0.56 if is_sense_plan else 0.52))
            dx2_cur = min(w, int(w * 0.19)) if is_sense_plan else dx2_grid
            dp = frame[dy1:dy2, dx1:dx2_cur]
            b_val = None
            if is_sense_plan and dp.size > 0:
                core_s = self._extract_outlined_white_core(dp, radius=3, white_min=200, dark_max=135)
                for pfx in ("LV:", "M:", "W:", "HP:"):
                    b_val = self._ocr_bw_banner_number(core_s, prefix=pfx, max_val=999, scale=3, min_val=1)
                    if b_val is not None:
                        break
                if b_val is None and ja_lbl:
                    m_jt = re.search(r"(\d{1,3})\s*(?:ターン|フーン|ター)", ja_lbl) or re.search(r"(\d{1,3})", ja_lbl)
                    if m_jt:
                        cand_v = int(m_jt.group(1))
                        if 1 <= cand_v <= 999:
                            b_val = cand_v
            if (not is_sense_plan) and dp.size > 0:
                enc_nb = _filter_digit_mask(
                    self._extract_enclosed_white_digits(dp, white_min=220, outline_max=200, exclude_blue=False),
                    dp.shape[0],
                )
                n_ccs_nb = max(0, int(cv2.connectedComponentsWithStats(enc_nb.astype(np.uint8), connectivity=8)[0]) - 1)
                for pfx in ("HP:", "M:", "W:", "LV:"):
                    b_val = self._ocr_bw_banner_number(enc_nb, prefix=pfx, max_val=999, scale=3, min_val=1)
                    if b_val is not None:
                        break
                if b_val is None or len(str(b_val)) < min(3, n_ccs_nb):
                    enc_eb = _filter_digit_mask(
                        self._extract_enclosed_white_digits(dp, white_min=225, outline_max=212, exclude_blue=True),
                        dp.shape[0],
                    )
                    for pfx in ("HP:", "M:", "W:", "LV:"):
                        b_cand = self._ocr_bw_banner_number(enc_eb, prefix=pfx, max_val=999, scale=3, min_val=1)
                        if b_cand is not None:
                            if b_val is None or len(str(b_cand)) > len(str(b_val)):
                                b_val = b_cand
                            break
                if b_val is None:
                    core_d = self._extract_outlined_white_core(dp, radius=3, white_min=200, dark_max=135)
                    for pfx in ("M:", "W:", "LV:", "HP:"):
                        b_val = self._ocr_bw_banner_number(core_d, prefix=pfx, max_val=999, scale=3, min_val=1)
                        if b_val is not None:
                            break
            if b_val is None:
                v_txt = self._ocr_crop_upscaled(
                    frame,
                    (dx1, dy1, dx2_cur, dy2),
                    scale=3,
                    use_en=True,
                    mode="white",
                    pad=12,
                )
                m_v = re.search(r"(\d{1,3})", v_txt)
                if m_v:
                    cand_v = int(m_v.group(1))
                    if 1 <= cand_v <= 999:
                        b_val = cand_v
            if b_val is None and is_sense_plan and buff_key in ("goodConditionTurns", "perfectConditionTurns"):
                # Also allow Japanese OCR on "56ターン" full row crop as a clean fallback
                full_lbl = self._ocr_crop_upscaled(
                    frame,
                    (dx1, max(0, y1 + int(ph * 0.30)), int(w * 0.30), dy2),
                    scale=3,
                    use_en=False,
                    mode="white",
                    pad=12,
                )
                m_t = re.search(r"(\d{1,3})\s*(?:ターン|フーン|ター)", full_lbl) or re.search(r"(\d{1,3})", full_lbl)
                if m_t:
                    cand_v = int(m_t.group(1))
                    if 1 <= cand_v <= 999:
                        b_val = cand_v
            if b_val is not None and 1 <= b_val <= 999:
                found[buff_key] = max(found.get(buff_key, 0), b_val)

        if found or self._last_buff_hud_occluded or is_sense_plan:
            return found

        # Dynamic cluster fallback for non-standard / synthetic layouts
        ry1, ry2 = int(h * 0.105), int(h * 0.56)
        rx1, rx2 = int(w * 0.018), int(w * 0.30)
        roi = frame[ry1:ry2, 0:rx2]
        if roi.size == 0:
            return {}

        icon_band = roi[:, rx1 : int(w * 0.102)]
        cyan_m = (
            (icon_band[:, :, 2] > 210)
            & (icon_band[:, :, 1] > 140)
            & (icon_band[:, :, 0] < 115)
            & ((icon_band[:, :, 2].astype(np.int16) - icon_band[:, :, 0].astype(np.int16)) >= 95)
        )
        if int(cyan_m.sum()) >= 60:
            blue_m = cyan_m
        else:
            blue_m = (
                (icon_band[:, :, 2] > 180)
                & (icon_band[:, :, 0] < 135)
                & ((icon_band[:, :, 2].astype(np.int16) - icon_band[:, :, 0].astype(np.int16)) >= 55)
            )
        row_counts = blue_m.sum(axis=1)
        active_rows = np.where(row_counts >= 4)[0]
        if len(active_rows) == 0:
            return {}

        # Group contiguous y-bands and split vertically touching diamonds (> 0.042h)
        min_dim = max(12, int(h * 0.011))
        max_dim = max(48, int(h * 0.062))
        max_single = max(36, int(h * 0.042))
        raw_clusters = self._contiguous_true_spans(active_rows, max_gap=3, min_len=min_dim)
        clusters: List[Tuple[int, int]] = []
        stack = list(raw_clusters)
        while stack:
            c1, c2 = stack.pop(0)
            if (c2 - c1) > max_single:
                mid_lo = c1 + int((c2 - c1) * 0.30)
                mid_hi = c1 + int((c2 - c1) * 0.70)
                if mid_hi > mid_lo:
                    cut = mid_lo + int(np.argmin(row_counts[mid_lo : mid_hi + 1]))
                    if cut - c1 >= min_dim:
                        stack.insert(0, (cut + 1, c2))
                        stack.insert(0, (c1, cut - 1))
                        continue
            clusters.append((c1, c2))

        for cy1, cy2 in clusters:
            if (cy2 - cy1 + 1) < min_dim or (cy2 - cy1 + 1) > max_dim:
                continue
            sub_blue = blue_m[cy1 : cy2 + 1, :]
            if int(sub_blue.sum()) < 60:
                continue
            xs_active = np.where(sub_blue.sum(axis=0) >= 3)[0]
            if len(xs_active) < 6:
                continue
            cx1, cx2 = int(xs_active.min()), int(xs_active.max())
            if (cx2 - cx1 + 1) < min_dim or (cx2 - cx1 + 1) > max_dim:
                continue

            patch = icon_band[cy1 : cy2 + 1, cx1 : cx2 + 1]
            p_blue = sub_blue[:, cx1 : cx2 + 1]
            # Restrict inner glyph mask strictly between leftmost and rightmost blue border per row
            wm = np.zeros(p_blue.shape, dtype=bool)
            for rr in range(p_blue.shape[0]):
                b_cols = np.where(p_blue[rr])[0]
                if len(b_cols) >= 2:
                    lx, rx = int(b_cols[0]) + 1, int(b_cols[-1])
                    if rx > lx:
                        row_px = patch[rr, lx:rx]
                        wm[rr, lx:rx] = (row_px[:, 0] > 205) & (row_px[:, 1] > 205) & (row_px[:, 2] > 205)
            buff_key = _classify_glyph(wm)
            if buff_key is None:
                continue

            val_y1 = max(0, cy1 - 6)
            val_y2 = min(roi.shape[0], cy2 + 8)
            val_x1 = min(roi.shape[1] - 1, rx1 + cx2 + 1)
            ja_right = min(w, max(rx2, int(w * 0.42)))
            ja_lbl = self._ocr_crop_upscaled(
                frame,
                (val_x1, ry1 + val_y1, ja_right, ry1 + val_y2),
                scale=2,
                use_en=False,
                mode="white",
                pad=10,
            )
            if any(tok in ja_lbl for tok in self._LEFT_BUFF_TOAST_TOKENS):
                self._last_buff_hud_occluded = True
                continue
            if any(tok in ja_lbl for tok in ("ターン", "回", "%", "％")):
                continue

            dp = roi[
                max(0, cy1 + int((cy2 - cy1) * 0.25)) : min(roi.shape[0], cy2 + 10),
                max(0, rx1 + int(cx1 + (cx2 - cx1) * 0.45)) : min(roi.shape[1], int(w * 0.22)),
            ]
            b_val = None
            if dp.size > 0:
                enc_d = self._extract_enclosed_white_digits(dp, white_min=225, outline_max=212, exclude_blue=True)
                for pfx in ("HP:", "M:", "W:", "LV:"):
                    b_val = self._ocr_bw_banner_number(enc_d, prefix=pfx, max_val=999, scale=3, min_val=1)
                    if b_val is not None:
                        break
            if b_val is None:
                val_patch = roi[val_y1:val_y2, min(roi.shape[1] - 1, rx1 + cx2 + 2) : rx2]
                if val_patch.size > 0:
                    core = self._extract_outlined_white_core(val_patch, radius=3, white_min=200, dark_max=135)
                    for pfx in ("M:", "W:", "LV:", "HP:"):
                        b_val = self._ocr_bw_banner_number(core, prefix=pfx, max_val=999, scale=3, min_val=1)
                        if b_val is not None:
                            break
            if b_val is None:
                v_txt = self._ocr_crop_upscaled(
                    frame,
                    (min(w - 1, rx1 + cx2 + 2), ry1 + val_y1, rx2, ry1 + val_y2),
                    scale=3,
                    use_en=True,
                    mode="white",
                    pad=12,
                )
                m_v = re.search(r"(\d{1,3})", v_txt)
                if m_v:
                    cand_v = int(m_v.group(1))
                    if 1 <= cand_v <= 999:
                        b_val = cand_v
            if b_val is not None and 1 <= b_val <= 999:
                found[buff_key] = max(found.get(buff_key, 0), b_val)
        return found

    def _parse_battle_screen(
        self,
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
        plan: str,
        meta: Dict[str, Any],
        turn_box_txt: str = "",
    ) -> ScreenObservation:
        h, w = frame.shape[:2]
        full_text = "".join(ln["text"] for ln in lines)

        was_in_battle = self._in_battle_scene
        self._in_battle_scene = True
        self._non_battle_streak = 0
        turns_remaining = self._last_turns_remaining
        phase_info = ""
        top_hdr_text = "".join(ln["text"] for ln in lines if ln["bbox"][1] < int(h * 0.08))
        is_exam_hud = ("数" not in turn_box_txt and "残りターン数" not in top_hdr_text) and (
            "残りタ" in turn_box_txt
            or bool(re.search(r"残り.{0,3}タ", turn_box_txt))
            or "残りタ" in top_hdr_text
            or any(k in f"{turn_box_txt} {top_hdr_text}" for k in ("1st", "2nd", "3rd", "4th", "5th", "6th", "ボーカル", "ポーカル", "ホーカル", "オーカル", "ダンス", "タンス", "ビジュアル", "ビシュアル", "ジュアル"))
        )
        turn_digit_txt = self._ocr_crop_upscaled(
            frame, (int(w * 0.045), int(h * 0.026), int(w * 0.21), int(h * 0.068)), scale=2, use_en=True, mode="white", pad=15
        )
        m_turn_en = re.search(r"(\d{1,2})", turn_digit_txt)
        m_turn = re.search(r"残りターン数?\s*(\d{1,2})", f"{turn_box_txt} {top_hdr_text}")
        raw_turn_cand: Optional[int] = None
        if is_exam_hud:
            if self.scenario_id in ("hif_honsen", "hif"):
                hif_hdr = f"{turn_box_txt} {top_hdr_text} {meta.get('week_label', '')}"
                if "FINAL" in hif_hdr.upper() or "ラウンド2" in hif_hdr or meta["week"] >= 8 or self._last_week >= 8:
                    meta["week"] = 9
                    meta["week_label"] = "H.I.F本戦ラウンド2"
                    self._last_week = 9
                else:
                    meta["week"] = 7
                    meta["week_label"] = "H.I.F本戦ラウンド1"
                    self._last_week = 7
            elif meta["week"] < 6:
                meta["week"] = 6
                self._last_week = 6
            elif meta["week"] >= 11 or self._last_week >= 11 or "最終" in turn_box_txt:
                meta["week"] = 13
                meta["week_label"] = "最終試験"
                self._last_week = 13
            # Try tight circular turn-ring OCR first so adjacent "ボーカル 4838%" never leaks "4"
            from PIL import ImageDraw, ImageFont, ImageOps
            tc_wide = frame[int(h * 0.006) : int(h * 0.085), int(w * 0.011) : int(w * 0.315)]
            tc_found = False
            if h >= 400 and w >= 300:
                # Full-stem digit box (0.041..0.076h) avoids clipping the bottom leg of italic '4' / '7' / '9'
                sub_dig = frame[int(h * 0.041) : int(h * 0.076), int(w * 0.044) : int(w * 0.120)]
                if sub_dig.size > 0:
                    try:
                        jfont_ring = ImageFont.truetype(r"C:\Windows\Fonts\msgothic.ttc", 42)
                    except Exception:
                        jfont_ring = ImageFont.load_default(size=36)
                    cand_ring_masks: List[np.ndarray] = []
                    m_hi = (sub_dig[:, :, 0] > 210) & (sub_dig[:, :, 1] > 210) & (sub_dig[:, :, 2] > 210)
                    if int(m_hi.sum()) >= 22:
                        cand_ring_masks.append(m_hi)
                    else:
                        # Dimmed-HUD fallback (e.g. tooltip/effect overlay): require dark outline so background glow is ignored
                        m_dim_raw = self._extract_outlined_white_core(sub_dig, radius=3, white_min=135, dark_max=95)
                        if int(m_dim_raw.sum()) >= 20:
                            n_cc_r, lbl_r, st_r, _ = cv2.connectedComponentsWithStats(m_dim_raw.astype(np.uint8), connectivity=8)
                            m_dim = np.zeros_like(m_dim_raw)
                            for cid_r in range(1, n_cc_r):
                                ar_r = int(st_r[cid_r, cv2.CC_STAT_AREA])
                                ch_r = int(st_r[cid_r, cv2.CC_STAT_HEIGHT])
                                cw_r = int(st_r[cid_r, cv2.CC_STAT_WIDTH])
                                if ar_r >= 12 and ch_r >= int(sub_dig.shape[0] * 0.38) and cw_r <= int(sub_dig.shape[1] * 0.88):
                                    m_dim[lbl_r == cid_r] = True
                            if int(m_dim.sum()) >= 20:
                                cand_ring_masks.append(m_dim)
                    for base_rm in cand_ring_masks:
                        if tc_found:
                            break
                        for rm_var in (self._deshear_mask(base_rm, shear=-0.20), base_rm):
                            for pfx_r in ("HP:", "LV:"):
                                dv_en = self._ocr_bw_banner_number(
                                    rm_var,
                                    prefix=pfx_r,
                                    max_val=15,
                                    scale=3,
                                    min_val=1,
                                    max_digits=2,
                                    en_first=True,
                                )
                                if dv_en is not None and 1 <= dv_en <= 15:
                                    raw_turn_cand = dv_en
                                    tc_found = True
                                    break
                            if tc_found:
                                break
                            ys_r, xs_r = np.where(rm_var)
                            if len(xs_r) >= 12:
                                crop_r = rm_var[int(ys_r.min()) : int(ys_r.max()) + 1, int(xs_r.min()) : int(xs_r.max()) + 1]
                                dh_r, dw_r = crop_r.shape
                                canvas_r = Image.new("RGB", (dw_r * 2 + 260, max(80, dh_r * 2 + 30)), (255, 255, 255))
                                draw_r = ImageDraw.Draw(canvas_r)
                                draw_r.text((15, (canvas_r.height - 42) // 2), "残り", fill=(0, 0, 0), font=jfont_ring)
                                dig_arr_r = np.where(crop_r[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                                dig_pil_r = Image.fromarray(dig_arr_r).resize(
                                    (max(4, dw_r * 2), max(8, dh_r * 2)), Image.Resampling.BICUBIC
                                )
                                canvas_r.paste(dig_pil_r, (105, (canvas_r.height - dh_r * 2) // 2))
                                draw_r.text((115 + dw_r * 2, (canvas_r.height - 42) // 2), "ターン", fill=(0, 0, 0), font=jfont_ring)
                                ring_txt_r = _clean_ocr_text(self.ocr.recognize_array(np.array(canvas_r)).get("text", ""))
                                m_ring_r = re.search(r"(\d{1,2})", ring_txt_r)
                                if m_ring_r:
                                    dv_r = int(m_ring_r.group(1))
                                    if 1 <= dv_r <= 15:
                                        raw_turn_cand = dv_r
                                        tc_found = True
                                        break
                if not tc_found:
                    sub = frame[int(h * 0.033) : int(h * 0.068), int(w * 0.048) : int(w * 0.118)]
                    if sub.size > 0:
                        ring_mw = (sub[:, :, 0] > 215) & (sub[:, :, 1] > 215) & (sub[:, :, 2] > 215)
                        if int(ring_mw.sum()) >= 25:
                            dh, dw = ring_mw.shape
                            canvas = Image.new("RGB", (dw * 2 + 260, max(80, dh * 2 + 30)), (255, 255, 255))
                            draw = ImageDraw.Draw(canvas)
                            try:
                                jfont = ImageFont.truetype(r"C:\Windows\Fonts\msgothic.ttc", 42)
                            except Exception:
                                jfont = ImageFont.load_default(size=36)
                            draw.text((15, (canvas.height - 42) // 2), "残り", fill=(0, 0, 0), font=jfont)
                            dig_arr = np.where(ring_mw[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                            dig_pil = Image.fromarray(dig_arr).resize((dw * 2, dh * 2), Image.Resampling.NEAREST)
                            canvas.paste(dig_pil, (105, (canvas.height - dh * 2) // 2))
                            draw.text((115 + dw * 2, (canvas.height - 42) // 2), "ターン", fill=(0, 0, 0), font=jfont)
                            ring_txt = _clean_ocr_text(self.ocr.recognize_array(np.array(canvas)).get("text", ""))
                            m_ring_d = re.search(r"(\d{1,2})", ring_txt)
                            if m_ring_d:
                                dv = int(m_ring_d.group(1))
                                if 1 <= dv <= 15:
                                    raw_turn_cand = dv
                                    tc_found = True
            if not tc_found and tc_wide.size > 0:
                mw = (tc_wide[:, :, 0] > 195) & (tc_wide[:, :, 1] > 195) & (tc_wide[:, :, 2] > 195)
                mw[int(tc_wide.shape[0] * 0.38) :, int(tc_wide.shape[1] * 0.43) :] = False
                bw = np.where(mw[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                bw2 = np.array(ImageOps.expand(Image.fromarray(bw).resize((bw.shape[1] * 2, bw.shape[0] * 2), Image.Resampling.BICUBIC), 15, (255, 255, 255)))
                for tln in self.ocr.recognize_array(bw2).get("lines", []):
                    if tln["bbox"][1] >= int(tc_wide.shape[0] * 0.72):
                        m_d = re.match(r"^\s*(\d{1,2})", tln["text"].strip())
                        if m_d:
                            dv = int(m_d.group(1))
                            if 1 <= dv <= 15:
                                raw_turn_cand = dv
                                tc_found = True
                                break
            if not tc_found and tc_wide.size > 0:
                tc_pad = np.array(ImageOps.expand(Image.fromarray(tc_wide), 10, (255, 255, 255)))
                for tln in self.ocr.recognize_array(tc_pad).get("lines", []):
                    tt = tln["text"].strip()
                    if tln["bbox"][0] < int(w * 0.115) and tln["bbox"][1] >= int(h * 0.018):
                        m_d = re.match(r"^\s*(\d{1,2})", tt)
                        if m_d:
                            dv = int(m_d.group(1))
                            if 1 <= dv <= 15:
                                raw_turn_cand = dv
                                break
            # Right attribute label (105..228, 45..135) + crisp English 4-digit multiplier fallback
            phase_raw = self._ocr_crop_upscaled(
                frame, (int(w * 0.145), int(h * 0.028), int(w * 0.315), int(h * 0.085)), scale=2, mode="white", pad=12
            )
            phase_en = self._ocr_crop_upscaled(
                frame, (int(w * 0.145), int(h * 0.050), int(w * 0.315), int(h * 0.085)), scale=3, use_en=True, mode="white", pad=12
            )
            m_en = re.search(r"(\d{4})", phase_en)
            attr_lbl: Optional[str] = None
            if any(k in phase_raw for k in ("ボーカル", "ポーカル", "ホーカル", "オーカル")):
                attr_lbl = "ボーカル"
            elif any(k in phase_raw for k in ("ダンス", "タンス")):
                attr_lbl = "ダンス"
            elif any(k in phase_raw for k in ("ビジュアル", "ビシュアル", "ジュアル")):
                attr_lbl = "ビジュアル"
            if attr_lbl is None:
                attr_p = frame[int(h * 0.032) : int(h * 0.080), int(w * 0.145) : int(w * 0.305)]
                if attr_p.size > 0:
                    pink_px = int(((attr_p[:, :, 0] > 200) & (attr_p[:, :, 2] > 120) & (attr_p[:, :, 1] < 150)).sum())
                    blue_px = int(((attr_p[:, :, 2] > 190) & (attr_p[:, :, 0] < 140)).sum())
                    yel_px = int(((attr_p[:, :, 0] > 200) & (attr_p[:, :, 1] > 160) & (attr_p[:, :, 2] < 130)).sum())
                    best_c = max((pink_px, "ボーカル"), (blue_px, "ダンス"), (yel_px, "ビジュアル"))
                    if best_c[0] >= 12:
                        attr_lbl = best_c[1]
            if attr_lbl and m_en and 1000 <= int(m_en.group(1)) <= 8000:
                phase_info = f"{attr_lbl}{m_en.group(1)}%"
            else:
                phase_info = phase_raw
        else:
            turn_cand: Optional[int] = None
            for ty1_f, ty2_f in ((0.034, 0.070), (0.026, 0.056)):
                t_patch = frame[int(h * ty1_f) : int(h * ty2_f), int(w * 0.05) : int(w * 0.185)]
                if t_patch.size > 0:
                    t_mask = (t_patch[:, :, 0] > 215) & (t_patch[:, :, 1] > 215) & (t_patch[:, :, 2] > 215)
                    turn_cand = self._ocr_bw_banner_number(t_mask, prefix="T:", max_val=15, scale=3, min_val=1)
                    if turn_cand is not None:
                        break
            if turn_cand is None:
                for m_cand in (m_turn_en, m_turn):
                    if m_cand:
                        val = int(m_cand.group(1))
                        if 1 <= val <= 15:
                            turn_cand = val
                            break
            if turn_cand is None:
                # Contrast-normalized fallback handles darkened overlays when holding/inspecting unplayable cards
                from PIL import ImageOps
                td_box = frame[int(h * 0.026) : int(h * 0.070), int(w * 0.04) : int(w * 0.19)]
                if td_box.size > 0 and td_box.max() > 0:
                    td_norm = np.clip(td_box.astype(np.float32) * (255.0 / max(1, int(td_box.max()))), 0, 255).astype(np.uint8)
                    td2 = Image.fromarray(td_norm).resize((td_box.shape[1] * 2, td_box.shape[0] * 2), Image.Resampling.BICUBIC)
                    td_txt = _clean_ocr_text(self.ocr.recognize_array(np.array(ImageOps.expand(td2, 15, (255, 255, 255)))).get("text", ""))
                    m_norm = re.search(r"(\d{1,2})", td_txt)
                    if m_norm:
                        val = int(m_norm.group(1))
                        if 1 <= val <= 15:
                            turn_cand = val
            raw_turn_cand = turn_cand

        if not was_in_battle:
            self._turn_drop_candidate = None
        if raw_turn_cand is not None and 1 <= raw_turn_cand <= 15:
            if was_in_battle and self._last_turns_remaining > 0:
                # Within a single continuous battle, remaining turns normally drop by <=2 per step;
                # if 2+ turns were missed earlier, recover once 2 consecutive battle frames confirm the lower turn.
                diff_turns = self._last_turns_remaining - raw_turn_cand
                if raw_turn_cand <= self._last_turns_remaining and diff_turns <= 2:
                    turns_remaining = raw_turn_cand
                    self._turn_drop_candidate = None
                elif raw_turn_cand < self._last_turns_remaining and diff_turns > 2:
                    prev_drop = self._turn_drop_candidate
                    if prev_drop is not None and raw_turn_cand <= prev_drop[0] and (prev_drop[0] - raw_turn_cand) <= 1:
                        drop_streak = prev_drop[1] + 1
                    else:
                        drop_streak = 1
                    if drop_streak >= 2:
                        turns_remaining = raw_turn_cand
                        self._turn_drop_candidate = None
                    else:
                        self._turn_drop_candidate = (raw_turn_cand, drop_streak)
            else:
                turns_remaining = raw_turn_cand
                self._turn_drop_candidate = None
        self._last_turns_remaining = turns_remaining
        if is_exam_hud and self.scenario_id in ("hif_honsen", "hif") and turns_remaining >= 11:
            meta["week"] = 9
            meta["week_label"] = "H.I.F本戦ラウンド2"
            self._last_week = 9

        # Top-center CLEAR/PERFECT score ring + exam score
        ring_hdr = self._ocr_crop_upscaled(frame, (int(w * 0.37), int(h * 0.015), int(w * 0.63), int(h * 0.075)), scale=1, mode="raw", pad=15)
        score = 0
        target_score = 0
        ring_num_txt = ""
        if is_exam_hud:
            # Player's active scoreboard card renders a taller score font (~17-27px vs rival ~12px at 1600p)
            from PIL import ImageOps
            sb = frame[int(h * 0.080) : int(h * 0.106), int(w * 0.29) : w]
            if sb.size > 0 and h >= 400 and w >= 300:
                bot_tol = max(5, int(h * 0.004))

                def _read_sb_mask(m_sb: np.ndarray) -> Optional[Tuple[int, int]]:
                    n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(m_sb.astype(np.uint8), connectivity=8)
                    clean_sb = np.zeros_like(m_sb)
                    valid_ccs: List[Tuple[int, int, int, int]] = []
                    for cid in range(1, n_cc):
                        x = int(cc_stats[cid, cv2.CC_STAT_LEFT])
                        y = int(cc_stats[cid, cv2.CC_STAT_TOP])
                        cw = int(cc_stats[cid, cv2.CC_STAT_WIDTH])
                        ch = int(cc_stats[cid, cv2.CC_STAT_HEIGHT])
                        area = int(cc_stats[cid, cv2.CC_STAT_AREA])
                        if int(h * 0.010) <= ch <= int(h * 0.028) and cw <= int(w * 0.055) and area >= 18:
                            valid_ccs.append((cid, x, y, ch))
                    if valid_ccs:
                        tall_bots = [y + ch for (_, _, y, ch) in valid_ccs if ch >= int(h * 0.012)]
                        max_bot = max(tall_bots) if tall_bots else max(y + ch for (_, _, y, ch) in valid_ccs)
                        for cid, _x, y, ch in valid_ccs:
                            if abs((y + ch) - max_bot) <= bot_tol:
                                clean_sb[cc_lbl == cid] = True
                    if not clean_sb.any():
                        for cid in range(1, n_cc):
                            ch = int(cc_stats[cid, cv2.CC_STAT_HEIGHT])
                            cw = int(cc_stats[cid, cv2.CC_STAT_WIDTH])
                            area = int(cc_stats[cid, cv2.CC_STAT_AREA])
                            cy_mid = int(cc_stats[cid, cv2.CC_STAT_TOP]) + ch * 0.5
                            if int(h * 0.007) <= ch <= int(h * 0.028) and cw <= int(w * 0.055) and area >= 8 and cy_mid >= sb.shape[0] * 0.35:
                                clean_sb[cc_lbl == cid] = True
                    sb_spans = self._contiguous_true_spans(
                        np.where(clean_sb.sum(axis=0) >= 2)[0],
                        max_gap=max(6, int(w * 0.014)),
                        min_len=6,
                    )
                    scored_spans: List[Tuple[int, int, int, int]] = []
                    for s1, s2 in sb_spans:
                        ys_s, _ = np.where(clean_sb[:, s1 : s2 + 1])
                        if len(ys_s) > 0:
                            scored_spans.append((int(np.ptp(ys_s)) + 1, s2 - s1, s1, s2))
                    scored_spans.sort(reverse=True)
                    for _, sw, s1, s2 in scored_spans:
                        span_mask = clean_sb[:, s1 : s2 + 1]
                        n_d, _, st_d, _ = cv2.connectedComponentsWithStats(span_mask.astype(np.uint8), connectivity=8)
                        cc_cnt = sum(
                            1 for cid_d in range(1, n_d)
                            if int(st_d[cid_d, cv2.CC_STAT_HEIGHT]) >= int(h * 0.009)
                        )
                        cand_sc = self._ocr_bw_banner_number(
                            span_mask,
                            prefix="HP:",
                            max_val=99999999,
                            scale=3,
                            min_val=0,
                            max_digits=8,
                            en_first=True,
                        )
                        if cand_sc is not None and len(str(cand_sc)) <= cc_cnt + 1:
                            return (cand_sc, sw)
                    return None

                enc_lo_res = _read_sb_mask(self._extract_enclosed_white_digits(sb, white_min=200, outline_max=170))
                enc_res = _read_sb_mask(self._extract_enclosed_white_digits(sb, white_min=222, outline_max=195))
                core_res = _read_sb_mask(self._extract_outlined_white_core(sb, radius=3, white_min=205, dark_max=135))
                best_sc, best_sw = 0, 0
                if enc_lo_res is not None:
                    best_sc, best_sw = enc_lo_res
                if enc_res is not None and (
                    best_sc == 0 or (len(str(enc_res[0])) > len(str(best_sc)) and enc_res[1] >= best_sw + 10)
                ):
                    best_sc, best_sw = enc_res
                if core_res is not None and (
                    best_sc == 0 or (len(str(core_res[0])) > len(str(best_sc)) and core_res[1] >= best_sw + 10)
                ):
                    best_sc, best_sw = core_res
                if best_sc > score:
                    score = best_sc
            if score == 0:
                sb_band = frame[int(h * 0.004) : int(h * 0.112), int(w * 0.30) : w]
                if sb_band.size > 0:
                    sb_2x = Image.fromarray(sb_band).resize((sb_band.shape[1] * 2, sb_band.shape[0] * 2), Image.Resampling.BICUBIC)
                    sb_res = self.ocr.recognize_array(np.array(ImageOps.expand(sb_2x, 12, (255, 255, 255))))
                    sb_cands: List[Tuple[int, str]] = []
                    for sln in sb_res.get("lines", []):
                        for it in (sln.get("words") or [sln]):
                            bx1, by1, bx2, by2 = it["bbox"]
                            if by1 >= int(h * 0.150) and re.search(r"\d", it["text"]):
                                sb_cands.append((by2 - by1, it["text"]))
                    sb_cands.sort(reverse=True)
                    if sb_cands and sb_cands[0][0] >= int(h * 0.020):
                        m_ex_sc = re.search(r"(\d[\d,]{0,8})", sb_cands[0][1])
                        if m_ex_sc:
                            score = int(m_ex_sc.group(1).replace(",", ""))
            if score == 0:
                sc_exam = self._ocr_crop_upscaled(
                    frame, (int(w * 0.29), int(h * 0.072), int(w * 0.52), int(h * 0.106)), scale=2, use_en=True, mode="white", pad=10
                )
                m_ex_sc = re.search(r"(\d[\d,]{0,8})", sc_exam)
                if m_ex_sc:
                    score = int(m_ex_sc.group(1).replace(",", ""))
            target_score = 0 if self.scenario_id in ("hif_honsen", "hif") else (900 if meta["week"] <= 6 else 4000)
            if not ring_hdr:
                ring_hdr = f"試験 {phase_info}".strip()
        else:
            # Colored score number inside top center ring (try colored mask, fall back to raw crop for light digits like pink 11)
            score_patch = frame[int(h * 0.060) : int(h * 0.106), int(w * 0.42) : int(w * 0.58)]
            col_mask = (np.min(score_patch, axis=2) < 200) & (np.max(score_patch, axis=2) > 135)
            bw_sc = np.where(col_mask[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
            from PIL import ImageOps
            sc_pil = Image.fromarray(bw_sc).resize((bw_sc.shape[1] * 2, bw_sc.shape[0] * 2), Image.Resampling.BICUBIC)
            ring_num_txt = _clean_ocr_text(self.en_ocr.recognize_array(np.array(ImageOps.expand(sc_pil, 20, (255, 255, 255)))).get("text", ""))
            if not re.search(r"\d", ring_num_txt):
                ring_num_txt = self._ocr_crop_upscaled(
                    frame, (int(w * 0.42), int(h * 0.060), int(w * 0.58), int(h * 0.106)), scale=2, use_en=True, mode="raw", pad=20
                )

            m_score = re.search(r"(\d{2,6})/(\d{3,6})", f"{ring_num_txt} {full_text}")
            if m_score:
                score, target_score = int(m_score.group(1)), int(m_score.group(2))
            elif re.search(r"(\d{1,6})", ring_num_txt):
                target_score = int(re.search(r"(\d{1,6})", ring_num_txt).group(1))
            if not ring_hdr and target_score > 0:
                ring_hdr = f"CLEARまで {target_score}"

        # Top-right battle HUD: white/cyan threshold captures primary stat, Genki shield, and Stamina cleanly
        from PIL import ImageOps
        hud_y1 = int(h * 0.078) if is_exam_hud else int(h * 0.009)
        hud_y2 = int(h * 0.153) if is_exam_hud else int(h * 0.132)
        tr_patch = frame[hud_y1:hud_y2, int(w * 0.69) : w]
        tr_mw = (tr_patch[:, :, 0] > 195) & (tr_patch[:, :, 1] > 195) & (tr_patch[:, :, 2] > 195)
        tr_bw = np.where(tr_mw[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
        tr_2x = Image.fromarray(tr_bw).resize((tr_bw.shape[1] * 2, tr_bw.shape[0] * 2), Image.Resampling.BICUBIC)
        tr_res = self.en_ocr.recognize_array(np.array(ImageOps.expand(tr_2x, 15, (255, 255, 255))))
        stamina = meta["stamina"]
        genki = 0
        best_hp_y = -1
        if is_exam_hud:
            # Anchor on blue Genki shield first (reliable for 1-, 2-, 3-digit shields), then crop fallback
            shield_gk = self._ocr_shield_anchored_genki(frame, (0.855, 0.102, 0.988, 0.135))
            if shield_gk is not None:
                genki = shield_gk
            if genki == 0:
                gk = frame[int(h * 0.104) : int(h * 0.132), int(w * 0.88) : int(w * 0.98)]
                if gk.size > 0:
                    gk_mw = (gk[:, :, 0] > 205) & (gk[:, :, 1] > 205) & (gk[:, :, 2] > 205)
                    gk_bw = np.where(gk_mw[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                    gk_3x = Image.fromarray(gk_bw).resize((gk_bw.shape[1] * 3, gk_bw.shape[0] * 3), Image.Resampling.BICUBIC)
                    gk_txt = self.en_ocr.recognize_array(np.array(ImageOps.expand(gk_3x, 18, (255, 255, 255)))).get("text", "")
                    m_gk = re.search(r"(\d{1,3})", gk_txt)
                    if m_gk:
                        genki = int(m_gk.group(1))
                    elif int(((gk[:, :, 2] > 165) & (gk[:, :, 0] < 135)).sum()) >= 20:
                        banner_gk = self._ocr_bw_banner_number(gk_mw, prefix="GK:", max_val=999, scale=3)
                        if banner_gk is not None:
                            genki = banner_gk
            hp_found = False
            heart_hp = self._ocr_heart_anchored_hp(frame, (0.73, 0.130, 0.90, 0.165), meta["max_stamina"])
            if heart_hp is not None:
                stamina = heart_hp
                hp_found = True
            if not hp_found:
                hp_box = frame[int(h * 0.132) : int(h * 0.163), int(w * 0.73) : int(w * 0.90)]
                if hp_box.size > 0:
                    hp_mw = (hp_box[:, :, 0] > 185) & (hp_box[:, :, 1] > 185) & (hp_box[:, :, 2] > 185)
                    hp_bw = np.where(hp_mw[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                    for probe_arr in (hp_bw, hp_box):
                        p3x = Image.fromarray(probe_arr).resize((probe_arr.shape[1] * 3, probe_arr.shape[0] * 3), Image.Resampling.BICUBIC)
                        hp_txt = self.ocr.recognize_array(np.array(ImageOps.expand(p3x, 18, (255, 255, 255)))).get("text", "")
                        m_hp = re.search(r"(\d{1,2})", hp_txt)
                        if m_hp and 0 <= int(m_hp.group(1)) <= max(35, meta["max_stamina"]):
                            stamina = int(m_hp.group(1))
                            hp_found = True
                            break
            if not hp_found:
                for ln in lines:
                    bx1, by1, bx2, by2 = ln["bbox"]
                    if int(w * 0.73) <= bx1 <= int(w * 0.90) and int(h * 0.130) <= by1 <= int(h * 0.164):
                        m_hp = re.search(r"(\d{1,2})", ln["text"])
                        if m_hp and 0 <= int(m_hp.group(1)) <= max(35, meta["max_stamina"]):
                            stamina = int(m_hp.group(1))
                            hp_found = True
                            break
            bar_hp = self._estimate_exam_stamina_from_bar(frame, meta["max_stamina"])
            if (not hp_found) and bar_hp is not None:
                stamina = bar_hp
                hp_found = True
            elif hp_found and bar_hp is not None and abs(stamina - bar_hp) >= max(4, int(round(meta["max_stamina"] * 0.14))):
                stamina = bar_hp
        if not is_exam_hud:
            for rln in tr_res.get("lines", []):
                ry = rln["bbox"][1]
                rx = rln["bbox"][0]
                rtxt = rln["text"].strip()
                m_num = re.search(r"(\d{1,4})", rtxt)
                if not m_num:
                    continue
                val = int(m_num.group(1))
                if ry < 185 and val >= 20:
                    dom_k = max(meta["stats"], key=lambda k: meta["stats"][k])
                    meta["stats"][dom_k] = val
                    self._last_stats[dom_k] = val
                elif 195 <= ry <= 325 and rx >= 260 and val <= 999:
                    genki = val
                elif ry >= 315 and rx < 265 and "+" not in rtxt and val <= 99:
                    if ry > best_hp_y:
                        best_hp_y = ry
                        stamina = val
            if best_hp_y < 0:
                reg_heart_hp = self._ocr_heart_anchored_hp(frame, (0.72, 0.080, 0.90, 0.118), meta["max_stamina"])
                if reg_heart_hp is not None:
                    stamina = reg_heart_hp
                    best_hp_y = 1
            if genki < 10:
                shield_reg_gk = self._ocr_shield_anchored_genki(frame, (0.855, 0.066, 0.992, 0.110))
                if shield_reg_gk is not None and shield_reg_gk > genki:
                    genki = shield_reg_gk
                elif genki == 0:
                    gk_reg = frame[int(h * 0.068) : int(h * 0.108), int(w * 0.892) : int(w * 0.988)]
                    if gk_reg.size > 0 and int(((gk_reg[:, :, 2] > 165) & (gk_reg[:, :, 0] < 135)).sum()) >= 20:
                        gk_reg_mw = (gk_reg[:, :, 0] > 200) & (gk_reg[:, :, 1] > 200) & (gk_reg[:, :, 2] > 200)
                        banner_reg_gk = self._ocr_bw_banner_number(gk_reg_mw, prefix="GK:", max_val=999, scale=3)
                        if banner_reg_gk is not None:
                            genki = banner_reg_gk
        if is_exam_hud and hp_found or (not is_exam_hud and best_hp_y >= 0):
            self._last_stamina = (stamina, meta["max_stamina"])
            self._pending_stamina_reading = None
            self._pending_stamina_streak = 0

        # Top-left Logic (👍好印象/✊やる気) & Sense (好調/集中/絶好調) buff diamonds + red feather halfCostTurns
        buffs: Dict[str, int] = {}
        plan_l = (plan or "logic").lower()
        is_logic_plan = plan_l == "logic"
        is_sense_plan = plan_l == "sense"
        diamond_buffs = (
            self._parse_left_buff_diamonds(frame, plan=plan_l)
            if (is_logic_plan or is_sense_plan)
            else {}
        )
        buff_hud_occluded = bool(
            ((is_logic_plan or is_sense_plan) and getattr(self, "_last_buff_hud_occluded", False))
            or self._has_left_buff_toast_in_lines(lines, w, h)
        )
        if diamond_buffs:
            buffs.update(diamond_buffs)
        elif is_logic_plan and not buff_hud_occluded:
            bf_patch = frame[int(h * 0.082) : int(h * 0.225), 0 : int(w * 0.305)]
            bf_mw = (bf_patch[:, :, 0] > 195) & (bf_patch[:, :, 1] > 195) & (bf_patch[:, :, 2] > 195)
            bf_bw = np.where(bf_mw[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
            bf_2x = Image.fromarray(bf_bw).resize((bf_bw.shape[1] * 2, bf_bw.shape[0] * 2), Image.Resampling.BICUBIC)
            bf_res = self.en_ocr.recognize_array(np.array(ImageOps.expand(bf_2x, 15, (255, 255, 255))))
            for bln in bf_res.get("lines", []):
                if bln["bbox"][1] >= 180:
                    m_mot = re.search(r"(\d{1,3})", bln["text"])
                    if m_mot:
                        mv = int(m_mot.group(1))
                        if 1 <= mv <= 200:
                            buffs["motivation"] = mv
                            break
            if "motivation" not in buffs and h >= 400 and w >= 300:
                mot_roi = frame[int(h * 0.125) : int(h * 0.195), int(w * 0.008) : int(w * 0.14)]
                fist_mask = (mot_roi[:, :, 2] > 195) & (mot_roi[:, :, 0] < 120)
                ys_f, _ = np.where(fist_mask)
                if len(ys_f) >= 40:
                    sub_mot = mot_roi[max(0, int(ys_f.min()) - 4) : min(mot_roi.shape[0], int(ys_f.max()) + 10), :]
                    mot_core = self._extract_outlined_white_core(sub_mot, radius=3, white_min=200, dark_max=135)
                    mot_val = self._ocr_bw_banner_number(mot_core, prefix="M:", max_val=200, scale=3, min_val=1)
                    if mot_val is not None:
                        buffs["motivation"] = mot_val
        if h >= 400 and w >= 300:
            fy1, fy2 = int(h * (0.145 if is_exam_hud else 0.105)), int(h * (0.192 if is_exam_hud else 0.160))
            fx1, fx2 = int(w * 0.74), int(w * 0.98)
            f_roi = frame[fy1:fy2, fx1:fx2]
            if f_roi.size > 0 and int(((f_roi[:, :, 0] > 190) & (f_roi[:, :, 1] < 110) & (f_roi[:, :, 2] < 110)).sum()) >= 45:
                f_txt = self._ocr_crop_upscaled(frame, (fx1, fy1, fx2, fy2), scale=3, mode="white", pad=10)
                m_ft = re.search(r"(\d+)\s*ターン", f_txt)
                if m_ft:
                    buffs["halfCostTurns"] = max(buffs.get("halfCostTurns", 0), int(m_ft.group(1)))

        # Detect SKIP (+2 stamina) button on right edge (x=632..708, y=1080..1165 on 720x1600)
        sk_x1, sk_y1, sk_x2, sk_y2 = int(w * 0.875), int(h * 0.675), min(w, int(w * 0.985)), int(h * 0.728)
        sk_pill = frame[sk_y1:sk_y2, sk_x1:sk_x2]
        sk_white = float(((sk_pill[:, :, 0] > 220) & (sk_pill[:, :, 1] > 220) & (sk_pill[:, :, 2] > 220)).mean()) if sk_pill.size else 0.0
        battle_options: List[Dict[str, Any]] = []
        if sk_white > 0.25:
            battle_options.append({
                "choice_id": 0,
                "name": "SKIP",
                "title": "SKIP (跳过回合 +2体力)",
                "action_type": "end_turn",
                "description": "SKIP跳过本回合并恢复+2体力",
                "bbox": BoundingBox(xmin=sk_x1, ymin=sk_y1, xmax=sk_x2, ymax=sk_y2).model_dump(),
            })

        # Recognize bottom hand cards: bottom row slots + dark-glass SSR fallback (ignore center tooltip titles!)
        cards: List[RecognizedCard] = []
        slots = self._detect_hand_card_slots(frame)
        seen_bottom_cxs: List[int] = []
        for x1, y1, x2, y2, is_sel in slots:
            matched: Optional[Dict[str, Any]] = None
            raw_title = ""
            y_top = int(h * 0.871) if is_sel else int(h * 0.878)
            y_bot = int(h * 0.897) if is_sel else int(h * 0.908)
            title_p = frame[y_top:y_bot, max(0, x1 + 4) : min(w, x2 - 4)]
            if title_p.size == 0:
                continue
            if is_sel and h >= 400 and w >= 300:
                # When a card is SELECTed, prefer crisp center tooltip header (handles darkened/paren cards like 私を超えて(翔))
                tip_p = frame[int(h * 0.515) : int(h * 0.575), int(w * 0.16) : int(w * 0.84)]
                if tip_p.size > 0 and float(tip_p.mean()) > 195.0:
                    tip_txt = _clean_ocr_text(self.ocr.recognize_array(tip_p).get("text", ""))
                    tip_clean = re.sub(r"(?:Active|Mental|Trouble|アクティブ|メンタル).*$", "", tip_txt, flags=re.I)
                    tip_clean = re.sub(r"-\d+$", "", tip_clean).strip()
                    matched = self.match_card_by_name(tip_clean or tip_txt, plan=plan)
                    if matched:
                        raw_title = tip_clean or tip_txt
            if not matched:
                t_2x = Image.fromarray(title_p).resize((title_p.shape[1] * 2, title_p.shape[0] * 2), Image.Resampling.BICUBIC)
                t_res = self.ocr.recognize_array(np.array(ImageOps.expand(t_2x, 12, (255, 255, 255))))
                raw_title = _clean_ocr_text(t_res.get("text", ""))
                matched = self.match_card_by_name(raw_title, plan=plan)
            if not matched:
                # Contrast fallback for dimmed/shaded cards
                gray_t = np.mean(title_p[:, :, :3], axis=2)
                bw_t = np.where((gray_t < 155)[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                t2_2x = Image.fromarray(bw_t).resize((bw_t.shape[1] * 2, bw_t.shape[0] * 2), Image.Resampling.BICUBIC)
                raw_title2 = _clean_ocr_text(self.ocr.recognize_array(np.array(ImageOps.expand(t2_2x, 15, (255, 255, 255)))).get("text", ""))
                matched = self.match_card_by_name(raw_title2, plan=plan)
                if matched:
                    raw_title = raw_title2
            if not matched and (is_sel or float(title_p.mean()) < 135.0):
                # Try tooltip preview header when dimmed/unplayable card is focused (e.g. 私を超えて(翔))
                tip_p = frame[int(h * 0.520) : int(h * 0.575), int(w * 0.18) : int(w * 0.82)]
                if tip_p.size > 0:
                    tip_txt = _clean_ocr_text(self.ocr.recognize_array(tip_p).get("text", ""))
                    tip_clean = re.sub(r"(?:Active|Mental|Trouble|アクティブ|メンタル).*$", "", tip_txt, flags=re.I)
                    matched = self.match_card_by_name(tip_clean or tip_txt, plan=plan)
                    if matched:
                        raw_title = tip_clean or tip_txt
            if not matched and is_sel:
                # Fallback to standard non-elevated title strip if gold rim triggered elevated crop
                alt_p = frame[int(h * 0.878) : int(h * 0.908), max(0, x1 + 4) : min(w, x2 - 4)]
                if alt_p.size > 0:
                    a2 = Image.fromarray(alt_p).resize((alt_p.shape[1] * 2, alt_p.shape[0] * 2), Image.Resampling.BICUBIC)
                    raw_alt = _clean_ocr_text(self.ocr.recognize_array(np.array(ImageOps.expand(a2, 12, (255, 255, 255)))).get("text", ""))
                    matched = self.match_card_by_name(raw_alt, plan=plan)
                    if matched:
                        raw_title = raw_alt
                        title_p = alt_p
            if matched:
                cx = (x1 + x2) // 2
                seen_bottom_cxs.append(cx)
                rt_y1 = int(h * 0.890) if not is_sel else int(h * 0.878)
                rt_y2 = int(h * 0.904) if not is_sel else int(h * 0.896)
                rt = frame[rt_y1:rt_y2, x1 + int((x2 - x1) * 0.55) : min(w, x2 - 4)]
                card_body = frame[int(h * 0.745) : int(h * 0.875), max(0, x1 + 6) : min(w, x2 - 6)]
                visual_playable = (
                    bool(float(np.percentile(card_body, 95)) >= 185.0)
                    if (card_body.size > 0 and float(card_body.max()) > 0.0)
                    else None
                )
                is_trouble_matched = (
                    int(matched.get("id") or 0) == 23
                    or str(matched.get("type") or "").lower() == "trouble"
                )
                if is_trouble_matched:
                    # Trouble (#23 眠気) always has a dark charcoal title strip (~105 mean) and costs 0 stamina;
                    # judge playability by its silver body brightness (visual_playable) instead of title strip luminance.
                    playable_flag = True if visual_playable is not False else False
                else:
                    playable_flag = float(title_p.mean()) >= 145.0 or float(np.percentile(title_p, 88)) >= 200.0
                    if visual_playable is not None:
                        playable_flag = playable_flag and visual_playable
                is_cust, cust_cnt = self._detect_hand_card_customization(frame[y1:y2, x1:x2])
                cards.append(
                    self._build_recognized_card(
                        matched,
                        raw_title,
                        rt,
                        BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2),
                        selected=is_sel,
                        playable=playable_flag,
                        visual_playable=visual_playable,
                        customized=is_cust,
                        customization_count=cust_cnt,
                    )
                )

        # Fallback: recover dark-glass SSR cards (e.g. 私を超えて(翔)) strictly in bottom hand title strip
        half_card_w = max(70, int(round(w * 0.129)))
        min_sep = max(110, int(round(w * 0.170)))
        for ln in lines:
            bx1, by1, bx2, by2 = ln["bbox"]
            cy = (by1 + by2) // 2
            cx = (bx1 + bx2) // 2
            if int(h * 0.84) <= cy <= int(h * 0.93):
                matched = self.match_card_by_name(ln["text"], plan=plan)
                if not matched:
                    continue
                if any(abs(cx - prev) < min_sep for prev in seen_bottom_cxs):
                    continue
                seen_bottom_cxs.append(cx)
                c_x1, c_x2 = max(5, cx - half_card_w), min(w - 5, cx + half_card_w)
                c_y1, c_y2 = int(h * 0.74), int(h * 0.91)
                title_strip = frame[max(0, by1 - 6) : min(h, by2 + 6), c_x1:c_x2]
                bright_title_px = int((np.max(title_strip, axis=2) > 180).sum()) if title_strip.size else 0
                playable_fb = bool(
                    bright_title_px >= 35
                    or float(title_strip.mean()) >= 145.0
                    or (title_strip.size > 0 and float(np.percentile(title_strip, 88)) >= 200.0)
                )
                rt_fb = frame[int(h * 0.890) : int(h * 0.904), cx : min(w, c_x2 - 4)]
                sel_band_fb = frame[int(h * 0.902) : int(h * 0.934), c_x1:c_x2]
                sel_fb = bool(
                    sel_band_fb.size > 0
                    and int(((sel_band_fb[:, :, 0] > 215) & (sel_band_fb[:, :, 1] > 150) & (sel_band_fb[:, :, 2] < 95)).sum()) >= 220
                )
                card_body_fb = frame[int(h * 0.745) : int(h * 0.875), max(0, c_x1 + 6) : min(w, c_x2 - 6)]
                vp_fb = (
                    bool(float(np.percentile(card_body_fb, 95)) >= 185.0)
                    if (card_body_fb.size > 0 and float(card_body_fb.max()) > 0.0)
                    else None
                )
                is_cust_fb, cust_cnt_fb = self._detect_hand_card_customization(frame[c_y1:c_y2, c_x1:c_x2])
                cards.append(
                    self._build_recognized_card(
                        matched,
                        ln["text"],
                        rt_fb,
                        BoundingBox(xmin=c_x1, ymin=c_y1, xmax=c_x2, ymax=c_y2),
                        selected=sel_fb,
                        playable=playable_fb,
                        visual_playable=vp_fb,
                        customized=is_cust_fb,
                        customization_count=cust_cnt_fb,
                    )
                )

        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
        clean_ring_hdr = re.sub(r"[/\\]+", "", ring_hdr or "").strip() or (f"CLEARまで {target_score}" if (not is_exam_hud and target_score > 0) else "実戦")
        if not is_exam_hud and re.search(r"PERF|P[E3]RF", (ring_hdr or "").upper()):
            clean_ring_hdr = f"PERFECTまで {target_score}" if target_score > 0 else "PERFECTまで"
        if is_exam_hud:
            total_turns_est = (13 if meta["week"] >= 9 else 10) if self.scenario_id in ("hif_honsen", "hif") else 10
        else:
            total_turns_est = 7 if meta["week"] <= 4 else 9
        return ScreenObservation(
            scene_type="battle_turn",
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=f"{meta['week_label'] or ctx['week_node'].title} ({clean_ring_hdr})",
            turn=max(1, total_turns_est - turns_remaining),
            turns_remaining=turns_remaining,
            stamina=stamina,
            max_stamina=meta["max_stamina"],
            genki=genki,
            score=score,
            target_score=target_score,
            p_points=meta["p_points"],
            buffs=buffs,
            buff_hud_occluded=buff_hud_occluded,
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            p_item_remaining_counts=self._parse_right_p_item_counters(frame),
            stage_bonuses=ctx["stage_bonuses"],
            cards=cards,
            hand_slot_count=max(len(slots), len(cards)),
            drinks=self._recognize_bottom_drinks(frame, plan),
            options=battle_options,
            raw_text="\n".join([turn_box_txt, ring_hdr, ring_num_txt] + [ln["text"] for ln in lines]),
        )

    def _parse_rank_result_score(
        self,
        frame: np.ndarray,
        lines: List[Dict[str, Any]],
        w: int,
        h: int,
    ) -> Tuple[int, str]:
        """Parse top player's settlement score (base + cyan bonus) on 現在順位 / 最終順位 result screens."""
        full_text = "".join(str(ln.get("text") or "") for ln in lines)
        if not any(tok in full_text for tok in ("現在順位", "最終順位", "最終スコア")):
            return (0, "")
        if h < 200 or w < 200:
            return (0, "")

        rank_ln = next(
            (ln for ln in lines if "順位" in str(ln.get("text") or "") and ln["bbox"][1] < int(h * 0.35)),
            None,
        )
        if rank_ln is not None:
            y1 = min(h - 10, int(rank_ln["bbox"][3]) + int(h * 0.030))
            y2 = min(h, int(rank_ln["bbox"][3]) + int(h * 0.115))
        else:
            name_ln = next(
                (
                    ln for ln in lines
                    if int(h * 0.14) <= ln["bbox"][1] <= int(h * 0.30) and ln["bbox"][0] >= int(w * 0.25)
                ),
                None,
            )
            if name_ln is not None:
                y1 = min(h - 10, int(name_ln["bbox"][3]))
                y2 = min(h, int(name_ln["bbox"][3]) + int(h * 0.068))
            else:
                y1, y2 = int(h * 0.19), int(h * 0.265)
        x1, x2 = int(w * 0.24), min(w, int(w * 0.96))
        if y2 <= y1 + 4 or x2 <= x1 + 10:
            return (0, "")
        band = frame[y1:y2, x1:x2]
        if band.size == 0:
            return (0, "")

        from PIL import ImageOps
        num_pat = r"\d{1,3}(?:\s*[,\.]\s*\d{3})+|\d{5,8}"
        def _extract_nums(s: str) -> List[int]:
            return [int(re.sub(r"[\s,\.]", "", m)) for m in re.findall(num_pat, s or "")]

        scale_raw = 3 if band.shape[0] < 60 else 2
        raw_pil = Image.fromarray(band).resize((band.shape[1] * scale_raw, band.shape[0] * scale_raw), Image.Resampling.BICUBIC)
        raw_txt = str(self.en_ocr.recognize_array(np.array(ImageOps.expand(raw_pil, 20, (20, 20, 20)))).get("text", "") or "")
        raw_vals = _extract_nums(raw_txt)

        inv_vals: List[int] = []
        if len(raw_vals) < 2:
            # Smooth inverted grayscale cropped tightly around bright digits preserves TrueType stroke antialiasing
            bright_m = band.max(axis=2) > 120
            if bright_m.any():
                ys_b, xs_b = np.where(bright_m)
                sub_band = band[
                    max(0, int(ys_b.min()) - 6) : min(band.shape[0], int(ys_b.max()) + 7),
                    max(0, int(xs_b.min()) - 6) : min(band.shape[1], int(xs_b.max()) + 7),
                ]
            else:
                sub_band = band
            inv_g = 255 - sub_band.max(axis=2)
            inv_rgb = np.stack([inv_g, inv_g, inv_g], axis=2)
            inv_scale = 3 if sub_band.shape[0] < 48 else 2
            p_inv = Image.fromarray(inv_rgb).resize((inv_rgb.shape[1] * inv_scale, inv_rgb.shape[0] * inv_scale), Image.Resampling.BICUBIC)
            inv_txt = str(self.en_ocr.recognize_array(np.array(ImageOps.expand(p_inv, 20, (255, 255, 255)))).get("text", "") or "")
            inv_vals = _extract_nums(inv_txt)

        cyan_vals: List[int] = []
        if len(raw_vals) < 2 or "+" not in raw_txt:
            cm = (
                (band[:, :, 2] > 150)
                & (band[:, :, 1] > 120)
                & ((band[:, :, 2].astype(np.int16) - band[:, :, 0].astype(np.int16)) > 35)
            )
            if int(cm.sum()) >= 15:
                ys_c, xs_c = np.where(cm)
                c_sub_mask = cm[
                    max(0, int(ys_c.min()) - 4) : min(band.shape[0], int(ys_c.max()) + 5),
                    max(0, int(xs_c.min()) - 4) : min(band.shape[1], int(xs_c.max()) + 6),
                ]
                bw_c = np.where(c_sub_mask[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                c_scale = 5 if band.shape[0] < 60 else 2
                p_c = Image.fromarray(bw_c).resize((bw_c.shape[1] * c_scale, bw_c.shape[0] * c_scale), Image.Resampling.BICUBIC)
                c_txt = str(self.en_ocr.recognize_array(np.array(ImageOps.expand(p_c, 20, (255, 255, 255)))).get("text", "") or "")
                cyan_vals = _extract_nums(c_txt)
                if not cyan_vals:
                    c_pad40 = str(
                        self.en_ocr.recognize_array(np.array(ImageOps.expand(p_c, 40, (255, 255, 255)))).get("text", "")
                        or ""
                    )
                    cyan_vals = _extract_nums(c_pad40)
                if not cyan_vals:
                    c_nat = str(
                        self.en_ocr.recognize_array(
                            np.array(ImageOps.expand(Image.fromarray(bw_c), 40, (255, 255, 255)))
                        ).get("text", "")
                        or ""
                    )
                    cyan_vals = _extract_nums(c_nat)
                if not cyan_vals:
                    bw_c_val = self._ocr_bw_banner_number(
                        c_sub_mask,
                        prefix="HP:",
                        max_val=99999999,
                        scale=2,
                        min_val=1,
                        max_digits=9,
                        en_first=True,
                    )
                    if bw_c_val is not None:
                        cyan_vals = [bw_c_val]

        if len(raw_vals) >= 2 and "+" in raw_txt:
            base_s, bonus_s = raw_vals[0], raw_vals[1]
            total_s = base_s + bonus_s
            return (total_s, f"現在順位 {base_s:,}+{bonus_s:,}={total_s:,}")
        cyan_v = cyan_vals[0] if cyan_vals else None

        # Isolate pure white base digits if raw/inv missed any non-cyan base score
        white_vals: List[int] = []
        wm = (
            (band[:, :, 0] > 175)
            & (band[:, :, 1] > 175)
            & (band[:, :, 2] > 175)
            & ((band[:, :, 2].astype(np.int16) - band[:, :, 0].astype(np.int16)) < 45)
        )
        has_non_cyan = any(cyan_v is None or v != cyan_v for v in (inv_vals + raw_vals))
        if (not has_non_cyan) and int(wm.sum()) >= 20:
            ys_w, xs_w = np.where(wm)
            wy1, wy2 = max(0, int(ys_w.min()) - 6), min(band.shape[0], int(ys_w.max()) + 7)
            wx1, wx2 = max(0, int(xs_w.min()) - 6), min(band.shape[1], int(xs_w.max()) + 7)
            w_sub = wm[wy1:wy2, wx1:wx2]
            w_max = band[wy1:wy2, wx1:wx2].max(axis=2)
            inv_w = np.where(w_sub, 255 - w_max, 255).astype(np.uint8)
            bw_w = np.stack([inv_w, inv_w, inv_w], axis=2)
            w_scale = 3 if bw_w.shape[0] < 48 else 2
            p_w = Image.fromarray(bw_w).resize((bw_w.shape[1] * w_scale, bw_w.shape[0] * w_scale), Image.Resampling.BICUBIC)
            w_txt = str(self.en_ocr.recognize_array(np.array(ImageOps.expand(p_w, 20, (255, 255, 255)))).get("text", "") or "")
            white_vals = _extract_nums(w_txt)
            if not white_vals:
                bw_bin = np.where(w_sub[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
                w_bin_txt = str(
                    self.en_ocr.recognize_array(
                        np.array(ImageOps.expand(Image.fromarray(bw_bin), 40, (255, 255, 255)))
                    ).get("text", "")
                    or ""
                )
                white_vals = _extract_nums(w_bin_txt)
            if not white_vals:
                bw_val = self._ocr_bw_banner_number(
                    w_sub,
                    prefix="HP:",
                    max_val=99999999,
                    scale=2,
                    min_val=1,
                    max_digits=9,
                    en_first=True,
                )
                if bw_val is not None:
                    white_vals = [bw_val]

        base_candidates = [v for v in (inv_vals + white_vals + raw_vals) if cyan_v is None or v != cyan_v]
        if base_candidates and cyan_v is not None:
            base_s = base_candidates[0]
            total_s = base_s + cyan_v
            return (total_s, f"現在順位 {base_s:,}+{cyan_v:,}={total_s:,}")
        all_vals = inv_vals + white_vals + raw_vals
        if len(all_vals) >= 2 and all_vals[0] != all_vals[1]:
            base_s, bonus_s = all_vals[0], all_vals[1]
            total_s = base_s + bonus_s
            return (total_s, f"現在順位 {base_s:,}+{bonus_s:,}={total_s:,}")
        if all_vals:
            return (all_vals[0], f"現在順位 {all_vals[0]:,}")
        if cyan_v is not None:
            return (cyan_v, f"現在順位 +{cyan_v:,}")
        return (0, "")

    def _parse_generic_screen(self, frame: np.ndarray, lines: List[Dict[str, Any]], plan: str, meta: Dict[str, Any]) -> ScreenObservation:
        h, w = frame.shape[:2]
        full_text = "".join(ln["text"] for ln in lines)
        options = self._extract_bottom_and_list_options(lines, w, h, frame=frame)
        scene = "popup" if (
            any(k in full_text for k in ("確認", "決定", "はい", "キャンセル", "閉じる", "次へ", "再開"))
            or any(o.get("name") == "閉じる" for o in options)
        ) else "skip"
        rank_score, rank_label = self._parse_rank_result_score(frame, lines, w, h)
        ctx = compute_week_stats_and_stage_bonus(scenario_id=self.scenario_id, week=meta["week"], plan=plan, override_stats=meta["stats"])
        wk_act = f"{meta['week_label'] or ctx['week_node'].title} | 持有P点:{meta['p_points']}"
        if rank_label:
            wk_act = f"{wk_act} | {rank_label}"
        return ScreenObservation(
            scene_type=scene,
            plan=plan,
            scenario_id=self.scenario_id,
            week=meta["week"],
            total_weeks=meta["total_weeks"],
            week_activity=wk_act,
            stamina=meta["stamina"],
            max_stamina=meta["max_stamina"],
            score=rank_score,
            p_points=meta["p_points"],
            idol_stats=meta["stats"],
            p_items=ctx["p_items"],
            stage_bonuses=ctx["stage_bonuses"],
            options=options,
            raw_text="\n".join(ln["text"] for ln in lines),
        )
