"""
Mock Frame Generator for Gakumas Demo.
Synthesizes realistic 1080x1920 game frames by reading real card/drink icon assets
and compositing HUD elements, status indicators, and candidate actions.
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from demo.schemas import (
    BoundingBox,
    RecognizedCard,
    RecognizedDrink,
    ScreenObservation,
)
from demo.core.db import GakumasDB
from demo.vision.roi import (
    BASE_WIDTH,
    BASE_HEIGHT,
    STATUS_WEEK_BBOX,
    STATUS_TURN_BBOX,
    STATUS_STAMINA_BBOX,
    STATUS_GENKI_BBOX,
    STATUS_SCORE_BBOX,
    STATUS_BUFFS_BBOX,
    END_TURN_BBOX,
    get_hand_bboxes,
    get_drink_bboxes,
)


class MockFrameGenerator:
    """
    Synthesizes standard 1080x1920 Gakumas game frames for offline testing,
    round-trip accuracy verification, and autonomous demo mode.
    """

    BUFF_COLORS = {
        "good_condition": (220, 130, 30),     # Orange
        "concentration": (40, 160, 230),       # Cyan / Blue
        "motivation": (220, 60, 200),          # Purple / Pink
        "good_impression": (50, 200, 60),      # Green
        "strong_spirit": (220, 210, 40),       # Yellow
        "full_effort": (230, 160, 40),         # Gold / Orange
        "preserve": (80, 180, 230),            # Sky Blue
    }

    def __init__(self, db: Optional[GakumasDB] = None, plan: str = "sense"):
        self.db = db or GakumasDB()
        self.plan = plan.lower()

    def _create_background(self, scene_type: str = "battle_turn") -> Image.Image:
        """Create a gradient background mimicking the game's stage aesthetic."""
        img = Image.new("RGB", (BASE_WIDTH, BASE_HEIGHT), (25, 20, 40))
        draw = ImageDraw.Draw(img)

        # Subtle vertical gradient
        for y in range(0, BASE_HEIGHT, 4):
            ratio = y / float(BASE_HEIGHT)
            if ratio < 0.25:
                # Top header bar (dark)
                r = int(18 + 10 * ratio)
                g = int(16 + 10 * ratio)
                b = int(30 + 15 * ratio)
            elif ratio < 0.70:
                # Mid stage area (subtle spotlight/stage purple)
                sub_ratio = (ratio - 0.25) / 0.45
                r = int(28 + 35 * (1.0 - abs(sub_ratio - 0.5) * 2))
                g = int(24 + 25 * (1.0 - abs(sub_ratio - 0.5) * 2))
                b = int(45 + 50 * (1.0 - abs(sub_ratio - 0.5) * 2))
            else:
                # Bottom hand card zone
                r = int(22 + 8 * (1.0 - ratio))
                g = int(20 + 8 * (1.0 - ratio))
                b = int(32 + 10 * (1.0 - ratio))
            draw.rectangle([0, y, BASE_WIDTH, y + 3], fill=(r, g, b))

        return img

    def _render_hud(
        self,
        img: Image.Image,
        stamina: int,
        max_stamina: int,
        genki: int,
        turn: int,
        turns_remaining: int,
        score: int,
        target_score: int,
        buffs: Dict[str, int],
    ) -> None:
        """Render status HUD elements (Turn, Stamina, Genki, Score, Buffs)."""
        draw = ImageDraw.Draw(img)

        # 1. Turn indicator box
        t_box = STATUS_TURN_BBOX
        draw.rectangle([t_box.xmin, t_box.ymin, t_box.xmax, t_box.ymax], fill=(35, 30, 50), outline=(90, 80, 110), width=2)
        # Text rendered via cv2 for perfect canonical OCR alignment if cv2 available
        # We will write text using cv2 on the numpy array later or Pillow text

        # 2. Stamina HP bar
        s_box = STATUS_STAMINA_BBOX
        draw.rectangle([s_box.xmin, s_box.ymin, s_box.xmax, s_box.ymax], fill=(30, 25, 45), outline=(70, 65, 90), width=2)
        # Left side: HP bar (x in [xmin + 10, xmin + 190])
        bar_max_w = 180
        pct = max(0.0, min(1.0, stamina / float(max_stamina if max_stamina > 0 else 30)))
        bar_w = int(bar_max_w * pct)
        if bar_w > 0:
            draw.rectangle([s_box.xmin + 10, s_box.ymin + 24, s_box.xmin + 10 + bar_w, s_box.ymin + 46], fill=(45, 175, 95))

        # 3. Genki bar
        g_box = STATUS_GENKI_BBOX
        draw.rectangle([g_box.xmin, g_box.ymin, g_box.xmax, g_box.ymax], fill=(30, 25, 45), outline=(60, 90, 140), width=2)
        # Left side: Genki shield icon
        draw.ellipse([g_box.xmin + 12, g_box.ymin + 14, g_box.xmin + 48, g_box.ymin + 46], fill=(45, 120, 210), outline=(120, 180, 250), width=2)

        # 4. Score box
        sc_box = STATUS_SCORE_BBOX
        draw.rectangle([sc_box.xmin, sc_box.ymin, sc_box.xmax, sc_box.ymax], fill=(30, 25, 45), outline=(70, 65, 90), width=2)

        # 5. Buff bar
        b_box = STATUS_BUFFS_BBOX
        draw.rectangle([b_box.xmin, b_box.ymin, b_box.xmax, b_box.ymax], fill=(25, 22, 38), outline=(60, 55, 80), width=1)
        cur_x = b_box.xmin + 15
        badge_y = b_box.ymin + 10
        for buff_name, stack in buffs.items():
            color = self.BUFF_COLORS.get(buff_name, (200, 150, 40))
            # Draw badge circle
            draw.ellipse([cur_x, badge_y, cur_x + 36, badge_y + 36], fill=color, outline=(255, 255, 255), width=2)
            cur_x += 80

        # 6. End turn button
        e_box = END_TURN_BBOX
        draw.rectangle([e_box.xmin, e_box.ymin, e_box.xmax, e_box.ymax], fill=(180, 50, 70), outline=(240, 120, 130), width=3)

    def _render_drinks(
        self,
        img: Image.Image,
        drinks: List[Union[Dict[str, Any], int, RecognizedDrink]],
    ) -> List[RecognizedDrink]:
        """Render up to 3 drink slots (filled or empty)."""
        draw = ImageDraw.Draw(img)
        bboxes = get_drink_bboxes(count=3)
        recognized_drinks: List[RecognizedDrink] = []

        for i, bbox in enumerate(bboxes):
            drink_input = drinks[i] if i < len(drinks) else None
            # Draw drink slot circular background
            cx, cy = bbox.center()
            radius = min(bbox.width, bbox.height) // 2 - 4
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(35, 30, 50), outline=(75, 70, 95), width=2)

            if drink_input is None:
                continue

            # Resolve drink data
            drink_data: Optional[Dict[str, Any]] = None
            if isinstance(drink_input, RecognizedDrink):
                drink_data = self.db.get_drink_by_id(drink_input.drink_id)
            elif isinstance(drink_input, dict):
                did = drink_input.get("id") or drink_input.get("drink_id")
                drink_data = self.db.get_drink_by_id(did) if did else None
            elif isinstance(drink_input, int):
                drink_data = self.db.get_drink_by_id(drink_input)

            if not drink_data:
                continue

            icon_path = drink_data.get("icon_path")
            if icon_path and Path(icon_path).exists():
                try:
                    icon_img = Image.open(icon_path).convert("RGB")
                    # Fit icon nicely inside circle (inner size matches recognizer inner crop)
                    icon_size = (radius * 2 - 4, radius * 2 - 4)
                    icon_resized = icon_img.resize(icon_size)
                    img.paste(icon_resized, (cx - icon_size[0] // 2, cy - icon_size[1] // 2))

                    recognized_drinks.append(RecognizedDrink(
                        drink_id=drink_data["id"],
                        name=drink_data.get("name") or "",
                        playable=True,
                        bbox=bbox,
                        dsl_summary=drink_data.get("description") or "",
                    ))
                except Exception:
                    pass

        return recognized_drinks

    def _render_cards(
        self,
        img: Image.Image,
        cards: List[Union[Dict[str, Any], int, RecognizedCard]],
    ) -> List[RecognizedCard]:
        """Render hand cards into horizontal bottom slots."""
        count = len(cards)
        bboxes = get_hand_bboxes(count=count)
        recognized_cards: List[RecognizedCard] = []

        for i, bbox in enumerate(bboxes):
            card_input = cards[i]
            card_data: Optional[Dict[str, Any]] = None
            is_playable = True
            is_upgraded = False

            if isinstance(card_input, RecognizedCard):
                card_data = self.db.get_card_by_id(card_input.card_id)
                is_playable = card_input.playable
                is_upgraded = card_input.upgraded
            elif isinstance(card_input, dict):
                cid = card_input.get("id") or card_input.get("card_id")
                card_data = self.db.get_card_by_id(cid) if cid else None
                is_playable = card_input.get("playable", True)
                is_upgraded = bool(card_input.get("upgraded", False))
            elif isinstance(card_input, int):
                card_data = self.db.get_card_by_id(card_input)
                is_playable = True
                is_upgraded = bool(card_data.get("upgraded", 0) == 1) if card_data else False

            if not card_data:
                continue

            draw = ImageDraw.Draw(img)
            # Card base background and border
            ctype = card_data.get("type") or "active"
            border_color = (200, 90, 80) if ctype == "active" else ((70, 150, 220) if ctype == "mental" else (160, 80, 180))
            bg_color = (40, 32, 50) if is_playable else (25, 20, 30)

            draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], fill=bg_color, outline=border_color, width=3)

            # Paste card icon
            icon_path = card_data.get("icon_path")
            card_w = bbox.width
            card_h = bbox.height
            icon_box_w = int(card_w * 0.80)
            icon_box_h = int(card_h * 0.58)
            icon_x = bbox.xmin + int(card_w * 0.10)
            icon_y = bbox.ymin + int(card_h * 0.11)

            if icon_path and Path(icon_path).exists():
                try:
                    icon_img = Image.open(icon_path).convert("RGB")
                    icon_resized = icon_img.resize((icon_box_w, icon_box_h))

                    # Apply dimming filter if unplayable
                    if not is_playable:
                        enhancer = ImageEnhance.Brightness(icon_resized)
                        icon_resized = enhancer.enhance(0.45)

                    img.paste(icon_resized, (icon_x, icon_y))
                except Exception:
                    pass

            # Render Upgraded '+' Badge in top-right corner
            if is_upgraded or card_data.get("upgraded", 0) == 1:
                badge_radius = 18
                bx = bbox.xmax - 28
                by = bbox.ymin + 28
                draw.ellipse([bx - badge_radius, by - badge_radius, bx + badge_radius, by + badge_radius], fill=(235, 185, 35), outline=(255, 240, 120), width=2)
                # '+' symbol inside badge
                draw.line([bx - 10, by, bx + 10, by], fill=(30, 20, 10), width=4)
                draw.line([bx, by - 10, bx, by + 10], fill=(30, 20, 10), width=4)

            recognized_cards.append(RecognizedCard(
                card_id=card_data["id"],
                name=card_data.get("name") or "",
                upgraded=is_upgraded or bool(card_data.get("upgraded", 0) == 1),
                playable=is_playable,
                bbox=bbox,
                hp_loss=0,
                dsl_summary=card_data.get("description") or "",
                type=card_data.get("type") or "active",
                rarity=card_data.get("rarity") or "N",
            ))

        return recognized_cards

    def _render_text_overlay(
        self,
        frame: np.ndarray,
        stamina: int,
        max_stamina: int,
        genki: int,
        turn: int,
        turns_remaining: int,
        score: int,
        target_score: int,
        buffs: Dict[str, int],
    ) -> np.ndarray:
        """Render canonical digital text onto the status regions."""
        if not _HAS_CV2:
            return frame

        # Text strings
        turn_str = f"{turn}/{turns_remaining}"
        hp_str = f"{stamina}/{max_stamina}"
        genki_str = f"{genki}"
        score_str = f"{score}/{target_score}"

        # 1. Turn text
        t_box = STATUS_TURN_BBOX
        for i, ch in enumerate(turn_str):
            cv2.putText(frame, ch, (t_box.xmin + 15 + i * 22, t_box.ymin + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 2. HP text (on the right half of s_box, xmin + 205)
        s_box = STATUS_STAMINA_BBOX
        for i, ch in enumerate(hp_str):
            cv2.putText(frame, ch, (s_box.xmin + 205 + i * 22, s_box.ymin + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 3. Genki text (on the right of genki shield, xmin + 70)
        g_box = STATUS_GENKI_BBOX
        for i, ch in enumerate(genki_str):
            cv2.putText(frame, ch, (g_box.xmin + 70 + i * 22, g_box.ymin + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 4. Score text
        sc_box = STATUS_SCORE_BBOX
        for i, ch in enumerate(score_str):
            cv2.putText(frame, ch, (sc_box.xmin + 30 + i * 22, sc_box.ymin + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 5. Buff stacks
        b_box = STATUS_BUFFS_BBOX
        cur_x = b_box.xmin + 15
        for buff_name, stack in buffs.items():
            stack_str = str(stack)
            for i, ch in enumerate(stack_str):
                cv2.putText(frame, ch, (cur_x + 42 + i * 16, b_box.ymin + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cur_x += 80

        # 6. End turn button text
        e_box = END_TURN_BBOX
        cv2.putText(frame, "END TURN", (e_box.xmin + 20, e_box.ymin + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return frame

    def generate_battle_frame(
        self,
        cards: Optional[List[Union[Dict[str, Any], int, RecognizedCard]]] = None,
        drinks: Optional[List[Union[Dict[str, Any], int, RecognizedDrink]]] = None,
        stamina: int = 30,
        max_stamina: int = 30,
        genki: int = 0,
        turn: int = 1,
        turns_remaining: int = 10,
        score: int = 0,
        target_score: int = 5000,
        buffs: Optional[Dict[str, int]] = None,
        plan: Optional[str] = None,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """Synthesize a complete battle_turn frame and ground truth ScreenObservation."""
        active_plan = (plan or self.plan).lower()
        if buffs is None:
            buffs = {}

        # Default cards if none provided: pick sample cards for plan
        if cards is None:
            available = [c for c in self.db.get_cards_for_plan(active_plan) if c.get("icon_path") and Path(c["icon_path"]).exists()]
            sample_count = min(3, len(available))
            cards = available[:sample_count] if sample_count > 0 else []

        # Default drinks if none provided: pick sample drinks for plan
        if drinks is None:
            available_drinks = [d for d in self.db.get_drinks_for_plan(active_plan) if d.get("icon_path") and Path(d["icon_path"]).exists()]
            drinks = available_drinks[:2] if len(available_drinks) >= 2 else []

        img = self._create_background(scene_type="battle_turn")
        self._render_hud(
            img,
            stamina=stamina,
            max_stamina=max_stamina,
            genki=genki,
            turn=turn,
            turns_remaining=turns_remaining,
            score=score,
            target_score=target_score,
            buffs=buffs,
        )
        rec_drinks = self._render_drinks(img, drinks)
        rec_cards = self._render_cards(img, cards)

        # Convert to numpy array (RGB)
        frame_arr = np.array(img, dtype=np.uint8)

        # Render digital text onto status regions
        frame_arr = self._render_text_overlay(
            frame_arr,
            stamina=stamina,
            max_stamina=max_stamina,
            genki=genki,
            turn=turn,
            turns_remaining=turns_remaining,
            score=score,
            target_score=target_score,
            buffs=buffs,
        )

        obs = ScreenObservation(
            scene_type="battle_turn",
            turn=turn,
            turns_remaining=turns_remaining,
            stamina=stamina,
            max_stamina=max_stamina,
            genki=genki,
            score=score,
            target_score=target_score,
            card_uses_remaining=1,
            plan=active_plan,
            buffs=buffs,
            cards=rec_cards,
            drinks=rec_drinks,
            options=[],
        )

        return frame_arr, obs

    def generate_produce_schedule_frame(
        self,
        options: Optional[List[Dict[str, Any]]] = None,
        plan: Optional[str] = None,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """Synthesize a produce_schedule scene frame with selectable lessons/activities."""
        active_plan = (plan or self.plan).lower()
        img = self._create_background(scene_type="produce_schedule")
        draw = ImageDraw.Draw(img)

        # Header title
        draw.rectangle([60, 100, 1020, 200], fill=(45, 38, 65), outline=(100, 90, 130), width=2)
        if _HAS_CV2:
            frame_arr = np.array(img, dtype=np.uint8)
            cv2.putText(frame_arr, "PRODUCE SCHEDULE", (320, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            img = Image.fromarray(frame_arr)
            draw = ImageDraw.Draw(img)

        default_options = options or [
            {"choice_id": 0, "name": "Vocal Lesson", "type": "lesson", "bonus": "+30 Vo"},
            {"choice_id": 1, "name": "Dance Lesson", "type": "lesson", "bonus": "+30 Da"},
            {"choice_id": 2, "name": "Visual Lesson", "type": "lesson", "bonus": "+30 Vi"},
        ]

        obs_options = []
        y_start = 450
        box_h = 240
        gap = 60

        for i, opt in enumerate(default_options):
            y1 = y_start + i * (box_h + gap)
            y2 = y1 + box_h
            bbox = BoundingBox(xmin=100, ymin=y1, xmax=980, ymax=y2)
            draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], fill=(50, 42, 70), outline=(130, 110, 170), width=3)
            # Button zone
            draw.rectangle([760, y1 + 70, 940, y2 - 70], fill=(200, 140, 50), outline=(240, 180, 80), width=2)

            obs_options.append({
                "choice_id": opt.get("choice_id", i),
                "name": opt.get("name", f"Option {i}"),
                "type": opt.get("type", "schedule"),
                "bbox": bbox.model_dump(),
            })

        frame_arr = np.array(img, dtype=np.uint8)
        if _HAS_CV2:
            for i, opt in enumerate(default_options):
                y1 = y_start + i * (box_h + gap)
                cv2.putText(frame_arr, opt.get("name", ""), (140, y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame_arr, "SELECT", (800, y1 + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        obs = ScreenObservation(
            scene_type="produce_schedule",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=active_plan,
            options=obs_options,
        )
        return frame_arr, obs

    def generate_card_reward_frame(
        self,
        cards: Optional[List[Union[Dict[str, Any], int]]] = None,
        plan: Optional[str] = None,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """Synthesize a card_reward scene frame presenting 3 cards to choose from."""
        active_plan = (plan or self.plan).lower()
        img = self._create_background(scene_type="card_reward")
        draw = ImageDraw.Draw(img)

        # Title
        draw.rectangle([100, 200, 980, 320], fill=(45, 38, 65), outline=(120, 100, 150), width=2)

        if cards is None:
            available = [c for c in self.db.get_cards_for_plan(active_plan) if c.get("icon_path") and Path(c["icon_path"]).exists()]
            cards = available[:3] if len(available) >= 3 else []

        obs_options = []
        card_w = 260
        card_h = 360
        gap = 50
        x_start = (BASE_WIDTH - (len(cards) * card_w + (len(cards) - 1) * gap)) // 2
        y_pos = 750

        for i, card_input in enumerate(cards):
            card_data = self.db.get_card_by_id(card_input) if isinstance(card_input, int) else card_input
            if not card_data:
                continue

            x1 = x_start + i * (card_w + gap)
            y1 = y_pos
            bbox = BoundingBox(xmin=x1, ymin=y1, xmax=x1 + card_w, ymax=y1 + card_h)

            draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], fill=(40, 35, 55), outline=(180, 150, 80), width=3)
            # Paste icon
            icon_path = card_data.get("icon_path")
            if icon_path and Path(icon_path).exists():
                try:
                    icon_img = Image.open(icon_path).convert("RGB").resize((int(card_w * 0.8), int(card_h * 0.58)))
                    img.paste(icon_img, (bbox.xmin + int(card_w * 0.1), bbox.ymin + int(card_h * 0.1)))
                except Exception:
                    pass

            obs_options.append({
                "choice_id": i,
                "card_id": card_data["id"],
                "name": card_data.get("name") or "",
                "bbox": bbox.model_dump(),
            })

        frame_arr = np.array(img, dtype=np.uint8)
        if _HAS_CV2:
            cv2.putText(frame_arr, "CARD REWARD - CHOOSE 1", (280, 275), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            for i, opt in enumerate(obs_options):
                x1 = x_start + i * (card_w + gap)
                cv2.putText(frame_arr, "CHOOSE", (x1 + 60, y_pos + card_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        obs = ScreenObservation(
            scene_type="card_reward",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=active_plan,
            options=obs_options,
        )
        return frame_arr, obs

    def generate_shop_frame(
        self,
        drinks: Optional[List[Union[Dict[str, Any], int]]] = None,
        plan: Optional[str] = None,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """Synthesize a shop scene with purchasable drinks and cards."""
        active_plan = (plan or self.plan).lower()
        img = self._create_background(scene_type="shop")
        draw = ImageDraw.Draw(img)

        # Title
        draw.rectangle([100, 150, 980, 270], fill=(60, 45, 30), outline=(180, 140, 70), width=3)

        if drinks is None:
            available = [d for d in self.db.get_drinks_for_plan(active_plan) if d.get("icon_path") and Path(d["icon_path"]).exists()]
            drinks = available[:3] if len(available) >= 3 else []

        obs_options = []
        item_w = 260
        item_h = 320
        gap = 50
        x_start = (BASE_WIDTH - (len(drinks) * item_w + (len(drinks) - 1) * gap)) // 2
        y_pos = 600

        for i, drink_input in enumerate(drinks):
            drink_data = self.db.get_drink_by_id(drink_input) if isinstance(drink_input, int) else drink_input
            if not drink_data:
                continue

            x1 = x_start + i * (item_w + gap)
            y1 = y_pos
            bbox = BoundingBox(xmin=x1, ymin=y1, xmax=x1 + item_w, ymax=y1 + item_h)
            draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], fill=(45, 38, 50), outline=(120, 100, 140), width=2)

            icon_path = drink_data.get("icon_path")
            if icon_path and Path(icon_path).exists():
                try:
                    icon_img = Image.open(icon_path).convert("RGB").resize((120, 120))
                    img.paste(icon_img, (bbox.xmin + (item_w - 120) // 2, bbox.ymin + 40))
                except Exception:
                    pass

            obs_options.append({
                "choice_id": i,
                "drink_id": drink_data["id"],
                "name": drink_data.get("name") or "",
                "price": 120,
                "bbox": bbox.model_dump(),
            })

        frame_arr = np.array(img, dtype=np.uint8)
        if _HAS_CV2:
            cv2.putText(frame_arr, "ITEM SHOP", (420, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 235, 140), 2)
            for i in range(len(obs_options)):
                x1 = x_start + i * (item_w + gap)
                cv2.putText(frame_arr, "120 P", (x1 + 80, y_pos + item_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 50), 2)

        obs = ScreenObservation(
            scene_type="shop",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=active_plan,
            options=obs_options,
        )
        return frame_arr, obs

    def generate_popup_frame(
        self,
        message: str = "Confirm Action?",
        plan: Optional[str] = None,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """Synthesize a modal popup dialog over dimmed screen."""
        active_plan = (plan or self.plan).lower()
        img = self._create_background(scene_type="popup")
        # Apply dark overlay for modal backdrop
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.40)
        draw = ImageDraw.Draw(img)

        # Dialog box in center
        d_box = BoundingBox(xmin=140, ymin=650, xmax=940, ymax=1150)
        draw.rectangle([d_box.xmin, d_box.ymin, d_box.xmax, d_box.ymax], fill=(55, 48, 70), outline=(170, 150, 210), width=4)

        # Buttons
        btn_confirm = BoundingBox(xmin=560, ymin=1000, xmax=880, ymax=1100)
        btn_cancel = BoundingBox(xmin=200, ymin=1000, xmax=520, ymax=1100)
        draw.rectangle([btn_confirm.xmin, btn_confirm.ymin, btn_confirm.xmax, btn_confirm.ymax], fill=(60, 160, 90), outline=(120, 220, 150), width=2)
        draw.rectangle([btn_cancel.xmin, btn_cancel.ymin, btn_cancel.xmax, btn_cancel.ymax], fill=(160, 60, 70), outline=(220, 110, 120), width=2)

        options = [
            {"choice_id": 0, "name": "CANCEL", "action": "cancel", "bbox": btn_cancel.model_dump()},
            {"choice_id": 1, "name": "CONFIRM", "action": "confirm", "bbox": btn_confirm.model_dump()},
        ]

        frame_arr = np.array(img, dtype=np.uint8)
        if _HAS_CV2:
            cv2.putText(frame_arr, message, (220, 840), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame_arr, "CANCEL", (btn_cancel.xmin + 85, btn_cancel.ymin + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame_arr, "CONFIRM", (btn_confirm.xmin + 75, btn_confirm.ymin + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        obs = ScreenObservation(
            scene_type="popup",
            turn=1,
            turns_remaining=10,
            stamina=30,
            max_stamina=30,
            genki=0,
            plan=active_plan,
            options=options,
        )
        return frame_arr, obs

    def generate_frame(
        self,
        scene_type: str = "battle_turn",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """General entry point to generate any game scene frame."""
        if "week" in kwargs and kwargs.get("week") is not None and scene_type == "week_flow":
            return self.generate_week_frame(**kwargs)
        if scene_type == "battle_turn":
            return self.generate_battle_frame(**kwargs)
        elif scene_type == "produce_schedule":
            return self.generate_produce_schedule_frame(**kwargs)
        elif scene_type == "card_reward":
            return self.generate_card_reward_frame(**kwargs)
        elif scene_type == "shop":
            return self.generate_shop_frame(**kwargs)
        elif scene_type == "popup":
            return self.generate_popup_frame(**kwargs)
        else:
            return self.generate_battle_frame(**kwargs)

    def generate_week_frame(
        self,
        week: int = 1,
        scenario_id: str = "hajime_regular",
        plan: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, ScreenObservation]:
        """
        Synthesize a game frame and ScreenObservation corresponding to a specific
        week in a produce scenario (e.g., Week 1..13 of hajime_regular).
        """
        from demo.core.scenarios import compute_week_stats_and_stage_bonus

        active_plan = (plan or self.plan).lower()
        ctx = compute_week_stats_and_stage_bonus(
            scenario_id=scenario_id,
            week=week,
            plan=active_plan,
            override_stats=kwargs.pop("idol_stats", None),
        )
        node = ctx["week_node"]
        scene_type = kwargs.pop("scene_type", None) or node.scene_type

        if scene_type == "produce_schedule":
            ascii_labels = ["Option A: Primary", "Option B: Secondary", "Option C: Sub/Rest"]
            opts = kwargs.pop("options", None) or [
                {
                    "choice_id": i,
                    "name": ascii_labels[i] if i < len(ascii_labels) else f"Option {i + 1}",
                    "title": act,
                    "description": node.operation_guide,
                    "type": "schedule",
                }
                for i, act in enumerate(node.activities)
            ]
            frame_arr, obs = self.generate_produce_schedule_frame(options=opts, plan=active_plan)
            for i, act in enumerate(node.activities):
                if i < len(obs.options):
                    obs.options[i]["title"] = act
                    obs.options[i]["name"] = act
                    obs.options[i]["description"] = node.operation_guide
        elif scene_type == "shop":
            frame_arr, obs = self.generate_shop_frame(plan=active_plan, **kwargs)
        elif scene_type == "card_reward":
            frame_arr, obs = self.generate_card_reward_frame(plan=active_plan, **kwargs)
        elif scene_type == "popup":
            frame_arr, obs = self.generate_popup_frame(plan=active_plan, **kwargs)
        else:
            default_buffs = {
                "sense": {"goodConditionTurns": 4, "concentration": 3},
                "logic": {"goodImpressionTurns": 5, "motivation": 4},
                "anomaly": {"fullPowerCharge": 6, "strong_spirit": 2},
            }.get(active_plan, {"goodConditionTurns": 3})
            b_kwargs: Dict[str, Any] = {
                "stamina": 26,
                "max_stamina": 30,
                "genki": 6,
                "turn": 1,
                "turns_remaining": node.battle_turns,
                "score": 0,
                "target_score": node.target_score or 1200,
                "buffs": default_buffs,
            }
            b_kwargs.update(kwargs)
            frame_arr, obs = self.generate_battle_frame(plan=active_plan, **b_kwargs)

        if _HAS_CV2:
            cv2.putText(
                frame_arr,
                f"WEEK {node.week:02d}/{ctx['total_weeks']:02d} [{scenario_id.upper()}]",
                (40, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (130, 220, 255),
                2,
            )
            w_box = STATUS_WEEK_BBOX
            cv2.rectangle(frame_arr, (w_box.xmin, w_box.ymin), (w_box.xmax, w_box.ymax), (30, 26, 48), -1)
            week_str = f"{node.week}/{ctx['total_weeks']}"
            for i, ch in enumerate(week_str):
                cv2.putText(
                    frame_arr,
                    ch,
                    (w_box.xmin + 15 + i * 22, w_box.ymin + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

        obs.scenario_id = ctx["scenario_id"]
        obs.week = node.week
        obs.total_weeks = ctx["total_weeks"]
        obs.week_activity = node.title
        obs.idol_stats = ctx["idol_stats"]
        obs.p_items = ctx["p_items"]
        obs.stage_bonuses = ctx["stage_bonuses"]
        return frame_arr, obs
