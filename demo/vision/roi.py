"""
Region of Interest (ROI) extraction and coordinate management for Gakumas frames.
Standard portrait resolution: 1080x1920.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from demo.schemas import BoundingBox

# Standard reference resolution (portrait)
BASE_WIDTH = 1080
BASE_HEIGHT = 1920

# Standard hand region (bottom of screen)
HAND_Y_MIN = 1380
HAND_Y_MAX = 1740

# Standard drink region (upper HUD)
DRINK_Y_MIN = 220
DRINK_Y_MAX = 340
DRINK_SLOT_WIDTH = 120
DRINK_SLOT_GAP = 20
DRINK_X_START = 660

# Standard status HUD bounding boxes (1080x1920)
STATUS_WEEK_BBOX = BoundingBox(xmin=740, ymin=20, xmax=1040, ymax=90)
STATUS_TURN_BBOX = BoundingBox(xmin=40, ymin=140, xmax=340, ymax=210)
STATUS_STAMINA_BBOX = BoundingBox(xmin=40, ymin=220, xmax=440, ymax=290)
STATUS_GENKI_BBOX = BoundingBox(xmin=40, ymin=300, xmax=340, ymax=360)
STATUS_SCORE_BBOX = BoundingBox(xmin=40, ymin=370, xmax=440, ymax=430)
STATUS_BUFFS_BBOX = BoundingBox(xmin=40, ymin=440, xmax=1040, ymax=520)
END_TURN_BBOX = BoundingBox(xmin=860, ymin=1760, xmax=1040, ymax=1880)


def scale_bbox(
    bbox: BoundingBox,
    from_size: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
    to_size: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
) -> BoundingBox:
    """Scale a bounding box between arbitrary screen resolutions."""
    from_w, from_h = from_size
    to_w, to_h = to_size
    if from_w == to_w and from_h == to_h:
        return bbox

    sx = to_w / float(from_w)
    sy = to_h / float(from_h)
    return BoundingBox(
        xmin=int(round(bbox.xmin * sx)),
        ymin=int(round(bbox.ymin * sy)),
        xmax=int(round(bbox.xmax * sx)),
        ymax=int(round(bbox.ymax * sy)),
    )


def to_normalized(
    bbox: BoundingBox,
    frame_size: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
) -> Tuple[float, float, float, float]:
    """Convert absolute bounding box to normalized coordinates in [0.0, 1.0]."""
    w, h = frame_size
    return (
        bbox.xmin / float(w),
        bbox.ymin / float(h),
        bbox.xmax / float(w),
        bbox.ymax / float(h),
    )


def from_normalized(
    norm_box: Tuple[float, float, float, float],
    frame_size: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
) -> BoundingBox:
    """Convert normalized coordinates [0.0, 1.0] to absolute BoundingBox."""
    w, h = frame_size
    n_xmin, n_ymin, n_xmax, n_ymax = norm_box
    return BoundingBox(
        xmin=int(round(n_xmin * w)),
        ymin=int(round(n_ymin * h)),
        xmax=int(round(n_xmax * w)),
        ymax=int(round(n_ymax * h)),
    )


def crop_roi(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Safely crop image array using bounding box coordinates."""
    h, w = frame.shape[:2]
    x1 = max(0, min(bbox.xmin, w))
    x2 = max(0, min(bbox.xmax, w))
    y1 = max(0, min(bbox.ymin, h))
    y2 = max(0, min(bbox.ymax, h))
    if x1 >= x2 or y1 >= y2:
        return np.zeros((0, 0, 3 if frame.ndim == 3 else 1), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def get_hand_bboxes(
    count: int = 3,
    frame_size: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
) -> List[BoundingBox]:
    """
    Calculate bounding boxes for hand cards (1 to 5 cards) horizontally distributed
    across the bottom region of the screen.
    """
    count = max(1, min(count, 5))
    w, h = frame_size

    # Base coordinates in 1080x1920
    content_left = 50
    content_right = 1030
    total_span = content_right - content_left

    if count == 1:
        card_w = 280
        gap = 0
        left = (BASE_WIDTH - card_w) // 2
        base_boxes = [BoundingBox(xmin=left, ymin=HAND_Y_MIN, xmax=left + card_w, ymax=HAND_Y_MAX)]
    elif count == 2:
        card_w = 280
        gap = 120
        total_w = 2 * card_w + gap
        left = (BASE_WIDTH - total_w) // 2
        base_boxes = [
            BoundingBox(xmin=left, ymin=HAND_Y_MIN, xmax=left + card_w, ymax=HAND_Y_MAX),
            BoundingBox(xmin=left + card_w + gap, ymin=HAND_Y_MIN, xmax=left + 2 * card_w + gap, ymax=HAND_Y_MAX),
        ]
    elif count == 3:
        card_w = 260
        gap = (total_span - 3 * card_w) // 2
        base_boxes = [
            BoundingBox(xmin=content_left + i * (card_w + gap), ymin=HAND_Y_MIN, xmax=content_left + i * (card_w + gap) + card_w, ymax=HAND_Y_MAX)
            for i in range(3)
        ]
    elif count == 4:
        card_w = 210
        gap = (total_span - 4 * card_w) // 3
        base_boxes = [
            BoundingBox(xmin=content_left + i * (card_w + gap), ymin=HAND_Y_MIN, xmax=content_left + i * (card_w + gap) + card_w, ymax=HAND_Y_MAX)
            for i in range(4)
        ]
    else:  # count == 5
        card_w = 175
        gap = (total_span - 5 * card_w) // 4
        base_boxes = [
            BoundingBox(xmin=content_left + i * (card_w + gap), ymin=HAND_Y_MIN, xmax=content_left + i * (card_w + gap) + card_w, ymax=HAND_Y_MAX)
            for i in range(5)
        ]

    if (w, h) == (BASE_WIDTH, BASE_HEIGHT):
        return base_boxes

    return [scale_bbox(b, (BASE_WIDTH, BASE_HEIGHT), (w, h)) for b in base_boxes]


def get_drink_bboxes(
    count: int = 3,
    frame_size: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
) -> List[BoundingBox]:
    """
    Calculate bounding boxes for up to 3 drink slots in HUD area.
    """
    count = max(1, min(count, 3))
    w, h = frame_size

    base_boxes = []
    for i in range(count):
        x1 = DRINK_X_START + i * (DRINK_SLOT_WIDTH + DRINK_SLOT_GAP)
        x2 = x1 + DRINK_SLOT_WIDTH
        base_boxes.append(BoundingBox(xmin=x1, ymin=DRINK_Y_MIN, xmax=x2, ymax=DRINK_Y_MAX))

    if (w, h) == (BASE_WIDTH, BASE_HEIGHT):
        return base_boxes

    return [scale_bbox(b, (BASE_WIDTH, BASE_HEIGHT), (w, h)) for b in base_boxes]


def get_status_bboxes(
    frame_size: Tuple[int, int] = (BASE_WIDTH, BASE_HEIGHT),
) -> Dict[str, BoundingBox]:
    """Get all standard status HUD bounding boxes scaled to given frame size."""
    w, h = frame_size
    base_dict = {
        "week": STATUS_WEEK_BBOX,
        "turn": STATUS_TURN_BBOX,
        "stamina": STATUS_STAMINA_BBOX,
        "genki": STATUS_GENKI_BBOX,
        "score": STATUS_SCORE_BBOX,
        "buffs": STATUS_BUFFS_BBOX,
        "end_turn": END_TURN_BBOX,
    }
    if (w, h) == (BASE_WIDTH, BASE_HEIGHT):
        return base_dict

    return {k: scale_bbox(v, (BASE_WIDTH, BASE_HEIGHT), (w, h)) for k, v in base_dict.items()}


def extract_hand_rois(
    frame: np.ndarray,
    count: int = 3,
) -> List[Tuple[BoundingBox, np.ndarray]]:
    """Extract hand card bounding boxes and image crops from frame."""
    h, w = frame.shape[:2]
    bboxes = get_hand_bboxes(count=count, frame_size=(w, h))
    return [(bbox, crop_roi(frame, bbox)) for bbox in bboxes]


def extract_drink_rois(
    frame: np.ndarray,
    count: int = 3,
) -> List[Tuple[BoundingBox, np.ndarray]]:
    """Extract drink slot bounding boxes and image crops from frame."""
    h, w = frame.shape[:2]
    bboxes = get_drink_bboxes(count=count, frame_size=(w, h))
    return [(bbox, crop_roi(frame, bbox)) for bbox in bboxes]


def extract_status_rois(
    frame: np.ndarray,
) -> Dict[str, Tuple[BoundingBox, np.ndarray]]:
    """Extract status HUD bounding boxes and image crops from frame."""
    h, w = frame.shape[:2]
    status_bboxes = get_status_bboxes(frame_size=(w, h))
    return {k: (bbox, crop_roi(frame, bbox)) for k, bbox in status_bboxes.items()}


class StatusOCRExtractor:
    """
    Lightweight, robust OCR extractor for status indicators (HP, Genki, Turn, Buffs).
    Supports both OpenCV and pure NumPy cross-correlation paths.
    """

    def __init__(self):
        self._templates: Dict[str, np.ndarray] = self._build_digit_templates()

    @staticmethod
    def _build_digit_templates() -> Dict[str, np.ndarray]:
        """Generate canonical 30x20 binary templates for digits 0-9, '/', 'T','U','R','N'."""
        templates: Dict[str, np.ndarray] = {}
        chars = "0123456789/TURN"
        for ch in chars:
            raw = np.zeros((30, 30), dtype=np.uint8)
            if _HAS_CV2:
                cv2.putText(raw, ch, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            else:
                # Fallback simple block digit renderer if cv2 is not present
                raw[6:24, 8:22] = 255

            # Find horizontal extent and center in 30x20 cell
            col_sum = (raw > 0).sum(axis=0)
            xs = np.where(col_sum > 0)[0]
            cell = np.zeros((30, 20), dtype=np.uint8)
            if len(xs) > 0:
                x1, x2 = xs.min(), xs.max() + 1
                w = x2 - x1
                px = max(0, (20 - w) // 2)
                cell[:, px:px + min(w, 20 - px)] = raw[:, x1:x1 + min(w, 20 - px)]
            templates[ch] = cell
        return templates

    def _match_cell(self, cell: np.ndarray, tmpl: np.ndarray) -> float:
        """Compute normalized cross-correlation between cell and template."""
        if _HAS_CV2:
            return float(cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)[0][0])
        # Pure numpy NCC
        c = cell.astype(np.float32)
        t = tmpl.astype(np.float32)
        c_zero = c - c.mean()
        t_zero = t - t.mean()
        c_norm = np.linalg.norm(c_zero)
        t_norm = np.linalg.norm(t_zero)
        if c_norm < 1e-4 or t_norm < 1e-4:
            return 0.0
        return float((c_zero * t_zero).sum() / (c_norm * t_norm))

    def ocr_text(self, roi: np.ndarray, allowed_chars: str = "0123456789/") -> str:
        """Segment and recognize text characters in a status ROI."""
        if roi.size == 0:
            return ""

        # Convert to grayscale
        if roi.ndim == 3:
            if _HAS_CV2:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY if roi.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
            else:
                gray = (0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]).astype(np.uint8)
        else:
            gray = roi

        # Threshold to binary (bright text on darker background)
        thresh = (gray > 160).astype(np.uint8) * 255

        # Column segmentation
        col_sum = (thresh > 0).sum(axis=0)
        chars: List[Tuple[int, int]] = []
        in_char = False
        start = 0
        for x, val in enumerate(col_sum):
            if val > 0 and not in_char:
                in_char = True
                start = x
            elif val == 0 and in_char:
                in_char = False
                if x - start >= 2:
                    chars.append((start, x))
        if in_char and len(col_sum) - start >= 2:
            chars.append((start, len(col_sum)))

        result = ""
        for x1, x2 in chars:
            w = x2 - x1
            if w > 40:
                # Wide combined cluster, split if necessary or take best
                pass
            cell = np.zeros((30, 20), dtype=np.uint8)
            slice_col = thresh[:, x1:x2]
            # Height normalization / crop
            row_sum = (slice_col > 0).sum(axis=1)
            ys = np.where(row_sum > 0)[0]
            if len(ys) == 0:
                continue
            y1, y2 = ys.min(), ys.max() + 1
            char_crop = slice_col[y1:y2, :]
            ch_h, ch_w = char_crop.shape

            # Resize if dimensions exceed cell
            if ch_w > 18:
                if _HAS_CV2:
                    char_crop = cv2.resize(char_crop, (18, ch_h))
                else:
                    char_crop = char_crop[:, :18]
                ch_w = char_crop.shape[1]
            if ch_h > 26:
                if _HAS_CV2:
                    char_crop = cv2.resize(char_crop, (ch_w, 26))
                else:
                    char_crop = char_crop[:26, :]
                ch_h = char_crop.shape[0]

            py = max(0, (30 - ch_h) // 2)
            px = max(0, (20 - ch_w) // 2)
            cell[py:py + ch_h, px:px + ch_w] = char_crop

            best_ch = ""
            best_score = -1.0
            for ch in allowed_chars:
                if ch not in self._templates:
                    continue
                score = self._match_cell(cell, self._templates[ch])
                if score > best_score:
                    best_score = score
                    best_ch = ch

            if best_score > 0.40 and best_ch:
                result += best_ch

        return result

    def ocr_stamina(self, roi: np.ndarray) -> Tuple[int, int]:
        """Extract stamina and max stamina (e.g. from '30/30' or HP bar)."""
        if roi.ndim >= 2 and roi.shape[1] > 200:
            text_crop = roi[:, int(roi.shape[1] * 0.45):]
        else:
            text_crop = roi
        text = self.ocr_text(text_crop, allowed_chars="0123456789/")
        if "/" in text:
            parts = text.split("/")
            try:
                cur = int(parts[0])
                mx = int(parts[1])
                return cur, mx
            except ValueError:
                pass
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 2:
            mid = len(digits) // 2
            try:
                return int(digits[:mid]), int(digits[mid:])
            except ValueError:
                pass
        return 30, 30

    def ocr_genki(self, roi: np.ndarray) -> int:
        """Extract genki integer value."""
        if roi.ndim >= 2 and roi.shape[1] > 100:
            text_crop = roi[:, int(roi.shape[1] * 0.20):]
        else:
            text_crop = roi
        text = self.ocr_text(text_crop, allowed_chars="0123456789")
        try:
            return int(text)
        except ValueError:
            return 0

    def ocr_turn(self, roi: np.ndarray) -> Tuple[int, int]:
        """Extract current turn and turns remaining."""
        text = self.ocr_text(roi, allowed_chars="0123456789/")
        if "/" in text:
            parts = text.split("/")
            try:
                cur = int(parts[0])
                rem = int(parts[1])
                return cur, rem
            except ValueError:
                pass
        return 1, 10

    def ocr_week(self, roi: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Extract current produce week and total weeks (e.g. '6/13') from top header ROI."""
        text = self.ocr_text(roi, allowed_chars="0123456789/")
        if "/" in text:
            parts = text.split("/")
            try:
                w = int(parts[0])
                tot = int(parts[1])
                if 1 <= w <= 25 and 10 <= tot <= 25:
                    return w, tot
            except ValueError:
                pass
        return None, None

    def ocr_score(self, roi: np.ndarray) -> Tuple[int, int]:
        """Extract score and target score."""
        text = self.ocr_text(roi, allowed_chars="0123456789/")
        if "/" in text:
            parts = text.split("/")
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return 0, 5000

    def ocr_buffs(self, roi: np.ndarray) -> Dict[str, int]:
        """
        Detect active buffs and stack counts across the buff bar.
        Recognizes buff types by canonical badge color hue and digit stacks.
        """
        if roi.size == 0:
            return {}

        buffs: Dict[str, int] = {}
        # Predefined buff canonical colors (HSV Hue or RGB)
        # good_condition (orange: R>200, G in 100-180, B<50)
        # concentration (cyan/blue: R<50, G>150, B>200)
        # motivation (purple/pink: R>200, G<100, B>200)
        # good_impression (green: R<100, G>200, B<100)
        # strong_spirit (yellow: R>200, G>200, B<50)
        # full_effort (magenta/gold: R>220, G>150, B<50)
        # preserve (sky blue: R<100, G>180, B>220)

        # Segment horizontal buff badges (each badge ~60-80px width)
        h, w = roi.shape[:2]
        col_variance = roi.std(axis=(0, 2)) if roi.ndim == 3 else roi.std(axis=0)
        active_cols = np.where(col_variance > 10.0)[0]
        if len(active_cols) == 0:
            return {}

        # Group contiguous active regions
        regions: List[Tuple[int, int]] = []
        start = active_cols[0]
        for i in range(1, len(active_cols)):
            if active_cols[i] - active_cols[i - 1] > 20:
                regions.append((start, active_cols[i - 1]))
                start = active_cols[i]
        regions.append((start, active_cols[-1]))

        for x1, x2 in regions:
            patch = roi[:, x1:x2]
            if patch.shape[1] < 15:
                continue

            # Identify dominant badge color in upper portion
            badge_crop = patch[:max(10, h // 2), :]
            mean_rgb = badge_crop.mean(axis=(0, 1)) if badge_crop.ndim == 3 else np.array([128, 128, 128])
            r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]

            buff_name = "buff"
            if r > 180 and g > 180 and b < 80:
                buff_name = "strong_spirit"
            elif r > 180 and 80 <= g <= 170 and b < 80:
                buff_name = "good_condition"
            elif r < 100 and g > 170 and b < 100:
                buff_name = "good_impression"
            elif r < 80 and g > 130 and b > 180:
                buff_name = "concentration"
            elif r > 180 and g < 100 and b > 180:
                buff_name = "motivation"
            elif r > 180 and g > 130 and b < 80:
                buff_name = "full_effort"
            elif r < 120 and g > 160 and b > 200:
                buff_name = "preserve"
            else:
                buff_name = "good_condition"

            # Extract stack number from text portion
            stack_text = self.ocr_text(patch, allowed_chars="0123456789")
            try:
                stack = int(stack_text) if stack_text else 1
            except ValueError:
                stack = 1
            buffs[buff_name] = stack

        return buffs

    def extract_status(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract all status numerical values from frame."""
        status_rois = extract_status_rois(frame)
        week, total_weeks = self.ocr_week(status_rois["week"][1])
        turn, turns_rem = self.ocr_turn(status_rois["turn"][1])
        stamina, max_stamina = self.ocr_stamina(status_rois["stamina"][1])
        genki = self.ocr_genki(status_rois["genki"][1])
        score, target_score = self.ocr_score(status_rois["score"][1])
        buffs = self.ocr_buffs(status_rois["buffs"][1])

        return {
            "week": week,
            "total_weeks": total_weeks,
            "turn": turn,
            "turns_remaining": turns_rem,
            "stamina": stamina,
            "max_stamina": max_stamina,
            "genki": genki,
            "score": score,
            "target_score": target_score,
            "buffs": buffs,
        }
