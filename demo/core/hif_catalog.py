"""
H.I.F Memory Ability Catalog, Produce Preferences Schema, and Plan Metadata.

Stores:
1. ProducePreferences: user-configurable card/drink priorities, bans, and produce rules.
2. HIF_MEMORY_ABILITY_TEMPLATES & HIF_MEMORY_CARD_PRESETS_BY_PLAN: authoritative H.I.F
   card-bound memory abilities (from wikiwiki.jp/gakumas_cont/HIF/メモリーアビリティ).
3. MEMORY_PASSIVE_TEMPLATES: common passive memory traits for the 4-slot manual editor.
4. build_hif_memory_catalog(db): resolves preset card names to DB card IDs & assets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


PLAN_METADATA: Dict[str, Dict[str, str]] = {
    "sense": {
        "id": "sense",
        "label_ja": "センス",
        "label_zh": "感性 Sense",
        "axis": "好調・集中",
        "accent": "sense",
        "description": "以『好調』『絶好調』『集中』为核心增益，搭配高倍率打牌连携。",
    },
    "logic": {
        "id": "logic",
        "label_ja": "ロジック",
        "label_zh": "理性 Logic",
        "axis": "好印象・やる気",
        "accent": "logic",
        "description": "以『好印象』回合追加结算或『やる気/元気』护盾转换爆发为核心。",
    },
    "anomaly": {
        "id": "anomaly",
        "label_ja": "アノマリー",
        "label_zh": "异常 Anomaly",
        "axis": "強気・温存・全力",
        "accent": "anomaly",
        "description": "围绕指針切换（強気/温存）与積攢『全力値』触发全力爆发回合。",
    },
}


@dataclass
class ProducePreferences:
    """User-configurable produce & drink selection preferences."""

    priority_card_ids: List[int] = field(default_factory=list)
    avoid_card_ids: List[int] = field(default_factory=list)
    keep_card_ids: List[int] = field(default_factory=list)
    card_score_overrides: Dict[int, int] = field(default_factory=dict)
    preferred_drink_ids: List[int] = field(default_factory=list)
    banned_drink_ids: List[int] = field(default_factory=lambda: [19])
    drink_score_overrides: Dict[int, int] = field(default_factory=dict)
    spend_all_final_shop: bool = True
    reroll_early_non_exhaust_mental: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority_card_ids": [int(x) for x in self.priority_card_ids],
            "avoid_card_ids": [int(x) for x in self.avoid_card_ids],
            "keep_card_ids": [int(x) for x in self.keep_card_ids],
            "card_score_overrides": {str(int(k)): int(v) for k, v in self.card_score_overrides.items()},
            "preferred_drink_ids": [int(x) for x in self.preferred_drink_ids],
            "banned_drink_ids": [int(x) for x in self.banned_drink_ids],
            "drink_score_overrides": {str(int(k)): int(v) for k, v in self.drink_score_overrides.items()},
            "spend_all_final_shop": bool(self.spend_all_final_shop),
            "reroll_early_non_exhaust_mental": bool(self.reroll_early_non_exhaust_mental),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ProducePreferences":
        if not isinstance(raw, dict):
            return cls()

        def _uniq_ints(vals: Optional[Iterable[Any]]) -> List[int]:
            out: List[int] = []
            seen: set[int] = set()
            for v in vals or ():
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if iv > 0 and iv not in seen:
                    seen.add(iv)
                    out.append(iv)
            return out

        def _clamp_overrides(mapping: Optional[Dict[Any, Any]]) -> Dict[int, int]:
            out: Dict[int, int] = {}
            if not isinstance(mapping, dict):
                return out
            for k, v in mapping.items():
                try:
                    ik = int(k)
                    iv = max(-8000, min(8000, int(v)))
                except (TypeError, ValueError):
                    continue
                if ik > 0 and iv != 0:
                    out[ik] = iv
            return out

        if "banned_drink_ids" in raw and raw["banned_drink_ids"] is not None:
            banned_drinks = _uniq_ints(raw.get("banned_drink_ids"))
        else:
            banned_drinks = [19]

        return cls(
            priority_card_ids=_uniq_ints(raw.get("priority_card_ids")),
            avoid_card_ids=_uniq_ints(raw.get("avoid_card_ids")),
            keep_card_ids=_uniq_ints(raw.get("keep_card_ids")),
            card_score_overrides=_clamp_overrides(raw.get("card_score_overrides")),
            preferred_drink_ids=_uniq_ints(raw.get("preferred_drink_ids")),
            banned_drink_ids=banned_drinks,
            drink_score_overrides=_clamp_overrides(raw.get("drink_score_overrides")),
            spend_all_final_shop=bool(raw.get("spend_all_final_shop", True)),
            reroll_early_non_exhaust_mental=bool(raw.get("reroll_early_non_exhaust_mental", True)),
        )


HIF_MEMORY_ABILITY_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "force_initial_hand": {
        "key": "force_initial_hand",
        "label_ja": "開始時手札に入る",
        "label_zh": "开局必入初始手牌",
        "category": "opening",
        "hooks": {"force_initial_hand": True},
    },
    "exile_trouble_1": {
        "key": "exile_trouble_1",
        "label_ja": "試験ごとに1回まで、指定スキル使用後、山札/捨札の「眠気」を除外",
        "label_zh": "每场1次：使用后将山札/弃牌中1张《眠気》移入除外",
        "category": "exile_trouble",
        "hooks": {"exiles_trouble_on_play": True},
    },
    "extra_use_draw_1": {
        "key": "extra_use_draw_1",
        "label_ja": "試験ごとに1回まで、指定スキル使用後、スキルカード使用数追加+1・スキルカードを引く",
        "label_zh": "每场1次：使用后追加使用数+1并抽1张牌",
        "category": "extra_use_draw",
        "hooks": {"free_extra_use_first_play": True, "extra_use_limit": 1, "draw_on_play": 1},
    },
    "extra_use_1": {
        "key": "extra_use_1",
        "label_ja": "試験ごとに1回まで、指定スキル使用後、スキルカード使用数追加+1",
        "label_zh": "每场1次：使用后追加使用数+1（不耗行动点）",
        "category": "extra_use_1",
        "hooks": {"free_extra_use_first_play": True, "extra_use_limit": 1},
    },
    "extra_use_2": {
        "key": "extra_use_2",
        "label_ja": "試験ごとに2回まで、指定スキル使用後、スキルカード使用数追加+1",
        "label_zh": "每场2次：使用后追加使用数+1（不耗行动点）",
        "category": "extra_use_2",
        "hooks": {"free_extra_use_first_play": True, "extra_use_limit": 2},
    },
}


# Verified against wikiwiki.jp/gakumas_cont/HIF/メモリーアビリティ & gakumas_library.db
HIF_MEMORY_CARD_PRESETS_BY_PLAN: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "sense": {
        "exile_trouble_1": (
            "ひと呼吸",
            "ペース配分",
        ),
        "extra_use_draw_1": (
            "ハイタッチ",
            "トークタイム",
            "軌道修正",
            "パンプアップ",
            "エキサイト",
        ),
        "extra_use_1": (
            "スタートダッシュ",
            "スタンドプレー",
            "シュプレヒコール",
            "破竹の勢い",
            "演出計画",
            "願いの力",
            "静かな意志",
            "存在感",
            "スポットライト",
            "精神統一",
        ),
        "extra_use_2": (
            "決めポーズ",
            "飛躍",
            "祝福",
            "立ち位置チェック",
            "眼力",
            "大声援",
            "始まりの合図",
            "意地",
            "成功への道筋",
            "再始動",
            "衝動",
            "軽い足取り",
            "愛嬌",
            "準備運動",
            "ファンサ",
            "勢い任せ",
            "バランス感覚",
            "楽観的",
            "深呼吸",
        ),
    },
    "logic": {
        "exile_trouble_1": (
            "デイドリーミング",
            "止められない想い",
            "オトメゴコロ",
            "ワクワクが止まらない",
        ),
        "extra_use_draw_1": (
            "キラメキ",
            "みんな大好き",
            "ファンシーチャーム",
            "ワクワクが止まらない",
            "本番前夜",
            "ひなたぼっこ",
            "ゆめみごこち",
            "止められない想い",
            "オトメゴコロ",
            "ゆるふわおしゃべり",
            "手拍子",
            "元気な挨拶",
            "おまもりミラクル",
            "がむしゃら",
            "リズミカル",
            "幸せのおまじない",
            "イメチェン",
            "メイクアップ",
            "ロマンチック",
            "ありがとうの言葉",
            "ハートの合図",
            "イメトレ",
            "やる気は満点",
            "天使の微笑み",
        ),
        "extra_use_1": (
            "やる気は満点",
        ),
        "extra_use_2": (
            "今日もおはよう",
            "もう少しだけ",
            "リスタート",
            "えいえいおー",
            "思い出し笑い",
            "パステル気分",
            "励まし",
            "ラブリーウインク",
            "きらきら紙吹雪",
            "あふれる思い出",
            "ふれあい",
            "幸せな時間",
        ),
    },
    "anomaly": {
        "exile_trouble_1": (
            "形勢逆転",
            "セッティング",
        ),
        "extra_use_draw_1": (
            "ジャストアピール",
            "一歩",
            "ラッキー♪",
            "精一杯",
            "ノンストップ",
            "ハッスル",
            "ハッピー♪",
            "嬉しい誤算",
            "涙の思い出",
            "巻き返し",
        ),
    },
}


MEMORY_PASSIVE_TEMPLATES: List[Dict[str, Any]] = [
    {
        "key": "start_good_impression",
        "label": "試験開始時 好印象+X",
        "unit": "層",
        "default_value": 2,
        "plans": ["logic", "all"],
        "description": "进入考试/本戦舞台时初始好印象数值加成",
    },
    {
        "key": "start_motivation",
        "label": "試験開始時 やる気+X",
        "unit": "点",
        "default_value": 2,
        "plans": ["logic", "all"],
        "description": "进入考试/本戦舞台时初始干劲数值加成",
    },
    {
        "key": "start_genki",
        "label": "試験開始時 元気+X",
        "unit": "点",
        "default_value": 4,
        "plans": ["sense", "logic", "anomaly", "all"],
        "description": "进入考试/本戦舞台时初始元气(护盾)加成",
    },
    {
        "key": "start_good_condition",
        "label": "試験開始時 好調+Xターン",
        "unit": "T",
        "default_value": 2,
        "plans": ["sense", "all"],
        "description": "进入考试/本戦舞台时初始好调回合数加成",
    },
    {
        "key": "start_concentration",
        "label": "試験開始時 集中+X",
        "unit": "点",
        "default_value": 2,
        "plans": ["sense", "all"],
        "description": "进入考试/本戦舞台时初始集中加成",
    },
    {
        "key": "start_full_power_charge",
        "label": "試験開始時 全力値+X",
        "unit": "点",
        "default_value": 2,
        "plans": ["anomaly", "all"],
        "description": "进入考试/本戦舞台时初始全力值加成",
    },
    {
        "key": "initial_p_points",
        "label": "プロデュース開始時 Pポイント+X",
        "unit": "P",
        "default_value": 50,
        "plans": ["sense", "logic", "anomaly", "all"],
        "description": "培育开始时追加初期P点数（用于商店/咨询）",
    },
    {
        "key": "param_bonus_pct",
        "label": "パラメータ上昇量 +%（参考）",
        "unit": "%",
        "default_value": 3,
        "plans": ["sense", "logic", "anomaly", "all"],
        "description": "通常回忆词条：Vo/Da/Vi 属性成长百分比加成",
    },
    {
        "key": "custom_note",
        "label": "自定义回忆词条备注",
        "unit": "",
        "default_value": 0,
        "plans": ["sense", "logic", "anomaly", "all"],
        "description": "记录其余育成词条或特化回忆说明",
    },
]


HIF_MEMORY_RULE_NOTE = (
    "H.I.F 本戦攻略机制说明（来源: wikiwiki.jp/gakumas_cont/HIF/メモリーアビリティ）："
    "① H.I.F 本戦获得的回忆将固定附带 1 项『指定技能使用后追加效果』的 H.I.F 专属アビリティ；"
    "② 所有 SSR（虹）技能卡及部分自带眠气的开关型 SR（金）卡在游戏内未实装 H.I.F 专属アビリティ；"
    "③ 若同时编入多个同名称アビリティ，游戏内不会重复叠加触发，请按实际回忆截图在此勾选录入。"
)


def _index_plan_cards_by_name(db: Any, plan: str) -> Dict[str, List[Dict[str, Any]]]:
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for row in db.get_cards_for_plan(plan):
        nm = str(row.get("name") or "").strip()
        if not nm:
            continue
        base_nm = nm[:-1] if nm.endswith("+") else nm
        by_name.setdefault(base_nm, []).append(row)
    for bucket in by_name.values():
        bucket.sort(key=lambda r: (0 if not r.get("upgraded") else 1, int(r.get("id") or 0)))
    return by_name


def build_hif_memory_catalog(db: Any) -> Dict[str, Any]:
    """Resolve H.I.F memory presets against DB cards into a structured JSON catalog."""
    plans_out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for plan in ("sense", "logic", "anomaly"):
        by_name = _index_plan_cards_by_name(db, plan)
        groups_out: Dict[str, List[Dict[str, Any]]] = {}
        for tmpl_key, card_names in HIF_MEMORY_CARD_PRESETS_BY_PLAN.get(plan, {}).items():
            tmpl = HIF_MEMORY_ABILITY_TEMPLATES[tmpl_key]
            entries: List[Dict[str, Any]] = []
            for base_nm in card_names:
                matches = by_name.get(base_nm, [])
                if not matches:
                    continue
                unup = next((r for r in matches if not r.get("upgraded")), matches[0])
                up = next((r for r in matches if r.get("upgraded")), unup)
                entries.append({
                    "name": base_nm,
                    "upgraded_name": str(up.get("name") or f"{base_nm}+"),
                    "card_id": int(unup["id"]),
                    "upgraded_card_id": int(up["id"]),
                    "base_id": int(unup.get("base_id") or unup["id"]),
                    "rarity": str(unup.get("rarity") or ""),
                    "type": str(unup.get("type") or ""),
                    "icon_path": unup.get("icon_path"),
                    "upgraded_icon_path": up.get("icon_path"),
                    "template_key": tmpl_key,
                    "template_label_ja": tmpl["label_ja"],
                    "template_label_zh": tmpl["label_zh"],
                    "hooks": dict(tmpl["hooks"]),
                })
            groups_out[tmpl_key] = entries
        plans_out[plan] = groups_out

    return {
        "rule_note": HIF_MEMORY_RULE_NOTE,
        "templates": HIF_MEMORY_ABILITY_TEMPLATES,
        "passive_templates": MEMORY_PASSIVE_TEMPLATES,
        "plans": plans_out,
    }
