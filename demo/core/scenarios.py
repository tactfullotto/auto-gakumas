"""
Gakumas Produce Scenario & Weekly Schedule Registry.

Records complete week-by-week activity flows for all Gakumas scenarios:
- hajime_regular: 定期公演『初』·常规 (Regular, 13 Weeks, Stat Cap 1000)
- hajime_pro:     定期公演『初』·进阶 (Pro, 16 Weeks, Stat Cap 1500)
- hajime_master:  定期公演『初』·大师 (Master, 18 Weeks, Stat Cap 1800)
- nia_master:     『N.I.A』Next Idol Audition (18 Weeks, 3 Audition Stages)
- hif_honsen:     『H.I.F 本戦』Hatsuboshi IDOL FESTIVAL Main Tournament (6 Prep Days + Round 1 + Interval + Round 2)

Note: Mid-produce P-drink acquisition/selection modals are random/dynamic events
recognized solely from live UI visual/OCR features (never hardcoded to fixed weeks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class WeekScheduleNode:
    week: int
    phase_key: str
    title: str
    scene_type: str  # 'produce_schedule' | 'battle_turn' | 'card_reward' | 'shop' | 'popup'
    activities: List[str]
    recommended_action: str
    operation_guide: str
    target_score: int = 0
    battle_turns: int = 8
    follow_up_scene: Optional[str] = None  # e.g. lesson -> card_reward


@dataclass(frozen=True)
class ProduceScenarioSpec:
    scenario_id: str
    name: str
    short_name: str
    difficulty: str
    total_weeks: int
    stat_cap: int
    midterm_week: int
    final_week: int
    description: str
    weeks: Dict[int, WeekScheduleNode] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. 『初』· 常规剧本 (Hajime Regular, 13 Weeks) — 完整周目活动表
# ---------------------------------------------------------------------------

HAJIME_REGULAR_WEEKS: Dict[int, WeekScheduleNode] = {
    1: WeekScheduleNode(
        week=1,
        phase_key="week1_regular_lesson",
        title="第1周 · 基础通常课程 (通常レッスン)",
        scene_type="produce_schedule",
        activities=["Vo通常课程 (Vocal+35, 体力-6)", "Da通常课程 (Dance+35, 体力-6)", "Vi通常课程 (Visual+35, 体力-6)"],
        recommended_action="Vo通常课程",
        operation_guide="第1周固定不触发SP课程且不可休息；识别第1周日程后，直接选择偶像主属性对应的通常课程，打下基础三维并在课后三选一拿流派卡。",
        follow_up_scene="card_reward",
    ),
    2: WeekScheduleNode(
        week=2,
        phase_key="week2_class_or_lesson",
        title="第2周 · 授业课堂 / 通常课程 (授業 / レッスン)",
        scene_type="produce_schedule",
        activities=["课堂授业 (主属性+40 & 获得技能卡, 体力-8)", "Vo通常课程 (Vocal+40, 体力-7)", "活动支给 (P点数/动态随机事件)"],
        recommended_action="课堂授业",
        operation_guide="第2周开放课堂授业与活动支给；优先选择课堂授业同时提升三维与扩充早期过渡牌库（途中可能随机弹出P饮料选择，均以画面实机识别为准）。",
        follow_up_scene="card_reward",
    ),
    3: WeekScheduleNode(
        week=3,
        phase_key="week3_supply_or_outing",
        title="第3周 · 活动支给 / 外出 / 课程 (活動支給 / おでかけ)",
        scene_type="produce_schedule",
        activities=["活动支给 (P点数 & 随机补给/P饮料选择)", "外出おでかけ (体力+12 & 随机事件)", "Da通常课程 (Dance+45, 体力-8)"],
        recommended_action="活动支给",
        operation_guide="第3周可根据体力与活动面板选择课程、活动支给或外出；P饮料并非固定在某一周触发，任意周目出现选饮料界面时皆由图像实时识别并自主权衡。",
    ),
    4: WeekScheduleNode(
        week=4,
        phase_key="week4_pre_midterm_shop",
        title="第4周 · 中间考前整备：咨询商店 / 外出 (相談 / おでかけ)",
        scene_type="shop",
        activities=["购买流派核心技能卡 (100P)", "购买初星水/乌龙茶P饮料 (120P)", "强化手牌主力卡 (+) (80P)"],
        recommended_action="购买初星水/乌龙茶P饮料",
        operation_guide="第4周为中间审查前2周整备窗口；进入咨询商店优先补齐契合流派的核心卡/P饮料或强化关键卡，并保留足够体力迎接第5周冲刺课。",
    ),
    5: WeekScheduleNode(
        week=5,
        phase_key="week5_midterm_oikomi",
        title="第5周 · 中间冲刺课程 (中間追い込みレッスン)",
        scene_type="battle_turn",
        activities=["中间冲刺课程打牌考核 (目标600分 · 8回合)"],
        recommended_action="进入打牌阶段按流派叠Buff达标PERFECT",
        operation_guide="第5周固定为中间冲刺课程（打牌阶段）：前中期叠好流派状态（好调/集中、好印象/干劲、强气/全力），在回合结束前打满600分拿满主副属性奖励。",
        target_score=600,
        battle_turns=8,
    ),
    6: WeekScheduleNode(
        week=6,
        phase_key="week6_midterm_exam",
        title="第6周 · 中间审查 (中間試験 / Mid-term Audition)",
        scene_type="battle_turn",
        activities=["中间审查舞台打牌考核 (目标1200分 · 争第1名)"],
        recommended_action="利用P道具被动与流行加成夺取中间审查第1名",
        operation_guide="第6周固定为『初·常规』中间审查：必须拿到第1名以解锁后半程高阶奖励与P道具加成。前期铺垫核心Buff，尾局主属性高倍率回合打出高伤卡。",
        target_score=1200,
        battle_turns=8,
    ),
    7: WeekScheduleNode(
        week=7,
        phase_key="week7_post_midterm_recovery",
        title="第7周 · 中间考后休整：外出 / 活动支给 / 休息 (試験明け休養)",
        scene_type="produce_schedule",
        activities=["外出おでかけ (体力+15 & 强化事件)", "活动支给 (P点数+180 & 高阶P饮料)", "休息 (体力+20)"],
        recommended_action="外出おでかけ",
        operation_guide="第7周刚结束中间审查，体力通常偏低；优先选择外出或休息恢复体力至安全线(>=20)，备战第8~10周的高收益SP课程。",
    ),
    8: WeekScheduleNode(
        week=8,
        phase_key="week8_sp_lesson",
        title="第8周 · 后半程SP课程训练 (SPレッスン)",
        scene_type="produce_schedule",
        activities=["Vo SP课程 [闪光SP] (Vocal+90, 体力-10)", "Da通常课程 (Dance+50, 体力-8)", "休息 (体力+20)"],
        recommended_action="Vo SP课程 [闪光SP]",
        operation_guide="第8周进入后半程属性爆发期；只要体力>=15，必选闪光SP课程大幅拉升主属性，课后从三选一中挑选高稀有度流派卡。",
        follow_up_scene="card_reward",
    ),
    9: WeekScheduleNode(
        week=9,
        phase_key="week9_high_tier_card_reward",
        title="第9周 · 后半程高阶授业与选卡奖励 (後半授業・カード獲得)",
        scene_type="card_reward",
        activities=["流派SSR/SR限1核心卡", "流派强化+主动得分卡", "流派强化+辅助Buff卡"],
        recommended_action="优先选取契合流派的限1核心卡或+强化卡",
        operation_guide="第9周为高阶技能卡成型关键周：严格按照职业流派（感性好调集中/理性好印象干劲/异常强气全力）选取SR/SSR限1核心卡，严禁混抓异流派杂牌。",
    ),
    10: WeekScheduleNode(
        week=10,
        phase_key="week10_sp_lesson_push",
        title="第10周 · 后半程SP课程冲刺 (後半SPレッスン)",
        scene_type="produce_schedule",
        activities=["Da SP课程 [闪光SP] (Dance+95, 体力-11)", "Vo SP课程 [闪光SP] (Vocal+95, 体力-11)", "休息 (体力+20)"],
        recommended_action="Da SP课程 [闪光SP]",
        operation_guide="第10周为常规课程最后冲刺周；优先选择主/副属性SP课程把三维面板推向700+，提升最终审查的面板得分乘区。",
    ),
    11: WeekScheduleNode(
        week=11,
        phase_key="week11_pre_final_shop",
        title="第11周 · 终考前整备：最终咨询商店 / 外出 (最終試験2週前・相談)",
        scene_type="shop",
        activities=["购入特制高阶P饮料 (150P)", "强化终结爆发卡 (+) (100P)", "删除基础过渡废卡 (100P)"],
        recommended_action="购入特制高阶P饮料",
        operation_guide="第11周为最终审查前2周最后整备窗口：花光剩余P点数购入决战P饮料、强化主力爆发卡或删减废卡压缩牌库，确保第12~13周满状态参战。",
    ),
    12: WeekScheduleNode(
        week=12,
        phase_key="week12_final_oikomi",
        title="第12周 · 最终冲刺课程 (最終追い込みレッスン)",
        scene_type="battle_turn",
        activities=["最终冲刺课程打牌考核 (目标2500分 · 全三维大提升)"],
        recommended_action="稳健叠Buff达成2500分PERFECT，保留决战饮料至第13周",
        operation_guide="第12周固定为最终冲刺课程（打牌阶段）：利用成型卡组打出2500分拿满全三维冲刺加成，尽量保留高价值P饮料留给第13周最终审查。",
        target_score=2500,
        battle_turns=10,
    ),
    13: WeekScheduleNode(
        week=13,
        phase_key="week13_final_exam",
        title="第13周 · 最终审查 (最終試験 / Final Audition)",
        scene_type="battle_turn",
        activities=["定期公演『初』最终审查舞台 (目标5000+分 · 冲击A+/S评级)"],
        recommended_action="倾泻全部P饮料与限1终结爆发卡冲击最高分",
        operation_guide="第13周为『初·常规』终局公演审查（10回合）：第1回合与最终回合享主属性高额倍率加成；前中期迅速叠满流派核心Buff，后半程配合全部P饮料与限1大招打出极限爆发。",
        target_score=5000,
        battle_turns=10,
    ),
}


# ---------------------------------------------------------------------------
# 2. 『初』· 进阶 (Pro, 16周) / 大师 (Master, 18周) / 『N.I.A』 (18周) 剧本表
# ---------------------------------------------------------------------------

def _build_hajime_pro_weeks() -> Dict[int, WeekScheduleNode]:
    mapping = [
        (1, "produce_schedule", "第1周 · 基础通常课程", ["Vo通常课程", "Da通常课程", "Vi通常课程"], "第1周固定无SP课程，选主属性通常课程。", 0),
        (2, "produce_schedule", "第2周 · 授业 / 通常课程", ["课堂授业", "Vo通常课程", "活动支给"], "优先课堂授业提升属性并拿卡。", 0),
        (3, "produce_schedule", "第3周 · SP课程 / 通常课程", ["主属性SP课程", "副属性通常课程", "外出"], "开启SP课程，优先主属性SP。", 0),
        (4, "produce_schedule", "第4周 · 活动支给 / 外出", ["活动支给", "外出", "课堂授业"], "补充P点数与P饮料。", 0),
        (5, "produce_schedule", "第5周 · SP课程 / 授业", ["主属性SP课程", "副属性SP课程", "休息"], "中期拉升主副属性。", 0),
        (6, "shop", "第6周 · 中间考前咨询商店", ["购买流派核心卡", "购买P饮料", "强化卡牌"], "中间冲刺课前整备牌库与饮料。", 0),
        (7, "battle_turn", "第7周 · 中间冲刺课程 (中間追い込み)", ["中间冲刺打牌 (目标1100分)"], "叠Buff打满冲刺课PERFECT。", 1100),
        (8, "battle_turn", "第8周 · 中间审查 (中間試験)", ["中间审查舞台 (目标2200分)"], "夺取第1名并获得初星手链等P道具。", 2200),
        (9, "produce_schedule", "第9周 · 考后外出 / 休息", ["外出", "活动支给", "休息"], "恢复体力备战后半程。", 0),
        (10, "produce_schedule", "第10周 · 后半程SP课程", ["主属性SP课程", "副属性SP课程", "休息"], "主副属性2A专精训练。", 0),
        (11, "card_reward", "第11周 · 后半程高阶授业选卡", ["SSR限1核心卡", "SR+强化流派卡", "辅助过牌卡"], "补齐高阶流派核心卡。", 0),
        (12, "produce_schedule", "第12周 · 后半程SP课程", ["主属性SP课程", "副属性SP课程", "外出"], "冲刺1000+主属性面板。", 0),
        (13, "produce_schedule", "第13周 · 后半程SP课程 / 支给", ["主属性SP课程", "活动支给", "休息"], "保持高面板与P点储备。", 0),
        (14, "shop", "第14周 · 最终考前咨询商店", ["购入决战P饮料", "强化核心爆发卡", "删减基础废卡"], "压缩牌库、备齐3瓶高品质饮料。", 0),
        (15, "battle_turn", "第15周 · 最终冲刺课程 (最終追い込み)", ["最终冲刺打牌 (目标4500分)"], "拿满终盘全三维冲刺奖励。", 4500),
        (16, "battle_turn", "第16周 · 最终审查 (最終試験)", ["Pro最终审查舞台 (目标10000+分)"], "全饮料+限1大招冲击A+/S评级。", 10000),
    ]
    return {
        w: WeekScheduleNode(
            week=w,
            phase_key=f"pro_week_{w}",
            title=title,
            scene_type=st,
            activities=acts,
            recommended_action=acts[0],
            operation_guide=guide,
            target_score=target,
            battle_turns=10 if target > 0 else 8,
        )
        for w, st, title, acts, guide, target in mapping
    }


def _build_hajime_master_weeks() -> Dict[int, WeekScheduleNode]:
    mapping = [
        (1, "produce_schedule", "第1周 · 基础通常课程", ["Vo通常课程", "Da通常课程", "Vi通常课程"], "Master第1周无SP且不可休息，选主属性课。", 0),
        (2, "produce_schedule", "第2周 · 授业 / 活动支给", ["课堂授业", "活动支给", "通常课程"], "扩充卡组与初期属性。", 0),
        (3, "produce_schedule", "第3周 · SP课程开启", ["主属性SP课程", "副属性SP课程", "外出"], "优先闪光SP课程。", 0),
        (4, "produce_schedule", "第4周 · SP课程 / 授业", ["主属性SP课程", "课堂授业", "休息"], "稳步提升三维与牌库质量。", 0),
        (5, "produce_schedule", "第5周 · 外出 / 活动支给", ["外出", "活动支给", "通常课程"], "维持血线与补充P点数。", 0),
        (6, "produce_schedule", "第6周 · SP课程训练", ["主属性SP课程", "副属性SP课程", "休息"], "中盘SP属性累积。", 0),
        (7, "shop", "第7周 · 中间考前咨询商店", ["删减基础废卡", "强化流派核心卡", "购买P饮料"], "精简卡组迎战Master中间审查。", 0),
        (8, "battle_turn", "第8周 · 中间冲刺课程 (中間追い込み)", ["Master中间冲刺 (目标1800分)"], "高倍率打满冲刺课。", 1800),
        (9, "battle_turn", "第9周 · 中间审查 (中間試験)", ["Master中间审查 (目标3800分)"], "全力夺得第1名解锁P道具。", 3800),
        (10, "produce_schedule", "第10周 · 考后休整 / 外出", ["外出", "活动支给", "休息"], "恢复体力并触发事件。", 0),
        (11, "produce_schedule", "第11周 · 后半程SP课程", ["主属性SP课程", "副属性SP课程", "休息"], "向1500+主属性迈进。", 0),
        (12, "card_reward", "第12周 · 高阶授业三选一", ["SSR定制核心卡", "SR+终结卡", "高阶过牌卡"], "精选核心爆发组件。", 0),
        (13, "produce_schedule", "第13周 · 后半程SP课程", ["主属性SP课程", "副属性SP课程", "外出"], "持续压榨SP训练。", 0),
        (14, "produce_schedule", "第14周 · 后半程SP课程", ["主属性SP课程", "课堂授业", "休息"], "拉满主副属性面板。", 0),
        (15, "produce_schedule", "第15周 · 考前SP课程 / 支给", ["主属性SP课程", "活动支给", "休息"], "储备决战体力与P点。", 0),
        (16, "shop", "第16周 · 最终考前咨询商店", ["购入3瓶顶配P饮料", "卡牌自定义改造/强化", "删减冗余卡"], "完成最终卡组构筑。", 0),
        (17, "battle_turn", "第17周 · 最终冲刺课程 (最終追い込み)", ["Master最终冲刺 (目标7000分)"], "冲刺1800属性上限。", 7000),
        (18, "battle_turn", "第18周 · 最终审查 (最終試験)", ["Master最终审查 (目标20000+分)"], "极限爆发冲击S+/SS评级。", 20000),
    ]
    return {
        w: WeekScheduleNode(
            week=w,
            phase_key=f"master_week_{w}",
            title=title,
            scene_type=st,
            activities=acts,
            recommended_action=acts[0],
            operation_guide=guide,
            target_score=target,
            battle_turns=12 if target > 0 else 8,
        )
        for w, st, title, acts, guide, target in mapping
    }


def _build_nia_master_weeks() -> Dict[int, WeekScheduleNode]:
    mapping = [
        (1, "produce_schedule", "第1周 · NIA开局营业 / 课程", ["商业营业 (粉丝+三维)", "SP课程", "外出"], "积累初期粉丝数与属性。", 0),
        (2, "produce_schedule", "第2周 · 营业 / 特别指导", ["商业营业", "SP课程", "活动支给"], "提升粉丝与P点。", 0),
        (3, "produce_schedule", "第3周 · SP课程 / 营业", ["主属性SP课程", "商业营业", "休息"], "兼顾三维与粉丝。", 0),
        (4, "card_reward", "第4周 · 试镜前卡组扩充", ["SSR流派卡", "SR+强化卡", "定制卡"], "选取一次试镜主力卡。", 0),
        (5, "shop", "第5周 · 一次试镜前特别指导/商店", ["卡牌自定义改造", "购买P饮料", "删卡"], "一次试镜前整备。", 0),
        (6, "battle_turn", "第6周 · 一次试镜 (1st Audition)", ["NIA一次试镜舞台 (目标4500分)"], "打满试镜分获取大量粉丝。", 4500),
        (7, "produce_schedule", "第7周 · 中盘营业 / 外出", ["外出", "商业营业", "休息"], "试镜后恢复体力。", 0),
        (8, "produce_schedule", "第8周 · 中盘SP课程", ["主属性SP课程", "副属性SP课程", "商业营业"], "提升三维乘区。", 0),
        (9, "produce_schedule", "第9周 · 中盘营业 / SP课程", ["商业营业", "主属性SP课程", "休息"], "冲击二次试镜粉丝门槛。", 0),
        (10, "card_reward", "第10周 · 高阶卡牌奖励", ["SSR终结卡", "限1核心Buff卡", "过牌卡"], "强化卡组爆发力。", 0),
        (11, "shop", "第11周 · 二次试镜前特别指导/商店", ["高阶卡牌改造", "购买顶配饮料", "删卡"], "二次试镜前整备。", 0),
        (12, "battle_turn", "第12周 · 二次试镜 (2nd Audition)", ["NIA二次试镜舞台 (目标12000分)"], "多回合高倍率得分争夺优胜。", 12000),
        (13, "produce_schedule", "第13周 · 终盘外出 / 恢复", ["外出", "活动支给", "休息"], "恢复体力备战终盘。", 0),
        (14, "produce_schedule", "第14周 · 终盘SP课程", ["主属性SP课程", "副属性SP课程", "商业营业"], "冲刺极高三维属性。", 0),
        (15, "produce_schedule", "第15周 · 终盘营业冲刺", ["大型商业营业", "主属性SP课程", "休息"], "拉满粉丝评级倍率。", 0),
        (16, "card_reward", "第16周 · 终盘觉醒选卡", ["SSR+觉醒卡", "限1爆发卡", "免耗神卡"], "最终牌库补强。", 0),
        (17, "shop", "第17周 · FINALE前特别指导/商店", ["终极卡牌自定义改造", "购入3瓶SSR饮料", "删减冗余卡"], "FINALE决战前最后整备。", 0),
        (18, "battle_turn", "第18周 · FINALE 最终试镜 (Final Audition)", ["NIA FINALE 决战舞台 (目标30000+分)"], "全火力输出冲击SSS评级。", 30000),
    ]
    return {
        w: WeekScheduleNode(
            week=w,
            phase_key=f"nia_week_{w}",
            title=title,
            scene_type=st,
            activities=acts,
            recommended_action=acts[0],
            operation_guide=guide,
            target_score=target,
            battle_turns=12 if target > 0 else 8,
        )
        for w, st, title, acts, guide, target in mapping
    }


def _build_hif_honsen_weeks() -> Dict[int, WeekScheduleNode]:
    """Build 9-phase flow for H.I.F 本戦 (6 Prep Days + Round 1 + Interval + Round 2)."""
    mapping = [
        (1, "produce_schedule", "1日目 · 授业 (授業)", ["有SP必最优先选SP授业", "短板/副属性课堂授业"], "H.I.F本戦沿用選抜試験メモリー卡组入场。第1天有SP必最优先选SP，否则通过授业补充偏低副属性(防△评价)并拿卡推『H.I.Fワッペン』+10スター性。", 0, 8),
        (2, "produce_schedule", "2日目 · 公开课 (公開レッスン · 无打牌)", ["有SP必最优先选SP公開レッスン", "主/副属性公開レッスン"], "公開レッスン为纯自动结算课程(不进打牌)，直接抬高三维面板与スター性；有SP必最优先选SP，多门SP或均无SP时选主属性或拉高短板。", 0, 8),
        (3, "produce_schedule", "3日目 · 外出 / 活动支给 (おでかけ / 差し入れ)", ["おでかけ(拿卡推スター性与接近22张)", "差し入れ(SR+强力P饮料)"], "おでかけ消耗P点买卡可叠『H.I.Fワッペン』+10スター性且助牌库凑满~22张(防R2被塞基础卡)；差し入れ保底SR+ P饮料，根据持有P点与饮料空位权衡。", 0, 8),
        (4, "produce_schedule", "4日目 · 授业 (授業)", ["有SP必最优先选SP授业", "授业补强三维与核心卡"], "有SP必最优先选SP，继续补充偏低三维属性并拿取契合流派的核心卡，P饮料弹出属随机事件，全部由图像识别决定。", 0, 8),
        (5, "produce_schedule", "5日目 · 公开课 (公開レッスン · 无打牌)", ["有SP必最优先选SP公開レッスン", "主/副属性公開レッスン"], "Round 1前最后自动面板冲刺，有SP必最优先选SP推高スター性与三维评分乘区。", 0, 8),
        (6, "shop", "6日目 · 咨询商店 (相談)", ["购入强力P饮料 / 删减废卡 / 走换牌与强化改造", "保留部分P点给インターバル"], "⚠️注意：Round 1与Round 2之间的インターバル【不能删卡】，多余初期废卡务必在本戦前相談清理(总牌库保持约22张防R2被塞基本卡)；尽量不乱添普通新卡，优先买强饮料、走换牌或留P点给インターバル买饮料/自定义强化。", 0, 8),
        (7, "battle_turn", "当日① · 本戦ラウンド1 (Round 1 · 9回合冲分战)", ["H.I.F Round 1 9回合高分対決"], "9回合正式得分战（R1与R2总分相加定胜负）：前中盘高速将限1卡与《ありのままのわたし》打入除外压缩牌库，后期靠双发《びしっとキメ顔》与可重复核爆卡全力冲分！", 600000, 9),
        (8, "shop", "当日② · インターバル (Interval 决战中场休整)", ["买满顶级P饮料", "走セレクトチェンジ/チェンジ换牌(不乱加牌)", "余钱全投強化/カスタマイズ强化自定义主力卡"], "Round 1后进入决战中场インターバル：【无删卡功能】且剩余P点不继承！切忌乱买普通『スキルカード交換』新卡塞胖牌库破坏循环；请优先：① 买满强力P饮料 → ② 走『セレクトチェンジ/チェンジ』1换1换强卡 → ③ 花不完的P点用于『スキルカード強化 / カスタマイズ』强化自定义主力卡！", 0, 8),
        (9, "battle_turn", "当日③ · 本戦ラウンド2 (Round 2 Finale · 12回合终局决战)", ["H.I.F Round 2 12回合极限爆发"], "12回合终局决战（R1+R2总分决定優勝/プリマステラ）：前中盘全速把限1卡打进除外压缩牌库并触发再演(留《びしっとキメ顔+》配《特製ハツボシエキス》双发)，终盘进入『オトメゴコロ+ → みんな大好き+ → 認めてくれるひと+』无限循环并倾泻《キセキの魔法+》冲刺极限总分！", 1000000, 12),
    ]
    return {
        w: WeekScheduleNode(
            week=w,
            phase_key=f"hif_phase_{w}",
            title=title,
            scene_type=st,
            activities=acts,
            recommended_action=acts[0],
            operation_guide=guide,
            target_score=target,
            battle_turns=turns,
        )
        for w, st, title, acts, guide, target, turns in mapping
    }


SCENARIO_REGISTRY: Dict[str, ProduceScenarioSpec] = {
    "hajime_regular": ProduceScenarioSpec(
        scenario_id="hajime_regular",
        name="定期公演『初』· 常规 (Regular)",
        short_name="初·常规 (13周)",
        difficulty="Regular",
        total_weeks=13,
        stat_cap=1000,
        midterm_week=6,
        final_week=13,
        description="全13周标准培育剧本：第1周基础课程(无SP) → 第2~3周授业/支给 → 第4周考前商店 → 第5周中间冲刺课 → 第6周中间审查 → 第7周休整 → 第8~10周后半SP课程与高阶选卡 → 第11周终考前商店 → 第12周最终冲刺课 → 第13周最终审查（P饮料在各周目随机触发，以画面识别为准）。",
        weeks=HAJIME_REGULAR_WEEKS,
    ),
    "hajime_pro": ProduceScenarioSpec(
        scenario_id="hajime_pro",
        name="定期公演『初』· 进阶 (Pro)",
        short_name="初·进阶 (16周)",
        difficulty="Pro",
        total_weeks=16,
        stat_cap=1500,
        midterm_week=8,
        final_week=16,
        description="全16周进阶培育剧本：属性上限1500，第7周中间冲刺、第8周中间审查、第15周最终冲刺、第16周最终审查。",
        weeks=_build_hajime_pro_weeks(),
    ),
    "hajime_master": ProduceScenarioSpec(
        scenario_id="hajime_master",
        name="定期公演『初』· 大师 (Master)",
        short_name="初·大师 (18周)",
        difficulty="Master",
        total_weeks=18,
        stat_cap=1800,
        midterm_week=9,
        final_week=18,
        description="全18周高难培育剧本：属性上限1800，第8周中间冲刺、第9周中间审查、第17周最终冲刺、第18周最终审查。",
        weeks=_build_hajime_master_weeks(),
    ),
    "nia_master": ProduceScenarioSpec(
        scenario_id="nia_master",
        name="『N.I.A』Next Idol Audition",
        short_name="N.I.A (18周)",
        difficulty="Master",
        total_weeks=18,
        stat_cap=2000,
        midterm_week=6,
        final_week=18,
        description="全18周三阶段试镜剧本：穿插商业营业与特别指导，第6周一次试镜、第12周二次试镜、第18周FINALE决战。",
        weeks=_build_nia_master_weeks(),
    ),
    "hif_honsen": ProduceScenarioSpec(
        scenario_id="hif_honsen",
        name="『H.I.F 本戦』Hatsuboshi IDOL FESTIVAL",
        short_name="H.I.F 本戦 (9阶段)",
        difficulty="H.I.F",
        total_weeks=9,
        stat_cap=3200,
        midterm_week=7,
        final_week=9,
        description="H.I.F 本戦决战流程：携带選抜試験メモリー入场，历经 1日目授业 → 2日目公開レッスン(无打牌) → 3日目おでかけ/差し入れ → 4日目授业 → 5日目公開レッスン → 6日目相談 → 当日①本戦ラウンド1(9T) → 当日②インターバル(无删卡/买满强力饮料/走换牌与强化自定义/不乱加牌) → 当日③本戦ラウンド2(12T)。",
        weeks=_build_hif_honsen_weeks(),
    ),
}


# ---------------------------------------------------------------------------
# 3. 职业玩法详解、被动P道具加成 (P-Items)、场景加成 (Stage Bonuses) 与初始数值
# ---------------------------------------------------------------------------

ARCHETYPE_PLAYSTYLE_GUIDES: Dict[str, Dict[str, Any]] = {
    "sense": {
        "name": "感性 (Sense) · 好调/集中爆发流",
        "core_mechanics": [
            "好调 (Good Condition)：使所有主动得分卡的基础得分提升 50%，每回合结束时层数 -1；维持好调不断档是一切输出的前提。",
            "绝好调 (Perfect Condition)：在好调激活时进一步提升得分倍率（每层好调额外提供 +10% 倍率），是尾局核爆的核心乘区。",
            "集中 (Concentration)：每点集中使主动得分卡的每次攻击额外附加固定得分（受好调与审查倍率双重放大），多段攻击卡收益翻倍。",
        ],
        "turn_playstyle": (
            "【无条件最优先】凡带有「+使用次数(使用数追加+1)」的角色固有启动卡（开局固定上手或不占出牌位）必须第一时间立刻打出，绝不拖延！\n"
            "【开局/前中期 (第1~4回合)】打出「好调」与「集中」Mental 养成卡；《初》通常/SP课程在前期打满目标分后剩余回合会自动兑换成体力，切勿因惧怕扣血而跳过回合。\n"
            "【爆分/尾局阶段 (主属性高倍率回合或最后3回合)】果断扣血连续打出高倍率多段主动卡与终结技抢满分数；体力>=10且有有效牌时严禁无效SKIP。"
        ),
        "default_p_items": [
            {
                "name": "当前剧本/偶像自带P道具（以实机画面与Buff数值为准）",
                "trigger": "满足装备P道具各自条件时自动生效",
                "effect": "切勿臆测或假设未在手牌效果/当前Buff中明确写出的额外加干劲/加元气/加分效果，请严格按手牌自身数值与当前画面状态计算收益。",
            },
        ],
    },
    "logic": {
        "name": "理性 (Logic) · 好印象叠层再演 / 干劲元气护盾双系",
        "core_mechanics": [
            "好印象 (Good Impression · 好印象主轴如[ガラクタロード]秦谷美铃): 每回合结束时触发「好印象层数 × 舞台倍率」的自动追加得分并 -1 层；配合「好印象强化(+50%/+100%/+200%)」与「好印象追加发动(+1)」及『ありのままのわたし』再演、高除外数触发『認めてくれるひと』(+280%) 越到后期得分越恐怖。",
            "干劲 (Motivation · 干劲元气主轴或好印象耗干劲协卡): 每点干劲使后续卡牌获得元気(护盾)时等量叠加护盾；好印象构筑亦常用少量干劲为『輝くキミへ』『オトメゴコロ』『私が最強！』等强力好印象卡支付干劲费用。",
            "固有引擎各异 (按当前出战P偶像与固有P道具区分): [ガラクタロード]秦谷美铃依靠早期打出《ありのままのわたし》进【除外】触发【再演】+4次好印象加层触发P道具『星々を統べる輝き』，并用『限1』卡把除外堆到7/11张解锁《認めてくれるひと+》280%爆破；[赤裸々]十王星南则依靠《私を超えて》每2张元気卡生成《私を超えて(翔)》连发爆分。",
            "护盾与扣血机制: `🟢绿血(cost-=X)` 先扣🛡️元気护盾且受羽毛减半，`🔴红血(stamina-=X)` 无视护盾直接扣真实体力；《初》课程「CLEARまで X」归0后剩余回合可换算回血，正式舞台/本戦剩血不带出结算，严禁无故点SKIP！",
        ],
        "turn_playstyle": (
            "【无条件最高优先】手上若有带「⭐固有启动」、「⚡+使用次数(不耗行动点)」或「⚡メモリー首次免费再动」的引擎/免耗卡，优先于普通1费卡率先打出（避免剩余出牌1->0直接结束回合把免费卡丢弃）！\n"
            "【围绕当前P卡专精轴运营】若出战好印象流偶像([ガラクタロード]秦谷美铃等)，前中盘高速把限1卡与《ありのままのわたし》打入除外压缩牌库(善用止められない想い/オトメゴコロ/ワクワク免费除外《眠気》，绝不手打眠気)，中后盘先喝《特製ハツボシエキス》再打『びしっとキメ顔+』叠双倍追击，终盘循环『オトメゴコロ+→みんな大好き+→認めてくれるひと+』与『キセキの魔法+』爆分；若为干劲护盾流，则堆高干劲与元气靠转化A卡终结。\n"
            "【绝不无效SKIP】阶段考试与H.I.F本戦剩余血量皆不带出结算，前中盘全速压缩除外，末盘务必果断消耗体力/护盾/饮料打光上限分数。"
        ),
        "default_p_items": [
            {
                "name": "当前剧本/偶像自带P道具（以实机画面与Buff数值为准）",
                "trigger": "按当前出战P卡固有道具与剧本道具触发",
                "effect": "严格依据已登记P道具与卡牌原文效果计算收益，切勿混淆不同偶像流派或臆测未记载的额外被动。",
            },
        ],
    },
    "anomaly": {
        "name": "异常 (Anomaly) · 强气/温存姿态与全力爆发流",
        "core_mechanics": [
            "强气姿态 (Strong Spirit / 強気)：主动得分卡伤害大幅提升（+100%~+200%），但卡牌消耗增加；打出强气卡可累积「全力充能」。",
            "温存姿态 (Preservation / 温存)：卡牌消耗大幅降低，退出温存时返还体力/额外抽牌与出牌次数，用于过渡蓄力与控血。",
            "全力状态 (Full Power / 全力)：当「全力充能 (fullPowerCharge)」累积满10点时在下回合自动进入全力姿态，卡牌倍率暴涨并自动从牌库/弃牌堆召回「全力限定」终结卡。",
        ],
        "turn_playstyle": (
            "【无条件最优先】手握「+使用次数」免费切换/启动固有卡时立刻打出，不占用本回合出牌机会。\n"
            "【蓄力过渡期】交替切换温存与强气姿态赚取费用减免、抽牌与充能点数，无需过度省血点SKIP；\n"
            "【全力爆发回合】确保在审查高倍率回合恰好攒满10点充能进入「全力姿态」，一回合内倾泻所有高倍率终结卡完成达标或高分结算。"
        ),
        "default_p_items": [
            {
                "name": "当前剧本/偶像自带P道具（以实机画面与Buff数值为准）",
                "trigger": "按实机当前装备与场上Buff状态生效",
                "effect": "切勿臆测额外白嫖效果，请严格以手牌效果原文与实机面板数值计算收益。",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# 4. 初始卡组模板 (按流派与代表 P 偶像提供卡库基准)
# ---------------------------------------------------------------------------

STARTER_DECK_BY_PLAN: Dict[str, List[int]] = {
    "logic": [1, 1, 3, 3, 13, 13, 21, 21, 11],
    "sense": [1, 1, 3, 3, 13, 13, 15, 17, 5],
    "anomaly": [1, 1, 3, 3, 13, 13],
}

IDOL_SIGNATURE_START_CARD: Dict[int, int] = {
    126: 756,  # 十王 星南 [赤裸々] -> 私を超えて+
    163: 920,  # 秦谷 美鈴 [ガラクタロード] -> ありのままのわたし+
}

IDOL_EXTRA_SIGNATURE_CARDS: Dict[int, List[int]] = {
    163: [920, 922],  # 秦谷 美鈴 [ガラクタロード] 携带 ありのままのわたし+ & 認めてくれるひと+
}


def get_scenario_spec(scenario_id: str = "hajime_regular") -> ProduceScenarioSpec:
    normalized = "hif_honsen" if (scenario_id or "").lower() in ("hif", "hif_honsen") else scenario_id
    return SCENARIO_REGISTRY.get(normalized, SCENARIO_REGISTRY["hajime_regular"])


def get_week_node(scenario_id: str = "hajime_regular", week: int = 1) -> WeekScheduleNode:
    spec = get_scenario_spec(scenario_id)
    clamped = max(1, min(spec.total_weeks, int(week)))
    return spec.weeks[clamped]


def compute_week_stats_and_stage_bonus(
    scenario_id: str = "hajime_regular",
    week: int = 1,
    plan: str = "sense",
    override_stats: Optional[Dict[str, int]] = None,
    p_idol_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns realistic idol stats (Vocal/Dance/Visual), stage criteria multipliers,
    and active P-Items for the given scenario, week, archetype plan, and optional P-Idol.
    """
    from demo.core.idol_guides import get_p_idol_guide, resolve_p_idol_id

    spec = get_scenario_spec(scenario_id)
    node = get_week_node(scenario_id, week)
    plan_key = (plan or "sense").lower()
    guide = ARCHETYPE_PLAYSTYLE_GUIDES.get(plan_key, ARCHETYPE_PLAYSTYLE_GUIDES["sense"])
    resolved_pid = resolve_p_idol_id(plan=plan_key, p_idol_id=p_idol_id)
    idol_guide = get_p_idol_guide(resolved_pid)
    p_items = [dict(idol_guide.unique_p_item)] if idol_guide is not None else list(guide["default_p_items"])

    # Progressive stat scaling across weeks (Week 1 base -> Final Week cap)
    progress = max(0.0, min(1.0, (week - 1) / max(1, spec.total_weeks - 1)))
    base_vo = int(150 + progress * (spec.stat_cap * 0.62))
    base_da = int(135 + progress * (spec.stat_cap * 0.52))
    base_vi = int(120 + progress * (spec.stat_cap * 0.38))
    stats = dict(override_stats) if override_stats else {"vocal": base_vo, "dance": base_da, "visual": base_vi}

    # Calculate score multipliers from stats
    vo_mult = round(100 + stats.get("vocal", base_vo) * 0.18, 1)
    da_mult = round(100 + stats.get("dance", base_da) * 0.16, 1)
    vi_mult = round(100 + stats.get("visual", base_vi) * 0.14, 1)

    stage_bonus = {
        "stage_name": node.title,
        "vocal_bonus_pct": vo_mult,
        "dance_bonus_pct": da_mult,
        "visual_bonus_pct": vi_mult,
        "special_rule": (
            "H.I.F 本戦正式决战舞台（ラウンド1 9T + ラウンド2 12T，两轮总分决定最终胜负）：不设封顶目标分，全回合叠强流派引擎+末盘倾泻所有高倍转化卡与P饮料冲刺最高总分！"
            if spec.scenario_id == "hif_honsen" and node.week in (7, 9)
            else "审查特殊规则：第1回合与最终回合为主属性流行爆发轮（得分倍率额外提升 50%）；"
            "每打出1张契合当前流派的核心卡，审查评价加成提升。"
            if "exam" in node.phase_key
            else "冲刺课特殊规则：在限定回合内达成 PERFECT 目标分即可额外获得大量主副属性与P点数奖励。"
            if "oikomi" in node.phase_key
            else "日常培育规则：根据当前进度日程与剩余体力选择收益最高的培育活动。"
        ),
    }

    return {
        "scenario_id": spec.scenario_id,
        "scenario_name": spec.name,
        "total_weeks": spec.total_weeks,
        "week": node.week,
        "week_node": node,
        "idol_stats": stats,
        "stage_bonuses": stage_bonus,
        "p_items": p_items,
        "playstyle_guide": guide,
        "idol_guide": idol_guide,
    }
