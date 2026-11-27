"""
P-Idol Specific Strategy Guide Registry (Crawled & DB-Verified).

Stores authoritative per-P-Idol operational guides extracted from Japanese Gakumas
strategy sites (SeesaaWiki, Game8, note.com, Morishimemo) and cross-verified
against gakumas_library.db DSL rules so Prompts can inject exact P-Idol mechanics,
synergies, anti-synergies, drink pairings, and turn-by-turn battle execution rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PIdolGuide:
    """Structured strategy & operational guide for a specific P-Idol card."""

    p_idol_id: int
    idol_name: str
    title: str
    rarity: str
    plan: str
    recommended_effect: str
    source_urls: Tuple[str, ...]
    unique_p_item: Dict[str, str]
    signature_cards: Tuple[str, ...]
    core_mechanics: Tuple[str, ...]
    battle_pacing_rules: Tuple[str, ...]
    drink_synergies: Tuple[str, ...]
    anti_synergies: Tuple[str, ...]
    recommended_cards: Dict[int, str]
    recommended_card_names: Dict[str, str]
    caution_cards: Dict[int, str]
    caution_card_names: Dict[str, str]


_SEKIRARA_SENA_GUIDE = PIdolGuide(
    p_idol_id=126,
    idol_name="十王 星南",
    title="赤裸々",
    rarity="SSR",
    plan="logic",
    recommended_effect="motivation",
    source_urls=(
        "https://seesaawiki.jp/gakumasu/d/%A1%DA%C0%D6%CD%E7%A1%B9%A1%DB%BD%BD%B2%A6%C0%B1%C6%EE",
        "https://game8.jp/gakuen-idolmaster/760902",
        "https://game8.jp/gakuen-idolmaster/753478",
        "https://note.com/fit_bee8608/n/n6168f2da2fb1",
        "https://morishimemo.com/gakumasu-memo-44",
    ),
    unique_p_item={
        "name": "最初の鳥",
        "trigger": "每2回合（偶数回合开始时）且当前元气 >= 30",
        "effect": (
            "体力回复+1（才能开花2及以上+2），并按除外区《私を超えて(翔)》每张追加获得"
            "【元气×20%（开花2+为25%）】的参数得分。注：元气<30时偶数回合不触发！"
        ),
    },
    signature_cards=(
        "#755/#756 私を超えて(Mental·固有启动): 开局必上手，+1使用数(不耗行动点)；"
        "此后每打出2张带「元気」效果的技能卡，在牌库最上方生成1张《私を超えて(翔)》"
        "（即使本体已强化+，生成的(翔)初始仍为未强化版80%）。",
        "#757/#758 私を超えて(翔)(Active·衍生核心大招): 要求元気>=15、固定消耗体力1、"
        "造成元気×80%(+强化130%)参数得分、并自带+1使用数(不耗行动点)+抽1张牌；"
        "因生成于牌库顶而几乎不畏惧「眠気」污染，且牌库/弃牌堆有多张(翔)时可同回合自抽连发。",
        "#876 夢を追い続ける一番星(H.I.F专用Legend): +1使用数，后续4次因卡牌费用减少体力时元気+5；"
        "自身亦属于「元気效果卡」，可计入(翔)的2次生成计数，尽量留到中后期联动(翔)扣血。",
    ),
    core_mechanics=(
        "【核心定位】赤裸々星南属于「やる気(干劲)→元気→元気%爆分」轴；但【自身固有卡与P道具完全不直接提供やる気或元気】"
        "——必须依靠牌库中抓来的「やる気(干劲)卡」先抬升增量，再靠「元気卡」滚雪球叠加护盾与发电《私を超えて(翔)》。",
        "【30元气阈值与续航】固有P道具『最初の鳥』只在【偶数回合开始时】检查【元気>=30】才触发回血；"
        "因此第1/3/5回合打完后尽量让回合末元气保底接近/达到30，中前盘无需为了触发P道具追伤而急着提前交出《私を超えて(翔)》。",
        "【(翔)门槛与联动特征】《私を超えて(翔)》要求【元気>=15】且【体力>1】(固定扣血1，穿透元气护盾)，"
        "但它自身【不带元気效果】，不会推高下一次(翔)生成计数；若当前有高元气与多张(翔)，可在不耗行动点下靠「抽1张」连续接力爆发。",
    ),
    battle_pacing_rules=(
        "第1回合起手：若《私を超えて》/《私を超えて+》上手，务必第1手无条件打出（可与《本番前夜》《ワクワクが止まらない》同回合免费连出），"
        "紧接着打出提高「やる気」或「元気」的M卡启动引擎，绝不在首回合漏打《私を超えて》。",
        "前中盘蓄力期（考试剩>=5回合）：优先出「やる気」加倍/叠加卡与高效「元気」卡，高频凑每2张元気卡稳定往牌库顶存《私を超えて(翔)》，"
        "并尽早在第1/3/5回合把元气稳在>=30吃偶数回合回血。⚠️【行动点严格铁律】：当剩余出牌次数=1时，先打出任何不带『+1使用次数』的普通卡会直接把行动点耗光(1->0)并立刻结束回合（导致手里未打的《私を超えて(翔)》白白进弃牌）！"
        "因此仅P饮料或自身也带『+1使用次数』的卡（《私を超えて》《本番前夜》《ワクワクが止まらない》）能排在(翔)前；凡《私を超えて(翔)》已解锁(元気>=15且体力>1)且本回合决定打出时，必须【先打(翔)】(+1使用数且抽1牌)，然后再出后续A/M卡！",
        "后期全段爆发（考试剩<=4回合或主属性高倍率回合）：高干劲+高元气成型后，优先【先打】牌库顶多张《私を超えて(翔)》的「+1使用数+抽1」同回合连抽爆破，"
        "衔接《届いて！》《せのびの魔法》《気分転換》把全额元气倾泻为得分；考试剩血不带出下周，后期只要还能行动决不因省1点体力点SKIP。",
        "日常/SP/冲刺课程清分铁律（带CLEARまで/PERFECTまで）：无论前中盘还是倒数第3/倒数第2/最终回合(`剩余<=3回合`)，只要 `CLEARまで`/`PERFECTまで > 0`，手牌已解锁的《私を超えて(翔)》与「0扣血」A卡/高分A卡必须立即先打出来打满 `CLEAR`/`PERFECT`（禁止在剩余<=3回合打0分纯M卡而留免费A卡在手里）；打穿归0后剩余回合自动换回大量体力！",
    ),
    drink_synergies=(
        "特製ハツボシエキス(#25)：仅强化本回合下一张非Legend主动卡(Active)且体力-2+耗血翻倍1T；"
        "对Mental卡《私を超えて》完全无效！必须留到后期手里握有《せのびの魔法》《届いて！》或强化版《私を超えて(翔)+》的爆发回合再喝。",
        "フレッシュビネガー：最赚时机是手里同时存在2张以上未强化《私を超えて(翔)》(80%->130%)或关键主力爆发卡时一并强化，切勿开局空牌/只有基础牌时浪费。",
        "烏龍茶(#2)：关键阈值微调神器——可用于单差一点元气达到>=15解锁《私を超えて(翔)》、偶数回合前凑>=30触发『最初の鳥』回血，或末盘垫元气打爆发大招。",
        "ホットコーヒー / センブリソーダ：持续干劲+抽牌加速神器，考试前中盘(1-5回合)有即可喝，显著加速每2张元気卡生成《私を超えて(翔)》的循环。",
    ),
    anti_synergies=(
        "⚠️ 严禁误用减耗负配合：若构筑中已有《最強パフォーマー》(#726)或H.I.F《夢を追い続ける一番星》(#876)（均依赖技能卡消费【减少体力】触发回元气），"
        "绝不叠加「消费体力削减」把《私を超えて(翔)》的体力-1降为0，因此《さっぱりひといき》(#336/#337)等减耗卡应慎选/避开（已有上述扣血回气引擎时优先再抽选替换）。",
        "⚠️ 严防好印象杂牌误导：虽然同属Logic池，但赤裸々星南本身没有任何好印象协同或被动，「好印象」非核心输出轴，选卡严禁抓只加少量好印象、不产やる気/元気的低质杂牌。",
        "⚠️ 防止无法出牌卡手：元气<15时《私を超えて(翔)》无法打出，体力=1时同样打不出《私を超えて(翔)》（且先扣至0体力会失败），勿在未达15元气前指望(翔)斩杀。",
    ),
    recommended_cards={
        755: "★赤裸々固有启动(M)·必抓必强化,开局上手+1使用数并每2张元気卡生(翔)",
        756: "★赤裸々固有启动(M+)·成本减半,第1手无条件打出",
        757: "★赤裸々衍生大招(A)·元気>=15可打,费1血,+1使用数+抽1牌可连发",
        758: "★赤裸々衍生大招(A+)·130%元気高伤,+1使用数+抽1牌",
        876: "★H.I.F专属Legend(M)·扣血时元气+5且本身计入元気触发(翔)计数",
        130: "★T0干劲引擎(M·限1)·开局可与私を超えて同回合免费连打起干劲",
        131: "★T0干劲引擎(M+·限1)·极高开局优先级",
        128: "★T0干劲引擎(M·限1)·核心开局干劲叠加卡",
        129: "★T0干劲引擎(M+·限1)·必拿核心干劲",
        534: "★T0干劲倍率器(M·限1)·干劲×1.5,中后期大幅拉高元気产出",
        535: "★T0干劲倍率器(M+·限1)·必拿",
        600: "★T0高额元気(M·限1)·直接堆高元气触发(翔)与P道具30线",
        601: "★T0高额元気(M+·限1)·必拿",
        726: "★T0 Legend神卡(M·限1)·搭配(翔)体力-1扣血无限刷高额元気(忌与减耗同时抓)",
        134: "★T0干劲+元気双修(M·限1)·完美贴合赤裸々引擎",
        135: "★T0干劲+元気双修(M+·限1)·极高抓取顺位",
        136: "★T0干劲核心(M·限1)·稳定补充干劲底数",
        137: "★T0干劲核心(M+·限1)",
        60: "★核心元気循环卡(M)·可重复打满2次稳定生产(翔)",
        61: "★核心元気循环卡(M+)·优质无限循环元気组件",
        714: "★高频元気循环卡(M)·可多次打出刷(翔),牌库顶生(翔)不大怕眠気",
        715: "★高频元気循环卡(M+)",
        150: "★T1优质元気/收割组件(A·限1)",
        151: "★T1优质元気/收割组件(A+·限1)",
        90: "★T1高元气+额外收益(A·限1)",
        91: "★T1高元气+额外收益(A+·限1)",
        46: "★T1常用元気卡(A·限1)·兼顾攒气与计数",
        47: "★T1常用元気卡(A+·限1)",
        98: "★T1辅助抽取/免费(+1使用数)",
        99: "★T1辅助抽取/免费(+1使用数+)",
        771: "★T0终局爆发(A·限1)·搭特製ハツボシエキス极强",
        772: "★T0终局爆发(A+·限1)·末盘爆分主力",
        152: "★T0终局元气转化大招(A·限1)·末盘配合高元気收割",
        153: "★T0终局元气转化大招(A+·限1)",
        304: "★T0后期爆发大招(A·限1)",
        305: "★T0后期爆发大招(A+·限1)",
        11: "★末盘元气转化斩杀卡(A·限1)·前中盘勿提前空甩",
        12: "★末盘元气转化斩杀卡(A+·限1)",
    },
    recommended_card_names={
        "私を超えて": "★赤裸々固有启动(M)·开局首手必打",
        "本番前夜": "★T0干劲引擎·完美搭固有开局",
        "ワクワクが止まらない": "★T0干劲核心",
        "わたしらしい色": "★T0干劲1.5倍",
        "ドラマチックホリデー": "★T0大量堆元气",
        "最強パフォーマー": "★T0 Legend·与(翔)扣血神联动",
        "イメトレ": "★T0干劲+元气",
        "えいえいおー": "★T0可重复元气生成器(刷翔)",
        "成長痛": "★高频元气生成器(牌库顶生翔防眠気)",
        "届いて！": "★T0终局元气爆发",
        "せのびの魔法": "★T0终局爆发(绝配ハツボシエキス)",
        "夢を追い続ける一番星": "★H.I.F专属·扣血回5元气",
    },
    caution_cards={
        336: "⚠️注意反协同·给消费体力削减会把(翔)体力-1降到0导致《最強パフォーマー》失灵(可再抽选避开)",
        337: "⚠️注意反协同·消费体力削减慎与扣血回气体系混抓",
        782: "⚠️条件苛刻(仅最终回合可用),常规稳健优先《最強パフォーマー》",
    },
    caution_card_names={
        "さっぱりひといき": "⚠️消费体力削减慎选(防使(翔)0耗血导致扣血回气失效)",
    },
)


_GARAKUTA_MISUZU_GUIDE = PIdolGuide(
    p_idol_id=163,
    idol_name="秦谷 美鈴",
    title="ガラクタロード",
    rarity="SSR",
    plan="logic",
    recommended_effect="goodImpressionTurns",
    source_urls=(
        "https://seesaawiki.jp/gakumasu/",
        "https://game8.jp/gakuen-idolmaster/",
        "https://appgamelog.com/",
    ),
    unique_p_item={
        "name": "星々を統べる輝き",
        "trigger": "通过卡牌/效果直接增加「好印象」累计每满 4 次时触发",
        "effect": (
            "赋予 2 回合【好印象追加发动 +1】（回合结束时好印象额外结算 1 次得分）"
            "并赋予 2 回合【好印象强化 +10%】；提高「直接增加好印象次数」的频率是核心发动机。"
        ),
    },
    signature_cards=(
        "#919/#920 ありのままのわたし / +(Mental·限1进除外·緑血7/5): "
        "元気+3(+4) 并赋予【好印象强化 +50%】；"
        "打出进【除外】后激活【再演】：此后通过直接效果增加好印象每满 3 次时，"
        "不耗费用与行动点自动再使用自身 1 次（每回合限 1 次、每场最多再演 2 次，最高叠加至 +150% 好印象强化）！务必前 1~3 回合尽早打出。",
        "#921/#922 認めてくれるひと / +(Active·可重复使用→打出进弃牌·緑血6): "
        "基础好印象 +3；若【除外】区技能卡 >= 7 张额外好印象 +5（触发第2段直接加好印象计数！）；"
        "若【除外】区技能卡 >= 11 张，额外造成好印象 ×170%(+强化 ×280%) 的巨额参数得分！可随牌堆循环多次打出收割。",
    ),
    core_mechanics=(
        "【核心引擎：好印象强化滚雪球 + 『再演』频次触发】"
        "ガラクタロード秦谷美铃围绕『ありのままのわたし』(#919/#920) 与固有P道具『星々を統べる輝き』构建双引擎："
        "先尽早把『ありのままのわたし』打入【除外】，再高频打出能直接增加好印象的卡牌（每第 3 次触发再演、每第 4 次触发P道具追加发动），"
        "把好印象叠加倍率拉到 +150%~+400%+ 并实现每回合多次好印象结算。",
        "【编入選抜メモリー与カスタム联动（必记！）】"
        "当前 H.I.F 選抜メモリー编成中：『止められない想い』(#296/#297)、『オトメゴコロ』(#298/#299)、『ワクワクが止まらない』(#128/#129) 打出后会【免费将 1 张《眠気》移动至除外区】（自动清除废牌+推进除外计数+加速压缩牌库，因此看到《眠気》绝不要浪费行动点主动去打！）；"
        "同时『やる気は満点』(#136/#137) 附带【每场首次打出 1 次免费再动（+1使用次数·不耗行动点）】，首次上手必须在普通耗点卡之前先打出送入【除外】！",
        "【H.I.F 本戦核心战术：高速除外压缩 → 终盘无限循环（圧縮ループ）】"
        "前中盘全速打出带【限1(`limit_count=1`)】的卡牌将其永久送进【除外】区压缩牌库（并迅速跨过除外>=7 / >=11 张门槛，绝不无故SKIP让限1卡掉回弃牌污染抽牌池！）；"
        "等到终盘牌堆+弃牌被压缩至仅剩 3~4 张可重复核心卡时，每回合稳定循环『オトメゴコロ+』(+1使用数·加好印象) → 『みんな大好き+』(+1使用数·120%分) → 『認めてくれるひと+』(#921/#922，除外>=7双段好印象、除外>=11轰出 280% 好印象核爆且回弃牌反复抽回)！",
        "【やる気(干劲)协同与商店换牌强化铁律】"
        "保留『ゆめみごこち』『本番前夜』『夢色リップ』『止められない想い』为『輝くキミへ』『オトメゴコロ』『みんな大好き』『私が最強！』提供干劲供能；"
        "进商店（相談/インターバル）尽量不要点普通『スキルカード交換』购买添加新牌塞胖牌库破坏循环！优先：① 买强力P饮料填槽 → ② 走『セレクトチェンジ/チェンジ』换牌提纯(不加牌数，或6日目删废卡) → ③ 花不完的P点全用于『スキルカード強化 / カスタマイズ』强化自定义主力卡！",
    ),
    battle_pacing_rules=(
        "单回合起手与免费再动铁律：同回合内必须最先打光所有不耗行动点的『+1使用次数』免费再动卡（含首次免费再动的『やる気は満点』、『アイドル魂』、『アイドル宣言』、『私がスター』、『止められない想い』、『輝くキミへ』、『オトメゴコロ』、『みんな大好き』）；多张同手时优先打过牌再动与能免费将《眠気》送进除外的『止められない想い / オトメゴコロ』！",
        "前中盘高速除外压缩期：若固有1『ありのままのわたし』(#919/#920) 上手且未除外，务必最优先打出进【除外】激活【再演】(+50%×3)；消耗本回合最后 1 点行动点时，在牌库压缩完成前优先打出【限1】M卡/辅助卡（如『ワクワクが止まらない(顺带免费除外眠気)』『ファンシーチャーム』『虹色ドリーマー』『トキメキ』『夢色リップ』『ノートの端の決意』『対象、ロックオン』）永久进【除外】压缩牌库！绝不主动花行动点打《眠気》，也绝不无故SKIP拖慢除外（仅将『びしっとキメ顔+』留到中后盘配『特製ハツボシエキス』双发开启，『キセキの魔法+』留到末 1~3 回合终结）。",
        "中后盘王炸双发启动（剩 3~6 回合且好印象>=10）：手中握有『びしっとキメ顔』(#777/#778) 与『特製ハツボシエキス』(#25) 时，务必【先喝特製ハツボシエキス → 立刻打出びしっとキメ顔+】，触发双倍效果叠出【4回合 +400% 好印象强化 + 每次直接加好印象双倍追击 200% 分】！",
        "终盘极小牌库循环收割（牌库已压缩且【除外>=11】）：每回合高速循环『オトメゴコロ+』(+1使用数·触发双倍キメ顔追击) → 『みんな大好き+』(+1使用数·120%分) → 『認めてくれるひと+』(2段加好印象触发双倍追击 + 280%核爆)！",
    ),
    drink_synergies=(
        "特製ハツボシエキス(#25)：下一张打出的非Legend主动卡效果额外再发动 1 次（耗2血+1T耗血翻倍，仅当回合1T有效！）。"
        "【头号最优目标】强烈优先绑定『びしっとキメ顔+』(#778/#777)！先喝エキス再打『びしっとキメ顔+』可直接叠出 4 回合 +400% 好印象强化 + 后续每次直接加好印象触发 2×100%=200% 双倍追击，配合终盘循环总分远超单发爆发卡！"
        "若持有 2 瓶エキス，切忌同回合连喝 2 瓶给同一张卡（不会变3倍，而是作用于接下来2张A卡）：第 1 瓶给『びしっとキメ顔+』，第 2 瓶给『輝くキミへ+』或【除外>=11】的『認めてくれるひと+』；当回合手牌无可用非Legend主动(A)卡时严禁空喝作废！",
        "センブリソーダ(#14)：前中盘抽牌压缩引擎神器（+2抽并后续5T开局抽1），极大幅加速【限1】卡全数除外进入终盘循环。",
        "パワフル漢方ドリンク(#13)：可定向把牌堆或弃牌中的偶像固有卡（『ありのままのわたし』或『認めてくれるひと+』）拉入手牌并减耗 3T，缺固有启动或终盘补刀时必喝。",
        "初星ホエイプロテイン(#18) & 厳選初星ティー(#23) & ブーストエキス(#12) & おしゃれハーブティー(#16)：前中盘早喝『厳選初星ティー』『ルイボスティー』『ブーストエキス』叠层增伤，终盘高好印象时喝『おしゃれハーブティー』(严禁好印象=0白喝)与『ホエイプロテイン』加行动点多转一轮循环。",
    ),
    anti_synergies=(
        "⚠️ 严禁误抓与好印象轴无关或定位冲突的牌：绝不保留『届いて！』(#152/#153)、『キセキの魔法』(#426/#427)，并避开『気分転換』(#11/#12)、『せのびの魔法』(#771/#772)、『あのときの約束』(#306/#307, 扣3层好印象!)、『きらきら紙吹雪』(#398/#399)、『最強パフォーマー』(#726)。",
        "⚠️ 严禁主动打《眠気》浪费行动点（交由『止められない想い / オトメゴコロ / ワクワクが止まらない』メモリー效果免费移入除外），也严禁在手牌无适配非Legend主动卡时空喝『特製ハツボシエキス』(#25)！",
        "⚠️ H.I.F 本戦商店与构筑铁律：进商店（相談/インターバル）尽量不要购买添加普通新技能卡（尤其严禁拿非限1可重复杂牌污染终盘循环，同时维持总牌库约22张防『H.I.F応援棒』塞基本卡）；优先买满强力P饮料，卡牌调整走『セレクトチェンジ/チェンジ』1换1换牌，花不完的P点全投进『スキルカード強化 / カスタマイズ』强化自定义核心卡！",
    ),
    recommended_cards={
        919: "★ガラクタロード固有1(M·限1进除外)·必第一时间打出激活【再演】+50%×3好印象强化",
        920: "★ガラクタロード固有1(M+·限1进除外)·降耗至5,尽早打出启动【再演】引擎",
        921: "★ガラクタロード固有2(A·可重复进弃牌)·除外>=7加倍好印象,>=11解锁170%好印象爆发",
        922: "★ガラクタロード固有2(A+·可重复进弃牌)·除外>=11解锁280%核爆,配エキス双发毁天灭地",
        777: "★T0好印象核爆引擎(A·限1)·好印象强化+200%(3T)且后续加好印象直接追伤100%",
        778: "★T0好印象核爆引擎(A+·限1)·4回合+200%强化+直接加好印象追击",
        304: "★T0核心持续伤+免费(+1使用数·A·限1)·每出牌追好印象30%分,耗干劲4",
        305: "★T0核心持续伤+免费(+1使用数·A+·限1)·每出牌追好印象50%分,极强",
        298: "★T0免费好印象卡(M·可重复·+1使用数)·直接+4好印象推再演与P道具计数,耗干劲3",
        299: "★T0免费好印象卡(M+·可重复·+1使用数)·直接+5/+7好印象推计数神器",
        138: "★T0免费干劲续航(M·可重复·+1使用数)·耗2好印象换4干劲保证不断干劲",
        139: "★T0免费干劲续航(M+·可重复·+1使用数)·仅耗1好印象换5干劲,必抓",
        96: "★T0免费输出卡(A·可重复·+1使用数)·好印象90%分,不耗行动点",
        97: "★T0免费输出卡(A+·可重复·+1使用数)·好印象120%分(仅耗2干劲)",
        781: "★T0 Legend超级爆发(A·可重复·+1使用数)·需干劲>=6耗5,好印象+12 & 250%伤 & 下T抽2",
        725: "★T0 Legend强化神卡(M·限1·+1使用数)·全牌好印象+6 & 好印象强化+100%,兼顾堆除外",
        724: "★T0 Legend过牌神卡(M·限1·+1使用数)·5回合抽牌+每回合自动打2张手牌R卡",
        616: "★T0好印象强化(M·限1)·好印象强化+100%(3T)+条件抽牌,极契合好印象轴",
        617: "★T0好印象强化(M+·限1)·绿血降至2且带3T稳定抽牌与堆除外",
        88: "★T1好印象+直接得分(A·可重复)·好印象+4并附60%伤,推触发计数器与补刀皆优",
        89: "★T1好印象+直接得分(A+·可重复)·好印象+5并80%伤",
        731: "★T1优质过牌+好印象(M·限1·+1使用数)·耗干劲补抽牌与好印象并堆除外",
        732: "★T1优质过牌+好印象(M+·限1·+1使用数)",
        296: "★T1双加+免费限1(M·限1·+1使用数)·不耗干劲还补3干劲+3好印象并堆除外",
        297: "★T1双加+免费限1(M+·限1·+1使用数)·补4干劲+5好印象并堆除外",
        508: "★T1零耗限1(M·限1)·0消耗+2好印象+回1血,+1使用数(好印象>=10)",
        509: "★T1零耗限1(M+·限1)·0消耗+2好印象+回3血,+1使用数",
        130: "★T1开局固定上手(M·限1)·第1回合到手直接+4好印象+3干劲并堆除外",
        131: "★T1开局固定上手(M+·限1)·+5好印象+4干劲",
        520: "★T1好印象×1.1雪球(M·限1)·补4干劲并在好印象>=15后连续3T乘以1.1",
        521: "★T1好印象×1.1雪球(M+·限1)·连续4T好印象1.1倍+5干劲",
        710: "★T1高好印象过牌(M·限1·+1使用数)·直接+6好印象+抽1并堆除外",
        711: "★T1高好印象过牌(M+·限1·+1使用数)·+7好印象+抽1",
        126: "★T1连续好印象计数(M·限1)·后续每张M卡再+1好印象极速叠再演与P道具",
        127: "★T1连续好印象计数(M+·限1)",
        136: "★T1低费限1(M·限1)·仅1费+4好印象快速叠层并堆除外",
        137: "★T1低费限1(M+·限1)",
        747: "★T1好印象强化限1(M·限1)·好印象+4并+10%好印象强化+堆除外",
        748: "★T1好印象强化限1(M+·限1)",
        366: "★T1限1输出兼叠层(A·限1)·好印象+3并90%分",
        367: "★T1限1输出兼叠层(A+·限1)·好印象+4并100%分",
        44: "★T1纯倍率(A·限1)·好印象150%分",
        45: "★T1纯倍率(A+·限1)·好印象200%分",
    },
    recommended_card_names={
        "ありのままのわたし": "★ガラクタロード固有1·必首优先打出进除外激活【再演】+50%×3强化",
        "認めてくれるひと": "★ガラクタロード固有2·可重复打出,除外>=11爆280%好印象分",
        "びしっとキメ顔": "★T0好印象强化+200%&每次加好印象追伤100%",
        "輝くキミへ": "★T0免费出牌追伤神卡(耗干劲)",
        "オトメゴコロ": "★T0免费直接+好印象推进【再演】&P道具计数",
        "ゆめみごこち": "★T0免费补干劲续航",
        "みんな大好き": "★T0免费好印象得分(不耗行动点)",
        "私が最強！": "★T0 Legend爆分+好印象12+下T抽2",
        "究極スマイル": "★T0 Legend全牌好印象+6&强化+100%",
        "エクセレント♪": "★T0 Legend5T抽牌&每T白打2张R卡",
        "ラブリーウインク": "★T1优质可重复好印象输出+推触发计数",
        "対象、ロックオン": "★T0好印象+100%强化&条件持续抽牌",
        "止められない想い": "★T1免费双加干劲+好印象限1卡",
        "夢色リップ": "★T1好印象×1.1雪球+补干劲",
        "本番前夜": "★T1开局固定上手+好印象+干劲",
    },
    caution_cards={
        152: "🚫角色定位不匹配·禁留牌！清零元気护盾的纯元気爆破卡,严禁收入或留在美铃牌库中",
        153: "🚫角色定位不匹配·禁留牌！严禁收入或留在美铃牌库中",
        426: "🚫角色定位不匹配·禁留牌！红血耗4且依赖未除外眠気,与美铃M卡好印象体系冲突,严禁留牌",
        427: "🚫角色定位不匹配·禁留牌！红血耗4且依赖未除外眠気,与美铃M卡好印象体系冲突,优先换出/删除",
        306: "⚠️反协同·消耗3层好印象且主靠元气得分,会打断ガラクタロード好印象叠层!",
        307: "⚠️反协同·消耗3层好印象且偏干劲盾流,慎选",
        11: "⚠️偏干劲护盾轴(仅吃元気转化)且红血耗5,对好印象美铃收益极低",
        12: "⚠️偏干劲护盾轴(仅吃元気转化),非好印象轴组件",
        771: "⚠️偏干劲护盾爆发(吃元気倍率),非好印象轴",
        772: "⚠️偏干劲护盾爆发(吃元気倍率),非好印象轴",
        398: "⚠️消耗宝贵3干劲却只按元気得分,与好印象轴核心牌抢干劲资源!",
        399: "⚠️消耗干劲按元気得分,容易卡死『輝くキミへ/オトメゴコロ』干劲",
        726: "⚠️偏赤裸々干劲扣血回气体系,对好印象美铃意义不大(远逊究極スマイル/エクセレント♪)",
    },
    caution_card_names={
        "届いて！": "🚫禁留牌·角色定位不匹配(清零护盾),严禁收入或留在牌库",
        "キセキの魔法": "🚫禁留牌·角色定位不匹配(红血+依赖眠気),严禁收入或留在牌库",
        "あのときの約束": "⚠️反协同·会直接扣3层好印象! 严重拖累美铃叠层",
        "気分転換": "⚠️干劲护盾转化卡·对好印象轴收益极低",
        "せのびの魔法": "⚠️干劲护盾爆发卡·非好印象核心",
        "きらきら紙吹雪": "⚠️反协同·浪费稀缺干劲打元気伤害,抢『オトメゴコロ/輝くキミへ』干劲",
    },
)


_SUISAI_UME_GUIDE = PIdolGuide(
    p_idol_id=136,
    idol_name="花海 佑芽",
    title="真っ白いページと水彩の主人公",
    rarity="SSR",
    plan="logic",
    recommended_effect="motivation",
    source_urls=(
        "https://seesaawiki.jp/gakumasu/",
    ),
    unique_p_item={
        "name": "夢を創るステッキ",
        "trigger": "每3回合结束时（第3/6/9/12回合末，限3次）",
        "effect": (
            "造成当前元气×110%（强化+为130%）的参数得分并清空元气归0，下回合额外抽1张牌；"
            "清零的元气计入累计消耗元气(consumedGenki)，供固有卡次回合0元气时返还80%！"
        ),
    },
    signature_cards=(
        "#799/#800 あたしがいるよ / +(Mental·限1): 元气+2, 干劲+1(+3); "
        "后续3次在回合开始且元气==0时，回复【累计消耗元气(consumedGenki) × 80%】的元气！尽早打出以在T3/T6/T9清零后次回合回满。",
        "#854 証明し続ける一番星(S3 H.I.F专属LG卡·自动配发·Mental·限1·开局手牌): "
        "+1使用次数，此后5回合内每次直接增加干劲后，随机将牌堆/弃牌中1张M卡置顶于牌库最上方。",
    ),
    core_mechanics=(
        "【核心引擎：干劲叠元气 + 每3回合『夢を創るステッキ』引爆清零 + consumedGenki 80%返还】"
        "每第3回合末（T3/T6/T9）将全部元气转化为110%(强化+130%)得分并归0，同时把清零数值累计入 consumedGenki；"
        "搭配固有卡『あたしがいるよ(+ )』在第4/7/10回合初元气为0时瞬间返还80%累计消耗元气再叠新回合爆发！",
        "【#854 証明し続ける一番星 控顶加速】S3专属LG卡入场即上手且自带+1使用次数，开局配合高效率干劲M卡连续把弃牌/牌库中的M卡置于牌堆顶加快运转。",
    ),
    battle_pacing_rules=(
        "第1回合起手有 #854《証明し続ける一番星》必优先打出（不耗行动点），再接直接加干劲的M卡触发M卡置顶，尽快铺出 #799/#800《あたしがいるよ(+)》。",
        "T3/T6/T9 清零回合尽全力拉高元气吃满 110%/130% 转化；若中盘配合《届いて！》主动清零也能反哺 consumedGenki，并在下一回合靠《あたしがいるよ》回气。",
    ),
    drink_synergies=(
        "特製ハツボシエキス(#25)：留给中后期高元气转化大招（如《せのびの魔法+》《届いて！+》）。",
        "烏龍茶(#2) / ホットコーヒー / センブリソーダ：帮助前3回合快速拉起干劲与元气底数。",
    ),
    anti_synergies=(
        "避免混入依赖『好印象』却不产干劲/元气的纯好印象组件。",
    ),
    recommended_cards={
        799: "★固有核心(M·限1)·元气0时回合初返还80%累计消耗元气(限3次)",
        800: "★固有核心+(M·限1)·干劲+3并触发80%消耗元气返还",
        854: "★S3专属LG(M·限1·开局上手)·+1使用数并在5回合直接加干劲时将M卡置顶牌库",
        782: "★消耗元气爆发(A·LG)·score+=consumedGenki×4,完美契合136",
        152: "★元气转化兼主动清零(A·限1)·清零计入consumedGenki并触发回气",
        153: "★元气转化兼主动清零+(A·限1)",
        771: "★核心元气倍伤(A·限1)",
        772: "★核心元气倍伤+(A·限1)",
    },
    recommended_card_names={
        "あたしがいるよ": "★136固有核心·consumedGenki 80%返还",
        "証明し続ける一番星": "★S3专属LG·5T直接加干劲时M卡置顶牌库",
        "輝きの到達点": "★consumedGenki×4神契合爆发",
    },
    caution_cards={},
    caution_card_names={},
)


P_IDOL_GUIDE_REGISTRY: Dict[int, PIdolGuide] = {
    _SEKIRARA_SENA_GUIDE.p_idol_id: _SEKIRARA_SENA_GUIDE,
    _GARAKUTA_MISUZU_GUIDE.p_idol_id: _GARAKUTA_MISUZU_GUIDE,
    _SUISAI_UME_GUIDE.p_idol_id: _SUISAI_UME_GUIDE,
}

# Authoritative per-idol 3D stat specialty order: (primary, secondary, tertiary) from high -> low.
# Keyed by `p_idols.idol_id` (character ID).
IDOL_STAT_SPECIALTY_ORDER: Dict[int, Tuple[str, str, str]] = {
    1: ("vocal", "dance", "visual"),    # 花海 咲季: Vo -> Da -> Vi
    2: ("vocal", "visual", "dance"),    # 月村 手毬: Vo -> Vi -> Da
    3: ("dance", "visual", "vocal"),    # 藤田 ことね: Da -> Vi -> Vo
    4: ("visual", "vocal", "dance"),    # 有村 麻央: Vi -> Vo -> Da
    5: ("vocal", "dance", "visual"),    # 葛城 リーリヤ: Vo -> Da -> Vi
    6: ("vocal", "visual", "dance"),    # 倉本 千奈: Vo -> Vi -> Da
    7: ("dance", "visual", "vocal"),    # 紫雲 清夏: Da -> Vi -> Vo
    8: ("visual", "dance", "vocal"),    # 篠澤 広: Vi -> Da -> Vo
    9: ("visual", "dance", "vocal"),    # 姫崎 莉波: Vi -> Da -> Vo
    10: ("dance", "vocal", "visual"),   # 花海 佑芽: Da -> Vo -> Vi
    11: ("vocal", "visual", "dance"),   # 十王 星南: Vo -> Vi -> Da
    12: ("vocal", "visual", "dance"),   # 秦谷 美鈴: Vo -> Vi -> Da
    13: ("vocal", "dance", "visual"),   # 雨夜 燕: Vo -> Da -> Vi
}


def get_idol_stat_order(
    p_idol_id: Optional[int] = None,
    *,
    db: Optional[Any] = None,
) -> Tuple[str, str, str]:
    """Return `(primary, secondary, tertiary)` stat keys (high -> low) for a P-Idol."""
    if p_idol_id is not None:
        pid = int(p_idol_id)
        if db is not None and hasattr(db, "get_p_idol_by_id"):
            try:
                p_row = db.get_p_idol_by_id(pid)
                if p_row and p_row.get("idol_id") is not None:
                    char_id = int(p_row["idol_id"])
                    if char_id in IDOL_STAT_SPECIALTY_ORDER:
                        return IDOL_STAT_SPECIALTY_ORDER[char_id]
            except Exception:
                pass
        if pid in IDOL_STAT_SPECIALTY_ORDER:
            return IDOL_STAT_SPECIALTY_ORDER[pid]
    return ("vocal", "visual", "dance")


def get_default_idol_stats(
    p_idol_id: Optional[int] = None,
    *,
    db: Optional[Any] = None,
    high: int = 2400,
    mid: int = 1800,
    low: int = 1400,
) -> Dict[str, int]:
    """Build default `{vocal, dance, visual}` stats aligned with the P-Idol's specialty order."""
    prim, sec, tert = get_idol_stat_order(p_idol_id, db=db)
    return {prim: int(high), sec: int(mid), tert: int(low)}

# Maps signature / P-Idol-specific card IDs to their P-Idol ID for auto-detection.
_SIGNATURE_CARD_TO_P_IDOL: Dict[int, int] = {
    755: 126,
    756: 126,
    757: 126,
    758: 126,
    876: 126,
    783: 130,
    784: 130,
    853: 130,
    799: 136,
    800: 136,
    854: 136,
    829: 123,
    830: 115,
    831: 110,
    834: 137,
    835: 114,
    836: 102,
    837: 97,
    838: 120,
    875: 106,
    919: 163,
    920: 163,
    921: 163,
    922: 163,
}


def get_p_idol_guide(p_idol_id: Optional[int]) -> Optional[PIdolGuide]:
    """Return the registered strategy guide for a P-Idol ID, if available."""
    if p_idol_id is None:
        return None
    return P_IDOL_GUIDE_REGISTRY.get(int(p_idol_id))


def resolve_p_idol_id(
    plan: Optional[str] = None,
    p_idol_id: Optional[int] = None,
    deck_ids: Optional[Sequence[int]] = None,
    hand_cards: Optional[Sequence[Any]] = None,
    default_logic_to_sekirara: bool = True,
) -> Optional[int]:
    """
    Resolve active P-Idol ID from explicit setting, signature cards in deck/hand,
    or plan default (currently only [赤裸々] 十王星南 #126 is registered for Logic).
    """
    if p_idol_id is not None:
        return int(p_idol_id)

    for cid in deck_ids or ():
        mapped = _SIGNATURE_CARD_TO_P_IDOL.get(int(cid))
        if mapped is not None:
            return mapped

    for c in hand_cards or ():
        cid = c.get("card_id") if isinstance(c, dict) else getattr(c, "card_id", None)
        if cid is not None:
            mapped = _SIGNATURE_CARD_TO_P_IDOL.get(int(cid))
            if mapped is not None:
                return mapped

    if default_logic_to_sekirara and (plan or "").lower() == "logic":
        return 126
    return p_idol_id


def resolve_card_family_meta(
    meta_map: Optional[Dict[Any, Dict[str, Any]]],
    card_id: Optional[int],
    *,
    db: Optional[Any] = None,
    row: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve card hooks or customizations across exact card_id and its base/upgraded siblings."""
    if not meta_map or card_id is None:
        return {}
    try:
        cid = int(card_id)
    except (TypeError, ValueError):
        return {}
    if cid <= 0:
        return {}

    if row is None and db is not None and hasattr(db, "get_card_by_id"):
        row = db.get_card_by_id(cid)

    candidates: List[int] = [cid]
    if row:
        for fk in ("base_id", "upgraded_id"):
            fv = row.get(fk)
            if fv is not None:
                try:
                    fi = int(fv)
                except (TypeError, ValueError):
                    fi = 0
                if fi > 0 and fi not in candidates:
                    candidates.append(fi)
        base_v = row.get("base_id")
        if base_v and db is not None and hasattr(db, "get_card_by_id"):
            try:
                brow = db.get_card_by_id(int(base_v))
                upg_v = brow.get("upgraded_id") if brow else None
                if upg_v is not None:
                    ui = int(upg_v)
                    if ui > 0 and ui not in candidates:
                        candidates.append(ui)
            except (TypeError, ValueError):
                pass

    merged: Dict[str, Any] = {}
    for cand in reversed(candidates):
        raw = meta_map.get(cand)
        if not isinstance(raw, dict):
            raw = meta_map.get(str(cand))
        if isinstance(raw, dict) and raw:
            merged.update(raw)
    return merged


def _get_legacy_163_memory_meta(
    scenario_id: Optional[str],
    p_idol_id: Optional[int],
    card_id: Optional[int],
    battle_played_counts: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Built-in H.I.F 選抜メモリー defaults for #163 [ガラクタロード] 秦谷 美鈴."""
    if card_id is None:
        return {}
    sc = (scenario_id or "").lower()
    if sc not in ("hif_honsen", "hif"):
        return {}
    if p_idol_id is not None and int(p_idol_id) != 163:
        return {}
    cid = int(card_id)
    if cid in (296, 297, 298, 299, 128, 129):
        return {
            "exiles_trouble": True,
            "grant_extra_use_now": False,
            "tag": " 【🧹メモリー効果：打出后免费将1张《眠気》移至除外】",
            "archive_note": "【編入メモリー効果】打出后免费将1张《眠気》移动至【除外】区（自动清除废牌+推进除外计数+加速牌库压缩，手牌有《眠気》时勿浪费行动点打眠気！）",
        }
    if cid in (136, 137):
        counts = battle_played_counts or {}
        played = int(counts.get(136, 0) or 0) + int(counts.get(137, 0) or 0)
        if played == 0:
            return {
                "exiles_trouble": False,
                "grant_extra_use_now": True,
                "extra_use_limit": 1,
                "extra_use_remaining": 1,
                "draw_on_play": 0,
                "tag": " 【⚡メモリー首次免费再动(+1使用数)：不耗行动点，先打进除外！】",
                "archive_note": "【編入メモリー/カスタム効果】每场战斗首次打出附带『スキルカード使用数追加+1』(1次免费再动·不耗行动点)，务必在普通耗点卡之前先打出送入【除外】压缩牌库！",
            }
        return {
            "exiles_trouble": False,
            "grant_extra_use_now": False,
            "extra_use_limit": 1,
            "extra_use_remaining": 0,
            "draw_on_play": 0,
            "tag": " [メモリー再动已用]",
            "archive_note": "【編入メモリー/カスタム効果】每场首次打出享1次免费再动",
        }
    if cid == 23:
        return {
            "exiles_trouble": False,
            "grant_extra_use_now": False,
            "avoid_manual_play": True,
            "tag": " 【⛔勿主动打出：留给止められない想い/オトメゴコロ/ワクワク免费移入除外！】",
            "archive_note": "干扰卡，勿主动消耗行动点打出，交由メモリー效果免费除外",
        }
    return {}


def get_memory_card_custom_meta(
    scenario_id: Optional[str],
    p_idol_id: Optional[int],
    card_id: Optional[int],
    battle_played_counts: Optional[Dict[int, int]] = None,
    *,
    card_memory_hooks: Optional[Dict[Any, Dict[str, Any]]] = None,
    db: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return H.I.F 選抜メモリー / カスタム per-card mechanics from configured hooks or legacy idol rules."""
    if card_id is None:
        return {}
    cid = int(card_id)
    row = db.get_card_by_id(cid) if (db is not None and hasattr(db, "get_card_by_id")) else None
    base_id = int(row.get("base_id") or cid) if row else cid
    fam_ids = {cid, base_id}
    if row and row.get("upgraded_id"):
        fam_ids.add(int(row["upgraded_id"]))

    h_meta = resolve_card_family_meta(card_memory_hooks, cid, db=db, row=row)
    legacy = _get_legacy_163_memory_meta(scenario_id, p_idol_id, cid, battle_played_counts)
    if not h_meta:
        return legacy

    counts = battle_played_counts or {}
    played = sum(int(counts.get(k, counts.get(str(k), 0)) or 0) for k in fam_ids)

    has_extra_key = ("free_extra_use_first_play" in h_meta) or ("extra_use_limit" in h_meta)
    has_exile_key = "exiles_trouble_on_play" in h_meta
    draw_on_play = max(0, int(h_meta.get("draw_on_play") or legacy.get("draw_on_play") or 0))

    if has_extra_key:
        raw_free = h_meta.get("free_extra_use_first_play")
        extra_lim = int(h_meta.get("extra_use_limit") or (1 if raw_free else 0))
        if raw_free is False and int(h_meta.get("extra_use_limit") or 0) <= 0:
            extra_lim = 0
        rem_extra = max(0, extra_lim - played)
        grant_now = rem_extra > 0
    else:
        extra_lim = int(legacy.get("extra_use_limit") or 0)
        rem_extra = int(legacy.get("extra_use_remaining") or 0)
        grant_now = bool(legacy.get("grant_extra_use_now"))

    if has_exile_key:
        exiles = bool(h_meta.get("exiles_trouble_on_play")) and base_id != 426 and played == 0
    else:
        exiles = bool(legacy.get("exiles_trouble"))

    if not (has_extra_key or has_exile_key or draw_on_play > 0):
        return legacy

    tags: List[str] = []
    archive_notes: List[str] = []
    if extra_lim > 0:
        draw_seg = f"+抽{draw_on_play}张" if draw_on_play > 0 else ""
        if grant_now:
            lim_note = f"(剩余{rem_extra}/{extra_lim})" if extra_lim > 1 else ""
            tags.append(f" 【⚡メモリー免费再动(+1使用数){draw_seg}{lim_note}：不耗行动点，先打进除外！】")
        else:
            tags.append(f" [メモリー再动已用({min(played, extra_lim)}/{extra_lim})]")
        archive_notes.append(
            f"【編入メモリー/カスタム効果】每场前{extra_lim}次打出附带『スキルカード使用数追加+1』{draw_seg}(免费再动·不耗行动点)"
        )
    elif draw_on_play > 0:
        tags.append(f" 【🃏メモリー抽{draw_on_play}张】")
        archive_notes.append(f"【編入メモリー/カスタム効果】打出时额外抽{draw_on_play}张")

    if exiles or (has_exile_key and bool(h_meta.get("exiles_trouble_on_play")) and base_id != 426):
        tags.append(" 【🧹メモリー効果：打出后免费将1张《眠気》移至除外】")
        archive_notes.append("【編入メモリー効果】打出后免费将1张《眠気》移动至【除外】区")

    return {
        "exiles_trouble": exiles,
        "grant_extra_use_now": grant_now,
        "extra_use_limit": extra_lim,
        "extra_use_remaining": rem_extra,
        "draw_on_play": draw_on_play,
        "tag": "".join(tags) or str(legacy.get("tag") or ""),
        "archive_note": " | ".join(archive_notes) or str(legacy.get("archive_note") or ""),
    }


def get_card_guide_annotation(
    guide: Optional[PIdolGuide],
    card_id: Optional[int] = None,
    card_name: str = "",
) -> str:
    """Return a concise strategy tag when a card matches the idol's recommended or caution list."""
    if guide is None:
        return ""
    if card_id is not None:
        cid = int(card_id)
        if cid in guide.caution_cards:
            return f" [{guide.caution_cards[cid]}]"
        if cid in guide.recommended_cards:
            return f" [{guide.recommended_cards[cid]}]"
    clean_name = (card_name or "").replace("+", "").strip()
    if clean_name:
        for nm, note in guide.caution_card_names.items():
            if nm in clean_name:
                return f" [{note}]"
        for nm, note in guide.recommended_card_names.items():
            if nm in clean_name:
                return f" [{note}]"
    return ""


def format_idol_system_snippet(guide: PIdolGuide) -> str:
    """Compact-yet-complete P-Idol tactical rules appended to System Prompts."""
    p_item = guide.unique_p_item
    if guide.p_idol_id == 126:
        return (
            f"【P卡专属操作指导 · [{guide.title}] {guide.idol_name} (#{guide.p_idol_id} · {guide.plan.upper()}/干劲轴)】\n"
            f"- 固有P道具《{p_item.get('name', '')}》：{p_item.get('trigger', '')} -> {p_item.get('effect', '')}\n"
            f"- 专属循环：第1手有《私を超えて》/《+》必先打出(+1使用数且自身零干劲/零元气，随后靠干劲M卡+元气M卡养引擎，每2张元気效果卡往牌库顶生1张默认未强化《私を超えて(翔)》)；(翔)需【元気>=15且体力>1】，不耗行动点且抽1牌，可在高元气时连续自抽爆破。\n"
            f"- 选卡/饮料与避坑：抓高干劲(本番前夜/ワクワク/わたしらしい色/最強パフォーマー)与高效元气(えいえいおー/ドラマチックホリデー/成長痛)及终局转化(せのびの魔法/届いて！/気分転換)；"
            f"严禁误抓无用好印象杂牌；已有《最強パフォーマー》等扣血回气时避开把(翔)降至0耗血的减耗卡(`さっぱりひといき`)；"
            f"`特製ハツボシエキス`仅留到末盘Active大招(せのびの魔法/届いて！/私を超えて(翔))回合喝，`フレッシュビネガー`留手里多张(翔)同强。"
        )
    if guide.p_idol_id == 163:
        return (
            f"【P卡专属操作指导 · [{guide.title}] {guide.idol_name} (#{guide.p_idol_id} · {guide.plan.upper()}/好印象·再演除外循环轴)】\n"
            f"- 固有P道具《{p_item.get('name', '')}》：{p_item.get('trigger', '')} -> {p_item.get('effect', '')}\n"
            f"- 編入メモリー效果：《止められない想い》《オトメゴコロ》《ワクワクが止まらない》打出免费将1张《眠気》移至【除外】(绝不手打《眠気》！)；《やる気は満点》每场首次打出免费再动(+1使用数)，务必先打进除外。\n"
            f"- 高速除外压缩→终盘循环：开局前1~3T尽早打出《ありのままのわたし》(#919/#920)激活【再演】(+50%×3)，全速把【限1】卡送进除外压缩牌库(绝不无故SKIP)，终盘靠《オトメゴコロ+》→《みんな大好き+》→《認めてくれるひと+》(除外>=11打280%且回弃牌)无限循环。\n"
            f"- 药水与商店铁律：《特製ハツボシエキス》(#25)头号优先配《びしっとキメ顔+》(#778，双发4T+400%强化+200%双倍追击；次选《認めてくれるひと+》/《キセキの魔法+》；无适配主动A卡严禁空喝)；进商店尽量不加新卡，优先买强力饮料、走『セレクトチェンジ/チェンジ』换牌，余钱投『強化/カスタマイズ』强化自定义主力卡！"
        )
    axis = "好印象轴" if guide.recommended_effect == "goodImpressionTurns" else "干劲轴"
    mechs = "；".join(guide.core_mechanics[:2])
    return (
        f"【P卡专属操作指导 · [{guide.title}] {guide.idol_name} (#{guide.p_idol_id} · {guide.plan.upper()}/{axis})】\n"
        f"- 固有P道具《{p_item.get('name', '')}》：{p_item.get('trigger', '')} -> {p_item.get('effect', '')}\n"
        f"- 核心机制：{mechs}"
    )


def format_idol_session_guide_block(guide: PIdolGuide) -> str:
    """Rich structured P-Idol guide block sent once during session initialization (and detailed mode)."""
    sigs = "\n".join(f"  - {line}" for line in guide.signature_cards)
    mechs = "\n".join(f"  - {line}" for line in guide.core_mechanics)
    pacing = "\n".join(f"  {idx}. {line}" for idx, line in enumerate(guide.battle_pacing_rules, 1))
    drinks = "\n".join(f"  - {line}" for line in guide.drink_synergies)
    avoids = "\n".join(f"  - {line}" for line in guide.anti_synergies)
    sources = " | ".join(guide.source_urls[:3])
    p_item = guide.unique_p_item
    return (
        f"【P卡角色专属攻略指导 · {guide.rarity} [{guide.title}] {guide.idol_name} (#{guide.p_idol_id})】\n"
        f"(攻略来源: {sources})\n"
        f"- 固有P道具《{p_item.get('name', '')}》：{p_item.get('trigger', '')} -> {p_item.get('effect', '')}\n"
        f"【专属卡牌与衍生机制】\n{sigs}\n"
        f"【核心运营铁律】\n{mechs}\n"
        f"【对局实战节奏】\n{pacing}\n"
        f"【饮料搭配特化】\n{drinks}\n"
        f"【避坑与反协同警告】\n{avoids}"
    )


def format_idol_turn_nudge(
    guide: Optional[PIdolGuide],
    obs: Any,
    remaining_deck_cids: Optional[Sequence[int]] = None,
    exhaust_count: int = 0,
) -> str:
    """Produce a concise 1-line dynamic reminder in battle only when an idol-specific condition applies."""
    if guide is None:
        return ""
    notes: List[str] = []
    hand_cids = {int(c.card_id) for c in (getattr(obs, "cards", None) or [])}
    drink_ids = {int(d.drink_id) for d in (getattr(obs, "drinks", None) or []) if getattr(d, "playable", True)}
    turn = int(getattr(obs, "turn", 1) or 1)
    genki = int(getattr(obs, "genki", 0) or 0)
    stamina = int(getattr(obs, "stamina", 0) or 0)
    turns_rem = int(getattr(obs, "turns_remaining", 10) or 10)
    buffs = getattr(obs, "buffs", None) or {}
    good_imp = int(buffs.get("goodImpressionTurns", 0) or buffs.get("good_impression", 0) or 0)

    if guide.p_idol_id == 126:
        if 755 in hand_cids or 756 in hand_cids:
            notes.append("手牌有固有启动《私を超えて》：不耗行动点且后续靠元気卡×2生《翔》，优先第1手打出")
        if 757 in hand_cids or 758 in hand_cids:
            if genki < 15:
                notes.append(f"手牌有《私を超えて(翔)》但当前元气{genki}<15不可打，先出元気卡或饮《烏龍茶》凑满15解锁")
            elif stamina <= 1:
                notes.append("手牌有《私を超えて(翔)》但体力<=1无法支付固定扣血1")
            else:
                notes.append(
                    "《私を超えて(翔)》已解锁(+1使用数+抽1牌)：注意剩余出牌=1时打普通卡会直接结束回合，本回合要打(翔)必须【先打(翔)】(或先喝饮料/先打自带+1使用数的卡)再出其他牌！"
                )
        if turn % 2 == 1 and 15 <= genki < 30 and turns_rem >= 3:
            notes.append(f"下回合为偶数回合，距触发『最初の鳥』回血门槛(元气30)仅差{30 - genki}点")
        if 25 in drink_ids and not (hand_cids & {757, 758, 771, 772, 152, 153, 304, 305}):
            notes.append("《特製ハツボシエキス》仅对本回合下张Active非Legend大招有效，当前缺大招时切勿白喝")
        if not notes:
            return ""
        return "- 🎀 [赤裸々星南提示]: " + "；".join(notes[:2])

    if guide.p_idol_id == 163:
        if 919 in hand_cids or 920 in hand_cids:
            notes.append("手牌有固有启动《ありのままのわたし》：尽早打出进【除外】以激活【再演】+50%×3好印象强化！")
        if 921 in hand_cids or 922 in hand_cids:
            has_25 = 25 in drink_ids
            if exhaust_count >= 11:
                extra_25 = "，若《びしっとキメ顔》已不在手/已用且手中《特製ハツボシエキス》未喝可先喝再双发核爆！" if has_25 else ""
                notes.append(
                    f"【除外已达{exhaust_count}张≥11】《認めてくれるひと》满额+280%/+170%爆发已解锁(可重复回弃牌循环){extra_25}"
                )
            elif exhaust_count >= 7:
                notes.append(
                    f"【除外{exhaust_count}/11张】《認めてくれるひと》已解锁+5好印象(2段加层)，再凑{11 - exhaust_count}张限1进除外即可解锁280%爆分"
                )
            else:
                notes.append(
                    f"【除外仅{exhaust_count}/7张】《認めてくれるひと》仅+3好印象，中前盘先多打限1卡推除外到7/11并压缩牌库"
                )
        if 777 in hand_cids or 778 in hand_cids:
            if good_imp >= 10:
                if 25 in drink_ids:
                    notes.append(
                        f"🔥《特製ハツボシエキス》+《びしっとキメ顔》王炸组合同手就绪(好印象{good_imp}≥10)！请【先喝特製ハツボシエキス】再打《びしっとキメ顔》直接叠4回合+400%强化与双倍追击！"
                    )
                else:
                    notes.append(
                        f"《びしっとキメ顔》已解锁(好印象{good_imp}≥10)：开后直接加好印象卡每次追伤100%，同回合可先/后配《オトメゴコロ/認めてくれるひと》吃满追击！"
                    )
            else:
                notes.append(f"《びしっとキメ顔》需好印象≥10(当前{good_imp})，先叠好印象解锁")
        if 25 in drink_ids:
            playable_non_l_active = [
                c
                for c in (getattr(obs, "cards", None) or [])
                if getattr(c, "playable", False)
                and int(getattr(c, "card_id", 0) or 0) != 781
                and getattr(c, "rarity", "N") != "L"
                and getattr(c, "type", "active") == "active"
            ]
            if not playable_non_l_active:
                notes.append("⛔当前手牌无可用非Legend主动(A)卡，《特製ハツボシエキス》仅本回合1T有效，严禁空喝浪费！")
            elif not (777 in hand_cids or 778 in hand_cids) and turns_rem > 2 and not (hand_cids & {921, 922, 426, 427} and exhaust_count >= 11):
                notes.append("《特製ハツボシエキス》(#25)头号推荐留配《びしっとキメ顔+》(次选除外≥11《認めてくれるひと+》/《キセキの魔法+》)，勿早用在低分卡上")
        if 23 in hand_cids and (hand_cids & {296, 297, 298, 299, 128, 129}):
            notes.append("手牌有《眠気》(#23)：勿手打浪费行动点！打《止められない想い/オトメゴコロ/ワクワクが止まらない》即可免费将其移至除外")
        if not notes:
            return ""
        return "- 🎀 [ガラクタ美鈴提示]: " + "；".join(notes[:3])

    return ""
