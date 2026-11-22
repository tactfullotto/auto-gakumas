"""
Unit & Integration Tests for Single-Produce Persistent Multi-Turn Session Mode:
- Once-per-Produce system + bootstrap archive prompt (cards, drinks, P-items, 13-week roadmap)
- Incremental delta prompts (events, deck diffs, newly discovered card/drink/item archives)
- AgentDecisionMaker multi-turn conversation history preservation, retry isolation, & context trimming
- GakumasRunner rule_skip bootstrap deferral and timeout-safe sync commits
"""

from unittest.mock import MagicMock, patch
import numpy as np
import openai
import pytest

from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.agent.prompts import get_session_system_prompt
from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.base_controller import BaseController
from demo.maa_bridge.runner import GakumasRunner
from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation


@pytest.fixture
def shared_db():
    db = GakumasDB("gakumas_library.db")
    yield db
    db.close()


class _StubLiveController(BaseController):
    def __init__(self):
        self._connected = True
        self.clicks = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def screencap(self):
        return np.zeros((1600, 720, 3), dtype=np.uint8)

    def click(self, x: int, y: int) -> bool:
        self.clicks.append((x, y))
        return True

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> bool:
        return True


def test_session_system_prompt_covers_core_rules_and_plans():
    for plan in ("sense", "logic", "anomaly"):
        sp = get_session_system_prompt(plan=plan, scenario_id="hajime_regular")
        assert "全培育持久化单会话" in sp
        assert "增量" in sp
        assert "CLEARまで" in sp
        assert "研修案内" in sp
        assert "choice_id" in sp
    logic_sp = get_session_system_prompt(plan="logic")
    assert "私を超えて" in logic_sp


def test_prompt_builder_session_bootstrap_then_incremental_delta(shared_db):
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db, plan="logic", p_idol_id=126)
    builder = PromptBuilder(db=shared_db, compact=False)
    bbox = BoundingBox(xmin=100, ymin=1100, xmax=260, ymax=1350)

    # Step 1: Bootstrap (Week 1 battle Turn 1 with starter signature #756 + #13)
    obs1 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        scenario_id="hajime_regular",
        week=1,
        week_activity="Vo.通常レッスン (CLEARまで 25)",
        turn=1,
        turns_remaining=8,
        stamina=31,
        max_stamina=31,
        genki=0,
        score=0,
        target_score=25,
        cards=[
            RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=13, name="表現の基本", bbox=bbox),
        ],
        drinks=[],
    )
    boot_prompt, boot_actions, boot_meta = builder.build_session_prompt(obs1, tracker, commit=True)
    assert boot_meta["is_bootstrap"] is True
    assert tracker.session_bootstrapped is True
    assert "唯一开局建档" in boot_prompt
    assert "剧本全周目路线表" in boot_prompt
    assert "初始卡库及衍生卡详细效果" in boot_prompt
    # Both #756 and its derived #758 token + starter #13 should be pre-archived once
    assert 756 in tracker.llm_synced_card_ids
    assert 758 in tracker.llm_synced_card_ids
    assert 13 in tracker.llm_synced_card_ids
    assert any("私を超えて+" in a.description for a in boot_actions)

    # Simulate playing #756 and earning new card #124 + new drink #9 in Week 2
    tracker.record_session_event("上步AI打出《私を超えて+》(#756)")
    tracker.add_card_to_deck(124)

    obs2 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        scenario_id="hajime_regular",
        week=3,
        week_activity="Vo.通常レッスン (CLEARまで 80)",
        turn=2,
        turns_remaining=7,
        stamina=24,
        max_stamina=31,
        genki=9,
        score=10,
        target_score=70,
        cards=[
            RecognizedCard(card_id=124, name="幸せな時間", bbox=bbox),
            RecognizedCard(card_id=13, name="表現の基本", bbox=bbox),
        ],
        drinks=[RecognizedDrink(drink_id=9, name="ミックススムージー", bbox=bbox)],
    )
    delta_prompt, _, _ = builder.build_session_prompt(obs2, tracker, commit=True)
    assert "增量状态同步" in delta_prompt
    # Static roadmap & already-archived #13 full effect must NOT repeat
    assert "剧本全周目路线表" not in delta_prompt
    assert "消耗DSL:" in delta_prompt  # Only for newly added card #124!
    assert "幸せな時間" in delta_prompt
    assert "新卡牌档案收录" in delta_prompt
    assert "新P饮料档案收录" in delta_prompt
    assert "ミックススムージー" in delta_prompt
    assert "卡库变动" in delta_prompt
    assert 124 in tracker.llm_synced_card_ids
    assert 9 in tracker.llm_synced_drink_ids

    # Step 3: Next turn in same battle with same known card #124 and drink #9 -> no new archives!
    obs3 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        scenario_id="hajime_regular",
        week=3,
        week_activity="Vo.通常レッスン (CLEARまで 70)",
        turn=3,
        turns_remaining=6,
        stamina=19,
        max_stamina=31,
        genki=9,
        score=20,
        target_score=60,
        cards=[RecognizedCard(card_id=124, name="幸せな時間", bbox=bbox)],
        drinks=[RecognizedDrink(drink_id=9, name="ミックススムージー", bbox=bbox)],
    )
    delta_prompt2, _, meta3 = builder.build_session_prompt(obs3, tracker, commit=True)
    assert meta3["is_bootstrap"] is False
    assert "新卡牌档案收录" not in delta_prompt2
    assert "新P饮料档案收录" not in delta_prompt2
    assert "完整效果见前序建档" in delta_prompt2


def test_agent_persistent_session_history_retry_isolation_and_trimming():
    maker = AgentDecisionMaker(
        config=AgentConfig(api_key="sk-test", max_retries=1, session_mode=True)
    )
    bbox = BoundingBox(xmin=0, ymin=0, xmax=50, ymax=50)
    from demo.schemas import LegalAction

    actions = [
        LegalAction(choice_id=0, action_type="play_card", description="A0", bbox=bbox),
        LegalAction(choice_id=1, action_type="end_turn", description="SKIP", bbox=bbox),
    ]
    obs = ScreenObservation(scene_type="battle_turn", plan="logic", scenario_id="hajime_regular")

    # Turn 1: malformed output first, then valid retry -> retry chatter must NOT enter session_messages
    bad_resp = MagicMock(
        choices=[MagicMock(message=MagicMock(content="oops not json"))],
        usage=MagicMock(prompt_tokens=100, completion_tokens=5),
    )
    ok1 = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"choice_id": 0, "reason": "先手固有"}'))],
        usage=MagicMock(prompt_tokens=120, completion_tokens=18),
    )
    ok2 = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"choice_id": 1, "reason": "结束"}'))],
        usage=MagicMock(prompt_tokens=150, completion_tokens=12),
    )

    with patch.object(maker.client.chat.completions, "create", side_effect=[bad_resp, ok1, ok2]) as mock_create:
        r1 = maker.decide("BOOTSTRAP PROMPT", actions, obs=obs)
        assert r1.source == "llm"
        assert r1.choice_id == 0
        # System + user_1 + clean assistant_1 = 3 messages (failed attempt not stored)
        assert len(maker.session_messages) == 3
        assert maker.session_messages[0]["role"] == "system"
        assert "全培育持久化单会话" in maker.session_messages[0]["content"]
        assert "oops" not in maker.session_messages[2]["content"]

        # rule_skip does NOT touch session_messages
        r_skip = maker.decide("IGNORED", [actions[0]], obs=obs)
        assert r_skip.source == "rule_skip"
        assert len(maker.session_messages) == 3

        # Turn 2 sends accumulated history [system, u1, a1, u2]
        r2 = maker.decide("DELTA TURN 2", actions, obs=obs)
        assert r2.choice_id == 1
        second_call_msgs = mock_create.call_args_list[2].kwargs["messages"]
        assert [m["role"] for m in second_call_msgs] == ["system", "user", "assistant", "user"]
        assert second_call_msgs[1]["content"] == "BOOTSTRAP PROMPT"
        assert second_call_msgs[3]["content"] == "DELTA TURN 2"
        assert len(maker.session_messages) == 5

    # Network failure leaves session_messages untouched
    with patch.object(maker.client.chat.completions, "create", side_effect=RuntimeError("timeout")):
        r_fb = maker.decide("FAILED DELTA", actions, obs=obs)
        assert r_fb.source == "fallback"
        assert len(maker.session_messages) == 5

    # Context-overflow trim preserves bootstrap + recent tail
    for i in range(10):
        maker._commit_session_turn(f"U{i}", 0, "ok")
    assert len(maker.session_messages) == 25
    ctx_err = openai.BadRequestError(
        message="maximum context length exceeded",
        response=MagicMock(status_code=400),
        body=None,
    )
    trimmed = maker._trim_session_on_context_overflow(
        list(maker.session_messages) + [{"role": "user", "content": "LATEST"}],
        ctx_err,
    )
    assert trimmed is not None
    assert trimmed[1]["content"] == "BOOTSTRAP PROMPT"
    assert trimmed[-1]["content"] == "LATEST"

    maker.reset_session()
    assert maker.session_messages == []


def test_runner_session_mode_defers_bootstrap_across_rule_skip_and_fallback(shared_db, monkeypatch):
    monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a, **_kw: None)

    stub = _StubLiveController()
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    agent = AgentDecisionMaker(
        config=AgentConfig(api_key="sk-test", max_retries=0, session_mode=True)
    )
    prompts_logged = []

    runner = GakumasRunner(
        mode="adb",
        controller=stub,
        agent=agent,
        db=shared_db,
        tracker=tracker,
        plan="logic",
        session_mode=True,
        on_prompt_built=lambda p, _a: prompts_logged.append(p),
    )

    bbox_card = BoundingBox(xmin=100, ymin=1100, xmax=240, ymax=1350)
    bbox_skip = BoundingBox(xmin=300, ymin=1400, xmax=500, ymax=1500)

    scene_queue = [
        # Frame 1: single-action skip screen -> rule_skip MUST NOT consume bootstrap!
        ScreenObservation(
            scene_type="skip",
            plan="logic",
            week=1,
            options=[{"choice_id": 0, "name": "点击屏幕继续", "title": "点击屏幕继续", "bbox": bbox_skip}],
        ),
        # Frame 2: real multi-choice battle screen, but network fails -> bootstrap STILL NOT committed!
        ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            week=1,
            turn=1,
            turns_remaining=8,
            stamina=31,
            max_stamina=31,
            cards=[
                RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, bbox=bbox_card),
                RecognizedCard(card_id=13, name="表現の基本", bbox=bbox_card),
            ],
        ),
        # Frame 3: LLM succeeds -> NOW bootstrap commits + includes earlier rule_skip/fallback receipts!
        ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            week=1,
            turn=1,
            turns_remaining=8,
            stamina=25,
            max_stamina=31,
            cards=[
                RecognizedCard(card_id=13, name="表現の基本", bbox=bbox_card),
                RecognizedCard(card_id=21, name="意識の基本", bbox=bbox_card),
            ],
        ),
        # Frame 4: Next step sends concise delta only!
        ScreenObservation(
            scene_type="battle_turn",
            plan="logic",
            week=1,
            turn=1,
            turns_remaining=8,
            stamina=25,
            max_stamina=31,
            genki=4,
            cards=[RecognizedCard(card_id=21, name="意識の基本", bbox=bbox_card)],
        ),
    ]
    monkeypatch.setattr(runner.scene_routing_reco, "recognize", lambda _f: scene_queue.pop(0))

    r1 = runner.run_once()
    assert r1.decision.source == "rule_skip"
    assert tracker.session_bootstrapped is False
    assert len(agent.session_messages) == 0

    with patch.object(agent.client.chat.completions, "create", side_effect=TimeoutError("net down")):
        r2 = runner.run_once()
    assert r2.decision.source == "fallback"
    assert tracker.session_bootstrapped is False
    assert len(agent.session_messages) == 0

    ok_turn1 = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"choice_id": 0, "reason": "叠元气"}'))],
        usage=MagicMock(prompt_tokens=900, completion_tokens=20),
    )
    ok_turn2 = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"choice_id": 0, "reason": "叠干劲"}'))],
        usage=MagicMock(prompt_tokens=1100, completion_tokens=18),
    )
    with patch.object(agent.client.chat.completions, "create", side_effect=[ok_turn1, ok_turn2]):
        r3 = runner.run_once()
        assert r3.decision.source == "llm"
        assert tracker.session_bootstrapped is True
        assert "唯一开局建档" in prompts_logged[-1]
        assert "系统自动跳过" in prompts_logged[-1]
        assert "兜底策略代选" in prompts_logged[-1]
        assert len(agent.session_messages) == 3

        r4 = runner.run_once()
        assert r4.decision.source == "llm"
        assert "增量状态同步" in prompts_logged[-1]
        assert "初始卡库及衍生卡详细效果" not in prompts_logged[-1]
        assert len(agent.session_messages) == 5


def test_checkpoint_interrupt_and_resume_restores_session_and_strips_temp_cards(shared_db, tmp_path, monkeypatch):
    """Verify GakumasRunner persists & recovers tracker/session/recognizer across process restarts."""
    from unittest.mock import MagicMock, patch
    import numpy as np
    from demo.agent.client import AgentConfig, AgentDecisionMaker
    from demo.core.prompt import PromptBuilder
    from demo.core.tracker import GameStateTracker
    from demo.maa_bridge.runner import GakumasRunner
    from demo.schemas import BoundingBox, RecognizedCard, ScreenObservation

    monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a, **_kw: None)

    class _StubCtrl:
        def __init__(self):
            self.is_connected = True
            self.clicks = []

        def connect(self):
            self.is_connected = True
            return True

        def disconnect(self):
            self.is_connected = False

        def screencap(self):
            return np.zeros((100, 100, 3), dtype=np.uint8)

        def click(self, x, y):
            self.clicks.append((x, y))
            return True

    ckpt_file = tmp_path / "produce_logic.json"
    bbox = BoundingBox(xmin=100, ymin=1100, xmax=240, ymax=1350)

    # Phase 1: Run until mid-W13 battle, accidentally pollute produce_deck with temp #757, then save checkpoint
    ctrl1 = _StubCtrl()
    agent1 = AgentDecisionMaker(config=AgentConfig(api_key="sk-test", max_retries=0, session_mode=True))
    tracker1 = GameStateTracker(plan="logic", p_idol_id=126)
    runner1 = GakumasRunner(
        mode="adb",
        controller=ctrl1,
        agent=agent1,
        db=shared_db,
        tracker=tracker1,
        prompt_builder=PromptBuilder(db=shared_db),
        plan="logic",
        scenario_id="hajime_regular",
        session_mode=True,
        checkpoint_path=ckpt_file,
    )
    runner1.scene_routing_reco.load_state({"last_week": 13, "last_stamina": [7, 31], "last_turns_remaining": 9})

    obs1 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=13,
        turn=3,
        turns_remaining=9,
        stamina=7,
        max_stamina=31,
        genki=12,
        score=75,
        target_score=4000,
        cards=[
            RecognizedCard(card_id=58, name="リスタート", bbox=bbox),
            RecognizedCard(card_id=757, name="私を超えて(翔)", bbox=bbox),
        ],
    )
    monkeypatch.setattr(runner1.scene_routing_reco, "recognize", lambda _f: obs1)
    ok_turn = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"choice_id": 0, "reason": "M卡叠好印象"}'))],
        usage=MagicMock(prompt_tokens=1200, completion_tokens=20),
    )
    with patch.object(agent1.client.chat.completions, "create", return_value=ok_turn):
        res1 = runner1.run_once()
    assert res1.decision.source == "llm"
    assert ckpt_file.exists()

    # Simulate temp card leaking into produce_deck before crash/save
    tracker1.produce_deck.append(757)
    runner1.save_checkpoint()

    # Phase 2: Fresh process instances resume from checkpoint via start()
    ctrl2 = _StubCtrl()
    agent2 = AgentDecisionMaker(config=AgentConfig(api_key="sk-test", max_retries=0, session_mode=True))
    tracker2 = GameStateTracker(plan="logic")
    prompts_resumed = []
    runner2 = GakumasRunner(
        mode="adb",
        controller=ctrl2,
        agent=agent2,
        db=shared_db,
        tracker=tracker2,
        prompt_builder=PromptBuilder(db=shared_db),
        plan="logic",
        scenario_id="hajime_regular",
        session_mode=True,
        checkpoint_path=ckpt_file,
        auto_resume=True,
        on_prompt_built=lambda p, _a: prompts_resumed.append(p),
    )
    assert runner2.start() is True
    assert 757 not in tracker2.produce_deck
    assert len(agent2.session_messages) == 3
    assert runner2.scene_routing_reco.export_state()["last_week"] == 13
    assert runner2.scene_routing_reco.export_state()["last_stamina"] == [7, 31]

    obs2 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=13,
        turn=4,
        turns_remaining=8,
        stamina=7,
        max_stamina=31,
        genki=18,
        score=145,
        target_score=4000,
        cards=[
            RecognizedCard(card_id=125, name="幸せな時間+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=2, name="アピールの基本+", upgraded=True, bbox=bbox),
        ],
    )
    monkeypatch.setattr(runner2.scene_routing_reco, "recognize", lambda _f: obs2)
    with patch.object(agent2.client.chat.completions, "create", return_value=ok_turn):
        res2 = runner2.run_once()
    assert res2.decision.source == "llm"
    assert "【程序中断恢复】" in prompts_resumed[-1]
    assert "增量状态同步" in prompts_resumed[-1]
    assert "考试前中盘·M卡叠Buff蓄力期" in prompts_resumed[-1]
    assert "[M卡·叠Buff/引擎]" in prompts_resumed[-1]
    assert "[A卡·主动得分]" in prompts_resumed[-1]

    # Phase 3: Reset clears saved checkpoint file
    runner2.reset_produce_session(clear_checkpoint=True)
    assert not ckpt_file.exists()


def test_session_drink_rationing_and_exam_m_before_a_tags(shared_db):
    """Verify prompt rules and dynamic tags guide drink rationing & exam M->A phasing."""
    from demo.agent.prompts import get_session_system_prompt, get_system_prompt
    from demo.core.prompt import PromptBuilder
    from demo.core.tracker import GameStateTracker
    from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation

    for sp in (get_session_system_prompt("logic", "hajime_regular"), get_system_prompt("logic", compact=False)):
        assert "P饮料" in sp
        assert "好印象=0" in sp
        assert "M" in sp and "A" in sp

    builder = PromptBuilder(db=shared_db)
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db, plan="logic", p_idol_id=126)
    bbox = BoundingBox(xmin=50, ymin=1000, xmax=200, ymax=1300)

    # Regular lesson with herbal tea + coffee -> prompts reserve warning
    obs_reg = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=12,
        turn=1,
        turns_remaining=9,
        stamina=18,
        max_stamina=31,
        cards=[RecognizedCard(card_id=125, name="幸せな時間+", upgraded=True, bbox=bbox)],
        drinks=[
            RecognizedDrink(drink_id=16, name="おしゃれハーブティー", bbox=bbox),
            RecognizedDrink(drink_id=6, name="ホットコーヒー", bbox=bbox),
        ],
    )
    p_reg, _, _ = builder.build_session_prompt(obs_reg, tracker, commit=False)
    assert "好印象转化爆发" in p_reg
    assert "日常速通期" in p_reg
    assert "M卡·叠Buff/引擎" in p_reg

    # Late exam turn -> urges flushing A cards & drinks (incl. upgrade drinks like フレッシュビネガー)
    obs_exam_late = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=13,
        week_activity="最終試験",
        turn=10,
        turns_remaining=2,
        stamina=7,
        max_stamina=31,
        genki=54,
        score=648,
        target_score=4000,
        cards=[RecognizedCard(card_id=11, name="気分転換", bbox=bbox)],
        drinks=[
            RecognizedDrink(drink_id=2, name="烏龍茶", bbox=bbox),
            RecognizedDrink(drink_id=11, name="フレッシュビネガー", bbox=bbox),
        ],
    )
    p_late, _, _ = builder.build_session_prompt(obs_exam_late, tracker, commit=False)
    assert "后期A卡集中收割期" in p_late
    assert "A卡·限1爆发" in p_late
    assert "终盘必喝" in p_late

    # Early/mid battle with 2+ un-upgraded playable cards urges drinking フレッシュビネガー now
    obs_two_unupg = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=13,
        week_activity="最終試験",
        turn=3,
        turns_remaining=7,
        stamina=24,
        max_stamina=31,
        genki=12,
        cards=[
            RecognizedCard(card_id=60, name="えいえいおー", upgraded=False, bbox=bbox),
            RecognizedCard(card_id=120, name="あふれる思い出", upgraded=False, bbox=bbox),
        ],
        drinks=[RecognizedDrink(drink_id=11, name="フレッシュビネガー", bbox=bbox)],
    )
    p_upg, _, _ = builder.build_session_prompt(obs_two_unupg, tracker, commit=False)
    assert "手牌≥2张未强化" in p_upg


def test_sekirara_sena_p_idol_guide_injected_into_session_and_detailed_prompts(shared_db):
    """Verify [赤裸々] 十王星南 (#126) crawled guide feeds system prompts, bootstrap briefing, card rewards, and turn nudges."""
    from demo.agent.prompts import get_session_system_prompt, get_system_prompt
    from demo.core.idol_guides import get_p_idol_guide, resolve_p_idol_id

    assert resolve_p_idol_id(plan="sense", p_idol_id=None, deck_ids=[755, 1]) == 126
    guide = get_p_idol_guide(126)
    assert guide is not None
    assert guide.unique_p_item["name"] == "最初の鳥"
    assert len(guide.source_urls) >= 3

    for sp in (
        get_session_system_prompt("logic", "hajime_regular", p_idol_id=126),
        get_system_prompt("logic", scene_type="battle_turn", compact=False, p_idol_id=126),
    ):
        assert "[赤裸々] 十王 星南" in sp
        assert "最初の鳥" in sp
        assert "さっぱりひといき" in sp
        assert "特製ハツボシエキス" in sp

    builder = PromptBuilder(db=shared_db)
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db, plan="logic", p_idol_id=126)
    bbox = BoundingBox(xmin=60, ymin=1050, xmax=240, ymax=1350)

    # 1. Session bootstrap includes full P-Idol guide block + real P-Item '最初の鳥'
    obs_boot = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=1,
        turn=1,
        turns_remaining=10,
        stamina=31,
        max_stamina=31,
        cards=[RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, bbox=bbox)],
        drinks=[],
    )
    boot_prompt, _, boot_meta = builder.build_session_prompt(obs_boot, tracker, commit=False)
    assert "P卡角色专属攻略指导" in boot_prompt
    assert "最初の鳥" in boot_prompt
    assert "偶数回合开始时" in boot_prompt
    assert "さっぱりひといき" in boot_prompt
    assert "最強パフォーマー" in boot_prompt
    builder.commit_session_sync(tracker, boot_meta)

    # 2. Incremental card_reward annotates T0 recommended & anti-synergy cards
    obs_reward = ScreenObservation(
        scene_type="card_reward",
        plan="logic",
        week=4,
        stamina=24,
        max_stamina=31,
        options=[
            {"title": "本番前夜", "card_id": 130, "bbox": bbox},
            {"title": "さっぱりひといき", "card_id": 336, "bbox": bbox},
            {"title": "跳过", "bbox": bbox},
        ],
    )
    rew_prompt, rew_actions, _ = builder.build_session_prompt(obs_reward, tracker, commit=False)
    assert "★T0干劲引擎" in rew_prompt
    assert "⚠️注意反协同" in rew_prompt
    assert any("★T0干劲引擎" in a.description for a in rew_actions)

    # 3. Incremental battle nudge surfaces 15-genki unlock & 30-genki P-item threshold cleanly
    obs_nudge = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=6,
        week_activity="中間試験",
        turn=3,
        turns_remaining=6,
        stamina=20,
        max_stamina=31,
        genki=12,
        cards=[RecognizedCard(card_id=757, name="私を超えて(翔)", bbox=bbox)],
        drinks=[RecognizedDrink(drink_id=25, name="特製ハツボシエキス", bbox=bbox)],
    )
    inc_prompt, _, _ = builder.build_session_prompt(obs_nudge, tracker, commit=False)
    assert "[赤裸々星南提示]" in inc_prompt
    assert "元气12<15不可打" in inc_prompt

    # 4. Non-session detailed prompt & session serialization also preserve p_idol_id=126
    detailed_prompt, _ = builder.build_prompt(obs_boot, tracker, compact=False)
    assert "最初の鳥" in detailed_prompt
    assert "P卡角色专属攻略指导" in detailed_prompt

    agent = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True, session_mode=True))
    msgs = agent._prepare_messages(boot_prompt, obs_boot, None, use_session=True, scenario_id="hajime_regular", p_idol_id=126)
    assert "[赤裸々] 十王 星南" in msgs[0]["content"]
    agent._commit_session_turn(boot_prompt, 0, "ok", system_msg=msgs[0]["content"], plan="logic", scenario_id="hajime_regular", p_idol_id=126)
    exported = agent.export_session()
    assert exported["p_idol_id"] == 126
    agent2 = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True, session_mode=True))
    assert agent2.load_session(exported) is True
    assert agent2._session_p_idol_id == 126


def test_session_and_system_prompts_explain_four_zone_draw_rules_and_track_battle_piles(shared_db, monkeypatch):
    """Verify 4-zone draw/discard/reshuffle/exhaust rules appear in prompts & survive mid-battle popups + checkpoints."""
    from demo.agent.prompts import get_system_prompt

    monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a, **_kw: None)

    for sp in (
        get_session_system_prompt("logic", "hajime_regular", p_idol_id=126),
        get_system_prompt("sense", scene_type="battle_turn", compact=False),
        get_system_prompt("logic", scene_type="battle_turn", compact=False),
        get_system_prompt("anomaly", scene_type="battle_turn", compact=False),
    ):
        for kw in ("手牌", "牌堆", "弃牌", "除外", "重复使用"):
            assert kw in sp
        assert "出完后" in sp

    builder = PromptBuilder(db=shared_db, compact=False)
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.produce_deck = [756, 13, 1, 21, 60]  # #756 & #13 limit=1; #1, #21, #60 repeatable
    tracker.start_new_battle(db=shared_db)
    bbox = BoundingBox(xmin=60, ymin=1050, xmax=240, ymax=1350)

    # Turn 1: draw [756, 13, 1] -> Draw Pile still has [21, 60]
    obs_t1 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=5,
        turn=1,
        turns_remaining=6,
        stamina=25,
        max_stamina=31,
        genki=4,
        cards=[
            RecognizedCard(card_id=756, name="私を超えて+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=13, name="表現の基本", bbox=bbox),
            RecognizedCard(card_id=1, name="アピールの基本", bbox=bbox),
        ],
    )
    boot_prompt, _, boot_meta = builder.build_session_prompt(obs_t1, tracker, commit=True)
    assert "打牌抽卡与四区牌库循环规则" in boot_prompt
    assert "【本场四区牌库实时分布】" in boot_prompt
    assert "1) 🖐️ 【手牌】(3张)" in boot_prompt
    assert "2) 🎴 【牌堆/待抽】(2张)" in boot_prompt
    assert "3) ♻️ 【弃牌】(0张" in boot_prompt
    assert "4) 🚫 【除外】(0张" in boot_prompt
    assert meta_has_four_zones(boot_meta, tracker) is True

    # Play #756 (limit=1 -> Exhaust) and #1 (repeatable -> Discard); leave #13 (limit=1) unplayed in hand
    c756 = shared_db.get_card_by_id(756)
    tracker.record_card_play(756, limit_count=1, base_id=755, dsl_actions=c756["dsl_actions"], db=shared_db)
    c1 = shared_db.get_card_by_id(1)
    tracker.record_card_play(1, limit_count=c1.get("limit_count"), base_id=1, dsl_actions=c1["dsl_actions"], db=shared_db)
    assert tracker.battle_exhaust_pile == [756]
    assert tracker.battle_discard_pile == [1]

    # Turn 2: Draw Pile had [21, 60] (2 cards), turn 2 draws 3 -> draws 21 & 60 (Draw empties!) then reshuffles Discard [1, 13] -> Draw and draws 1!
    obs_t2 = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=5,
        turn=2,
        turns_remaining=5,
        stamina=22,
        max_stamina=31,
        genki=8,
        cards=[
            RecognizedCard(card_id=21, name="意識の基本", bbox=bbox),
            RecognizedCard(card_id=60, name="えいえいおー", bbox=bbox),
            RecognizedCard(card_id=1, name="アピールの基本", bbox=bbox),
        ],
    )
    delta_prompt, _, _ = builder.build_session_prompt(obs_t2, tracker, commit=True)
    assert tracker.battle_reshuffle_count == 1
    assert tracker.battle_draw_pile == [13]  # Unplayed #13 went Discard -> Draw! Exhausted #756 never returns!
    assert tracker.battle_exhaust_pile == [756]
    assert 756 not in tracker.battle_draw_pile and 756 not in tracker.battle_discard_pile
    assert "牌堆抽空触发重洗(第1次)" in delta_prompt
    assert "4) 🚫 【除外】(1张，已打出限1牌，本场不可再使用): 《私を超えて+》×1" in delta_prompt

    # Mid-battle drink confirm popup in GakumasRunner must preserve battle zones
    stub = _StubLiveController()
    agent = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
    runner = GakumasRunner(mode="adb", controller=stub, agent=agent, db=shared_db, tracker=tracker, plan="logic", session_mode=False)
    pop_obs = ScreenObservation(
        scene_type="popup",
        plan="logic",
        week=5,
        options=[{"choice_id": 0, "name": "使う", "title": "使う", "bbox": bbox}],
    )
    monkeypatch.setattr(runner.scene_routing_reco, "recognize", lambda _f: pop_obs)
    runner.run_once()
    assert tracker.in_battle is True
    assert tracker.battle_exhaust_pile == [756]
    assert tracker.battle_draw_pile == [13]

    # Checkpoint preserves 4-zone piles accurately
    ckpt = tracker.export_checkpoint()
    restored = GameStateTracker(plan="logic", p_idol_id=126)
    restored.load_checkpoint(ckpt, db=shared_db)
    assert restored.battle_hand == [21, 60, 1]
    assert restored.battle_draw_pile == [13]
    assert restored.battle_exhaust_pile == [756]
    assert restored.battle_reshuffle_count == 1


def meta_has_four_zones(_meta, tracker: GameStateTracker) -> bool:
    return (
        len(tracker.battle_hand) == 3
        and len(tracker.battle_draw_pile) == 2
        and len(tracker.battle_discard_pile) == 0
        and len(tracker.battle_exhaust_pile) == 0
    )


def test_action_order_rule_perfect_countdown_and_popup_preserves_in_battle(shared_db, monkeypatch):
    """Verify +1 use ordering warnings, PERFECT countdown formatting, mid-battle popup in_battle preservation, and card_reward dedup."""
    from demo.agent.prompts import LOGIC_SYSTEM_PROMPT, PRODUCE_SESSION_SYSTEM_PROMPT
    from demo.core.idol_guides import format_idol_turn_nudge, get_p_idol_guide

    monkeypatch.setattr("demo.maa_bridge.runner.time.sleep", lambda *_a, **_kw: None)

    guide = get_p_idol_guide(126)
    assert guide is not None
    for prompt_txt in (
        LOGIC_SYSTEM_PROMPT,
        PRODUCE_SESSION_SYSTEM_PROMPT,
        " ".join(guide.battle_pacing_rules),
    ):
        assert "剩余出牌" in prompt_txt
        assert "先打" in prompt_txt

    bbox = BoundingBox(xmin=60, ymin=1050, xmax=240, ymax=1350)
    obs_perf = ScreenObservation(
        scene_type="battle_turn",
        plan="logic",
        week=5,
        week_activity="中間まで 1週 (PERFE@T*で)",
        turn=7,
        turns_remaining=2,
        stamina=18,
        max_stamina=31,
        genki=25,
        score=0,
        target_score=201,
        cards=[
            RecognizedCard(card_id=758, name="私を超えて(翔)+", upgraded=True, bbox=bbox),
            RecognizedCard(card_id=14, name="表現の基本+", upgraded=True, bbox=bbox),
        ],
    )
    nudge = format_idol_turn_nudge(guide, obs_perf)
    assert "必须【先打(翔)】" in nudge

    builder = PromptBuilder(db=shared_db, compact=False)
    tracker = GameStateTracker(plan="logic", p_idol_id=126)
    tracker.init_produce_deck(db=shared_db, plan="logic", p_idol_id=126)
    tracker.start_new_battle(db=shared_db)

    p_perf, _, meta_b = builder.build_session_prompt(obs_perf, tracker, commit=True)
    assert "距离达成 CLEAR/PERFECT 还需 201 分" in p_perf
    assert "已过CLEAR、正冲击满额PERFECT" in p_perf
    assert "🚨 【课程末盘清分死令】" in p_perf
    assert "必须【先打】『+1使用次数』免费A卡" in p_perf
    assert meta_b["in_battle"] is True

    # Mid-battle drink confirm popup keeps sync_meta["in_battle"] True and suppresses false 新战斗 banner
    obs_popup = ScreenObservation(
        scene_type="popup",
        plan="logic",
        week=5,
        stamina=18,
        max_stamina=31,
        options=[{"choice_id": 0, "name": "使う", "title": "使う", "bbox": bbox}],
    )
    _, _, meta_pop = builder.build_session_prompt(obs_popup, tracker, commit=True)
    assert meta_pop["in_battle"] is True
    assert tracker.llm_last_in_battle is True

    p_after_pop, _, _ = builder.build_session_prompt(obs_perf, tracker, commit=True)
    assert "进入新战斗" not in p_after_pop

    # Two consecutive card_reward '受け取る' frames for the same card add it once only
    stub = _StubLiveController()
    runner = GakumasRunner(
        mode="adb",
        controller=stub,
        agent=AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True)),
        db=shared_db,
        tracker=GameStateTracker(plan="logic", p_idol_id=126),
        plan="logic",
        session_mode=False,
    )
    runner.tracker.init_produce_deck(db=shared_db, plan="logic", p_idol_id=126)
    deck_before = len(runner.tracker.produce_deck)
    rew_obs_1 = ScreenObservation(
        scene_type="card_reward",
        plan="logic",
        week=5,
        options=[{"choice_id": 0, "card_id": 154, "name": "受け取る", "title": "受け取る", "bbox": bbox}],
    )
    rew_obs_2 = ScreenObservation(
        scene_type="card_reward",
        plan="logic",
        week=5,
        options=[{"choice_id": 0, "card_id": 154, "name": "受け取る", "title": "受け取る", "bbox": bbox}],
    )
    skip_obs = ScreenObservation(
        scene_type="skip",
        plan="logic",
        week=5,
        options=[{"choice_id": 0, "name": "点击屏幕继续", "title": "点击屏幕继续", "bbox": bbox}],
    )
    q = [rew_obs_1, rew_obs_2, skip_obs]
    monkeypatch.setattr(runner.scene_routing_reco, "recognize", lambda _f: q.pop(0))
    runner.run_once()
    runner.run_once()
    assert runner.tracker.produce_deck.count(154) == 1
    assert len(runner.tracker.produce_deck) == deck_before + 1
    runner.run_once()
    assert runner._claimed_card_in_reward_scene is False



