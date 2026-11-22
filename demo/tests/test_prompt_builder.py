"""
Unit tests for demo/core/prompt.py and demo/core/rules.py.
"""

from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.rules import (
    evaluate_card_cost_and_usability,
    summarize_dsl_actions,
)
from demo.core.tracker import GameStateTracker
from demo.schemas import BoundingBox, RecognizedCard, RecognizedDrink, ScreenObservation


def test_rules_cost_shielded_by_genki():
    card = {"id": 1, "dsl_cost": "cost-=5"}
    # Genki (8) >= cost (5) -> hp loss is 0
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=8,
        buffs={},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 0

    # Genki (3) < cost (5) -> hp loss is 2
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=3,
        buffs={},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 2


def test_rules_fixed_stamina_cost_bypasses_genki():
    card = {"id": 152, "dsl_cost": "stamina-=3"}
    # Even with 20 genki, fixed stamina cost reduces HP directly by 3
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=10,
        genki=20,
        buffs={},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 3


def test_rules_stamina_survival_check():
    card = {"id": 1, "dsl_cost": "stamina-=5"}
    # Stamina 5 is not enough to survive 5 loss (must have stamina > hp_loss)
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=5,
        genki=0,
        buffs={},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert not playable
    assert "体力不足" in reason


def test_rules_buff_cost_check():
    card = {"id": 100, "dsl_cost": "concentration-=3"}
    # Concentration 2 < 3 -> unplayable
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=0,
        buffs={"concentration": 2},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert not playable
    assert "concentration不足" in reason

    # Concentration 3 >= 3 -> playable
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=0,
        buffs={"concentration": 3},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 0


def test_rules_condition_predicates():
    # Card 5 挑戦 requires goodConditionTurns >= 1
    card = {"id": 5, "dsl_cost": "cost-=7", "dsl_conditions": "if:goodConditionTurns>=1"}

    # No goodCondition -> unplayable
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=10,
        buffs={"goodConditionTurns": 0},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert not playable
    assert "条件未满足" in reason

    # Has goodCondition -> playable
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=10,
        buffs={"goodConditionTurns": 2},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable


def test_rules_used_limit_card():
    card = {"id": 182, "limit_count": 1}
    # Card already in used_limit_cards -> unplayable
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=10,
        buffs={},
        card_uses_remaining=1,
        used_limit_cards={182},
    )
    assert not playable
    assert "已使用过" in reason


def test_rules_exhausted_turn_actions():
    card = {"id": 1, "dsl_cost": "cost-=1"}
    # card_uses_remaining == 0 -> unplayable
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=card,
        stamina=20,
        genki=10,
        buffs={},
        card_uses_remaining=0,
        used_limit_cards=set(),
    )
    assert not playable
    assert "本回合出牌次数已耗尽" in reason


def test_summarize_dsl_actions():
    summary = summarize_dsl_actions(
        "score+=25; genki+=7; goodConditionTurns+=3; cardUsesRemaining+=1"
    )
    assert "得分+25" in summary
    assert "元气+7" in summary
    assert "好调+3回合" in summary
    assert "出牌次数+1" in summary


def test_prompt_builder_battle_prompt_compact_and_actions():
    with GakumasDB() as db:
        builder = PromptBuilder(db)
        tracker = GameStateTracker("sense")
        bbox = BoundingBox(xmin=100, ymin=1400, xmax=250, ymax=1600)

        obs = ScreenObservation(
            scene_type="battle_turn",
            turn=3,
            turns_remaining=7,
            stamina=25,
            max_stamina=30,
            genki=5,
            score=300,
            plan="sense",
            buffs={"goodConditionTurns": 2, "concentration": 3},
            cards=[
                RecognizedCard(card_id=5, name="挑戦", bbox=bbox),
                RecognizedCard(card_id=1, name="アピールの基本", bbox=bbox),
            ],
            drinks=[
                RecognizedDrink(drink_id=1, name="初星水", bbox=bbox),
            ],
        )

        prompt, actions = builder.build_prompt(obs, tracker)

        # Verify full-context uncapped prompt sections (playstyle, P-Items, stage bonus, initial/current stats, full card effects)
        assert "一、剧本流程与当前周目" in prompt
        assert "二、职业流派核心玩法" in prompt
        assert "三、被动道具加成 (P-Items)" in prompt
        assert "四、场景与舞台审查加成" in prompt
        assert "五、初始与当前面板数值" in prompt
        assert "六、打牌阶段：当前抽到的手牌与可用饮料效果" in prompt

        # Compact mode also remains available when compact=True
        compact_prompt, _ = builder.build_prompt(obs, tracker, compact=True)
        approx_tokens = len(compact_prompt) // 2
        assert approx_tokens < 800, f"Compact prompt too long: ~{approx_tokens} tokens"

        # Check legal actions sequence
        assert len(actions) == 4
        assert [a.choice_id for a in actions] == [0, 1, 2, 3]
        assert actions[0].action_type == "play_card"
        assert actions[1].action_type == "play_card"
        assert actions[2].action_type == "use_drink"
        assert actions[3].action_type == "end_turn"


def test_prompt_builder_zero_playable_cards_leaves_only_end_turn():
    with GakumasDB() as db:
        builder = PromptBuilder(db)
        tracker = GameStateTracker("sense")
        bbox = BoundingBox(xmin=100, ymin=1400, xmax=250, ymax=1600)

        # Stamina is 1, card costs 7, no genki -> card unplayable
        obs = ScreenObservation(
            scene_type="battle_turn",
            turn=1,
            turns_remaining=10,
            stamina=1,
            max_stamina=30,
            genki=0,
            plan="sense",
            cards=[
                RecognizedCard(card_id=5, name="挑戦", bbox=bbox),
            ],
            drinks=[],
        )

        prompt, actions = builder.build_prompt(obs, tracker)

        # Only end_turn is legal!
        assert len(actions) == 1
        assert actions[0].choice_id == 0
        assert actions[0].action_type == "end_turn"


def test_prompt_builder_option_scene():
    builder = PromptBuilder()
    obs = ScreenObservation(
        scene_type="produce_schedule",
        stamina=25,
        options=[
            {"title": "Vo 训练", "description": "Vo+30, 体力-10"},
            {"title": "Da 训练", "description": "Da+30, 体力-10"},
            {"title": "休息", "description": "体力+20"},
        ],
    )
    prompt, actions = builder.build_prompt(obs)
    assert len(actions) == 3
    assert actions[0].choice_id == 0
    assert actions[0].action_type == "pick_option"
    assert "Vo 训练" in actions[0].description
    assert "Da 训练" in actions[1].description
    assert "休息" in actions[2].description


def test_generate_legal_actions_idempotency():
    """Fix 1: Ensure calling generate_legal_actions_and_update_obs multiple times is idempotent."""
    with GakumasDB() as db:
        builder = PromptBuilder(db)
        tracker = GameStateTracker("sense")
        bbox = BoundingBox(xmin=100, ymin=1400, xmax=250, ymax=1600)

        obs = ScreenObservation(
            scene_type="battle_turn",
            turn=1,
            turns_remaining=10,
            stamina=20,
            max_stamina=30,
            genki=5,
            score=0,
            plan="sense",
            cards=[
                RecognizedCard(card_id=5, name="挑戦", bbox=bbox),
                RecognizedCard(card_id=1, name="アピールの基本", bbox=bbox),
            ],
            drinks=[
                RecognizedDrink(drink_id=1, name="初星水", bbox=bbox),
            ],
        )

        # First call
        actions_1 = builder.generate_legal_actions_and_update_obs(obs, tracker)
        obs_dump_1 = obs.model_dump()
        actions_dump_1 = [a.model_dump() for a in actions_1]

        # Second call on the exact same observation object
        actions_2 = builder.generate_legal_actions_and_update_obs(obs, tracker)
        obs_dump_2 = obs.model_dump()
        actions_dump_2 = [a.model_dump() for a in actions_2]

        # Third call
        actions_3 = builder.generate_legal_actions_and_update_obs(obs, tracker)
        obs_dump_3 = obs.model_dump()
        actions_dump_3 = [a.model_dump() for a in actions_3]

        assert actions_dump_1 == actions_dump_2 == actions_dump_3
        assert obs_dump_1 == obs_dump_2 == obs_dump_3


def test_legal_action_payload_limit_count_and_base_id():
    """Fix 2: Ensure card LegalAction.payload contains limit_count and base_id."""
    with GakumasDB() as db:
        builder = PromptBuilder(db)
        bbox = BoundingBox(xmin=100, ymin=1400, xmax=250, ymax=1600)

        # Card 182 is '手拍子' with limit_count = 1
        # Card 1 is 'アピールの基本' with limit_count = None
        obs = ScreenObservation(
            scene_type="battle_turn",
            stamina=30,
            cards=[
                RecognizedCard(card_id=182, name="手拍子", bbox=bbox),
                RecognizedCard(card_id=1, name="アピールの基本", bbox=bbox),
            ],
        )

        actions = builder.generate_legal_actions_and_update_obs(obs)
        card_actions = [a for a in actions if a.action_type == "play_card"]
        assert len(card_actions) == 2

        # Card 182
        payload_182 = card_actions[0].payload
        assert "limit_count" in payload_182
        assert "base_id" in payload_182
        assert payload_182["limit_count"] == 1
        assert payload_182["base_id"] == 182
        assert payload_182["card_id"] == 182

        # Card 1
        payload_1 = card_actions[1].payload
        assert "limit_count" in payload_1
        assert "base_id" in payload_1
        assert payload_1["limit_count"] is None
        assert payload_1["base_id"] == 1
        assert payload_1["card_id"] == 1


def test_fixed_stamina_cost_immunity_to_nullify_and_half_cost():
    """Fix 3: nullifyCostCards and halfCostTurns only modify cost-=, NOT stamina-=."""
    # Fixed stamina (stamina-=4) is immutable: ignores nullifyCostCards and halfCostTurns
    fixed_card = {"id": 152, "dsl_cost": "stamina-=4"}

    # With nullifyCostCards
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=fixed_card,
        stamina=20,
        genki=10,
        buffs={"nullifyCostCards": 1},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 4, "Fixed stamina penalty must not be reduced by nullifyCostCards"

    # With halfCostTurns
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=fixed_card,
        stamina=20,
        genki=10,
        buffs={"halfCostTurns": 1},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 4, "Fixed stamina penalty must not be reduced by halfCostTurns"

    # Standard stamina cost (cost-=4) DOES benefit from nullifyCostCards and halfCostTurns
    regular_card = {"id": 1, "dsl_cost": "cost-=4"}

    # With nullifyCostCards -> cost becomes 0
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=regular_card,
        stamina=20,
        genki=0,
        buffs={"nullifyCostCards": 1},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 0

    # With halfCostTurns -> cost 4 becomes ceil(4/2) = 2
    playable, hp_loss, reason = evaluate_card_cost_and_usability(
        card=regular_card,
        stamina=20,
        genki=0,
        buffs={"halfCostTurns": 1},
        card_uses_remaining=1,
        used_limit_cards=set(),
    )
    assert playable
    assert hp_loss == 2


def test_summarize_dsl_actions_deduplication_and_cap():
    """Fix 4: Deduplicate repeated clauses and cap output length at <= 60 chars."""
    # Duplicate clauses deduplication
    repeated_dsl = "addCardToDeck(832); addCardToDeck(832); addCardToDeck(832); addCardToDeck(832)"
    summary = summarize_dsl_actions(repeated_dsl)
    assert summary == "addCardToDeck(832)"

    # Capping length at 60 characters for verbose multi-clause DSL
    long_dsl = "; ".join([f"score+={i*10}" for i in range(15)])
    long_summary = summarize_dsl_actions(long_dsl)
    assert len(long_summary) <= 60
    assert long_summary.endswith("...")


def test_extreme_prompt_token_budget_under_800_tokens():
    """Fix 4: Ensure extreme observation with 5 complex cards, 3 drinks, and 10 buffs stays < 800 tokens (< 1500 chars)."""
    with GakumasDB() as db:
        builder = PromptBuilder(db)
        tracker = GameStateTracker("sense")
        bbox = BoundingBox(xmin=100, ymin=1400, xmax=250, ymax=1600)

        # 5 complex cards (IDs 932, 831, 832, 930, 342)
        # Note: 831, 832, 342 are complex DB cards; 932 and 930 have custom multi-clause DSL summaries
        cards = [
            RecognizedCard(
                card_id=932,
                name="レジェンドスター+",
                bbox=bbox,
                dsl_summary="集中+=5; score+=concentration*2.5; concentration+=3; score+=50",
            ),
            RecognizedCard(card_id=831, name="笑顔を生み出す一番星", bbox=bbox),
            RecognizedCard(card_id=832, name="微熱ノスタルジー♡", bbox=bbox),
            RecognizedCard(
                card_id=930,
                name="レジェンドスター",
                bbox=bbox,
                dsl_summary="集中+=4; score+=concentration*2.0; concentration+=2; score+=30",
            ),
            RecognizedCard(card_id=342, name="こくりとひとくち", bbox=bbox),
        ]

        # 3 drinks
        drinks = [
            RecognizedDrink(drink_id=1, name="初星水", bbox=bbox),
            RecognizedDrink(drink_id=2, name="烏龍茶", bbox=bbox),
            RecognizedDrink(drink_id=3, name="ルイボスティー", bbox=bbox),
        ]

        # 10 buffs
        buffs = {
            "goodConditionTurns": 4,
            "perfectConditionTurns": 2,
            "concentration": 10,
            "motivation": 6,
            "goodImpressionTurns": 5,
            "fullPowerCharge": 3,
            "nullifyCostCards": 1,
            "halfCostTurns": 2,
            "doubleCostTurns": 0,
            "cardUsesRemaining": 2,
        }

        obs = ScreenObservation(
            scene_type="battle_turn",
            turn=5,
            turns_remaining=5,
            stamina=22,
            max_stamina=30,
            genki=12,
            score=1800,
            target_score=5000,
            plan="sense",
            buffs=buffs,
            cards=cards,
            drinks=drinks,
        )

        # Full context prompt (default, no 800-token truncation)
        full_prompt, actions = builder.build_prompt(obs, tracker)
        assert "一、剧本流程与当前周目" in full_prompt
        assert "二、职业流派核心玩法" in full_prompt
        assert "三、被动道具加成 (P-Items)" in full_prompt
        assert "四、场景与舞台审查加成" in full_prompt
        assert "五、初始与当前面板数值" in full_prompt
        assert "六、打牌阶段：当前抽到的手牌与可用饮料效果" in full_prompt
        assert "レジェンドスター+" in full_prompt

        # Compact mode (when compact=True is requested)
        prompt, actions = builder.build_prompt(obs, tracker, compact=True)

        # Verify token budget and length in compact mode: strictly < 800 tokens (< 1500 chars)
        prompt_len = len(prompt)
        approx_tokens = prompt_len // 2
        assert prompt_len < 1500, f"Prompt length {prompt_len} exceeds 1500 character budget!"
        assert approx_tokens < 800, f"Prompt token count {approx_tokens} exceeds 800 token budget!"

        # Verify action descriptions are concise and do NOT repeat the verbose DSL summary
        card_actions = [a for a in actions if a.action_type == "play_card"]
        assert len(card_actions) == 5
        for act in card_actions:
            # Concise format: e.g. "打出卡牌: レジェンドスター (需扣血2)" or "(无实际扣血)"
            assert act.description.startswith("打出卡牌: ")
            assert " | " not in act.description
            assert ("(需扣血" in act.description or "(无实际扣血)" in act.description)


def test_hajime_regular_13_weeks_visual_recognition_and_flow():
    """Verify all 13 weeks of 初·常规 (hajime_regular) synthesize, visually recognize week 1..13, and build rich prompts."""
    from demo.core.scenarios import HAJIME_REGULAR_WEEKS
    from demo.maa_bridge.custom_recognition import GakumasSceneRoutingRecognition
    from demo.vision.mock_generator import MockFrameGenerator

    with GakumasDB() as db:
        gen = MockFrameGenerator(db=db, plan="sense")
        reco = GakumasSceneRoutingRecognition(db=db, plan="sense")
        builder = PromptBuilder(db=db)

        for w in range(1, 14):
            expected_node = HAJIME_REGULAR_WEEKS[w]
            frame, gt_obs = gen.generate_week_frame(week=w, scenario_id="hajime_regular", plan="sense")
            assert gt_obs.week == w
            assert gt_obs.scene_type == expected_node.scene_type

            # Recognize week and scene directly from the synthesized image pixels
            rec_obs = reco.recognize(frame)
            assert rec_obs is not None
            assert rec_obs.week == w, f"Expected week {w}, visually recognized week {rec_obs.week}"
            assert rec_obs.total_weeks == 13
            assert rec_obs.scene_type == expected_node.scene_type
            assert expected_node.title in rec_obs.week_activity

            prompt, actions = builder.build_prompt(rec_obs)
            assert f"第 {w} / 13 周" in prompt
            assert expected_node.title in prompt
            assert len(actions) > 0


