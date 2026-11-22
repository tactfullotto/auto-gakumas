"""
End-to-End Integration Tests for Produce / Reward / Shop / Popup / Skip Scenes.
Verifies full closed-loop pipeline execution without modifying root assets.
"""

from pathlib import Path
import numpy as np
import pytest

from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.app import create_configured_simulator
from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.runner import GakumasRunner
from demo.vision.mock_generator import MockFrameGenerator
from demo.vision.roi import BASE_HEIGHT, BASE_WIDTH


@pytest.fixture
def shared_db():
    db = GakumasDB("gakumas_library.db")
    yield db
    db.close()


@pytest.fixture
def offline_agent():
    return AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))


def test_e2e_produce_schedule_loop(shared_db, offline_agent):
    """End-to-end produce_schedule scene: frame synthesis -> recognition -> prompt -> click."""
    ctrl = MockController(db=shared_db, plan="sense", auto_progress=True)
    ctrl.set_scenario("produce_schedule")

    runner = GakumasRunner(
        mode="mock",
        controller=ctrl,
        agent=offline_agent,
        db=shared_db,
        plan="sense",
    )
    assert runner.start() is True
    res = runner.run_once()

    assert res.success is True
    assert res.scene_type == "produce_schedule"
    assert res.observation is not None
    assert len(res.observation.options) == 3
    assert len(res.legal_actions) == 3
    assert res.decision is not None
    assert res.decision.source == "fallback"
    assert 0 <= res.decision.choice_id < len(res.legal_actions)
    assert res.click_coords is not None
    assert len(ctrl.click_history) == 1


def test_e2e_card_reward_three_choice_pick(shared_db, offline_agent):
    """End-to-end card_reward (3-choice card addition) scene across all 3 plans."""
    for plan in ("sense", "logic", "anomaly"):
        ctrl = MockController(db=shared_db, plan=plan, auto_progress=True)
        ctrl.set_scenario("card_reward")

        prompt_captured = {}

        def _on_prompt(prompt_text, actions):
            prompt_captured["text"] = prompt_text
            prompt_captured["count"] = len(actions)

        runner = GakumasRunner(
            mode="mock",
            controller=ctrl,
            agent=offline_agent,
            db=shared_db,
            plan=plan,
            on_prompt_built=_on_prompt,
        )
        runner.start()
        res = runner.run_once()

        assert res.success is True
        assert res.scene_type == "card_reward"
        assert prompt_captured["count"] == 3
        assert "一、剧本流程与当前周目" in prompt_captured["text"]
        assert "七、合法动作列表与决策输出" in prompt_captured["text"]
        assert res.decision is not None
        assert res.decision.choice_id in (0, 1, 2)
        chosen_bbox = res.action_taken.bbox
        cx, cy = res.click_coords
        assert chosen_bbox.xmin <= cx <= chosen_bbox.xmax
        assert chosen_bbox.ymin <= cy <= chosen_bbox.ymax


def test_e2e_produce_to_reward_to_battle_progression(shared_db, offline_agent):
    """End-to-end multi-scene progression: produce_schedule -> card_reward -> battle_turn -> popup."""
    ctrl = MockController(db=shared_db, plan="logic", auto_progress=True)
    ctrl.queue_scenarios([
        {"scene_type": "produce_schedule"},
        {"scene_type": "card_reward"},
        {"scene_type": "battle_turn", "turn": 1, "turns_remaining": 8, "stamina": 28, "genki": 4},
        {"scene_type": "popup", "message": "Continue Produce?"},
    ])
    # First screencap should start on produce_schedule
    ctrl.set_scenario("produce_schedule")
    ctrl.scenario_queue = [
        {"scene_type": "card_reward"},
        {"scene_type": "battle_turn", "turn": 1, "turns_remaining": 8, "stamina": 28, "genki": 4},
        {"scene_type": "popup", "message": "Continue Produce?"},
    ]

    runner = GakumasRunner(
        mode="mock",
        controller=ctrl,
        agent=offline_agent,
        db=shared_db,
        plan="logic",
    )
    results = runner.run_loop(max_steps=4, delay_between_steps=0.0)

    assert len(results) == 4
    assert all(r.success for r in results)
    assert [r.scene_type for r in results] == [
        "produce_schedule",
        "card_reward",
        "battle_turn",
        "popup",
    ]


def test_e2e_skip_cutscene_detection_and_single_action_rule_skip(shared_db, offline_agent):
    """Verify skip cutscene detection and single-legal-action zero-token bypass (source='rule_skip')."""
    # 1. Skip cutscene frame recognized as 'skip' with 1 option
    skip_frame = np.zeros((BASE_HEIGHT, BASE_WIDTH, 3), dtype=np.uint8)
    skip_frame[50:110, 910:1030] = (240, 240, 240)
    ctrl_skip = MockController(db=shared_db, plan="sense", auto_progress=False)
    ctrl_skip.inject_frame(skip_frame)
    runner_skip = GakumasRunner(mode="mock", controller=ctrl_skip, agent=offline_agent, db=shared_db, plan="sense")
    runner_skip.start()
    res_skip = runner_skip.run_once()
    assert res_skip.success is True
    assert res_skip.scene_type == "skip"
    assert res_skip.observation is not None
    assert len(res_skip.observation.options) == 1
    assert res_skip.observation.options[0]["name"] == "SKIP"

    # 2. Single legal action (battle_turn where all 3 cards exceed stamina -> only end_turn is legal)
    expensive_cards = [
        c for c in shared_db.get_cards_for_plan("sense")
        if c.get("dsl_cost") and "cost-=" in c["dsl_cost"] and c.get("icon_path") and Path(c["icon_path"]).exists()
    ][:3]
    assert len(expensive_cards) == 3

    ctrl_battle = MockController(db=shared_db, plan="sense", auto_progress=False)
    frame_battle, _ = ctrl_battle.generator.generate_battle_frame(
        cards=expensive_cards,
        drinks=[],
        turn=1,
        turns_remaining=5,
        stamina=1,
        genki=0,
        plan="sense",
    )
    ctrl_battle.inject_frame(frame_battle)
    runner_battle = GakumasRunner(mode="mock", controller=ctrl_battle, agent=offline_agent, db=shared_db, plan="sense")
    runner_battle.start()
    res_battle = runner_battle.run_once()

    assert res_battle.success is True
    assert len(res_battle.legal_actions) == 1
    assert res_battle.legal_actions[0].action_type == "end_turn"
    assert res_battle.decision is not None
    assert res_battle.decision.source == "rule_skip"
    assert res_battle.decision.prompt_tokens == 0
    assert res_battle.decision.completion_tokens == 0
    assert res_battle.decision.choice_id == 0


def test_e2e_pipeline_simulator_produce_and_reward(shared_db, offline_agent):
    """Verify MaaPipelineSimulator executes pipeline.json across ProduceSchedule and CardReward."""
    ctrl = MockController(db=shared_db, plan="anomaly", auto_progress=True)
    ctrl.set_scenario("produce_schedule")
    ctrl.scenario_queue = [{"scene_type": "card_reward"}]

    tracker = GameStateTracker(plan="anomaly")
    prompt_builder = PromptBuilder(db=shared_db)
    sim = create_configured_simulator(
        db=shared_db,
        plan="anomaly",
        controller=ctrl,
        agent=offline_agent,
        tracker=tracker,
        prompt_builder=prompt_builder,
    )

    sim.current_node = "ProduceSchedule"
    steps = sim.run_task(entry_node="ProduceSchedule", max_steps=2)

    assert len(steps) == 2
    assert steps[0].node_name == "ProduceSchedule"
    assert steps[0].recognition_hit is True
    assert steps[0].action_success is True
    assert steps[1].recognition_hit is True
    assert len(ctrl.click_history) == 2
