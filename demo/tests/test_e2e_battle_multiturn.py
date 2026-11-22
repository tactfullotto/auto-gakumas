"""
End-to-End Multi-Turn Battle, CLI Entrypoint, and Root Asset Read-Only Invariance Tests.
Covers sense, logic, and anomaly archetypes across multiple turns.
"""

from pathlib import Path
import sqlite3
import pytest

from demo.agent.client import AgentConfig, AgentDecisionMaker
from demo.app import main as app_main
from demo.core.db import GakumasDB
from demo.core.prompt import PromptBuilder
from demo.core.tracker import GameStateTracker
from demo.maa_bridge.mock_controller import MockController
from demo.maa_bridge.runner import GakumasRunner


@pytest.fixture
def shared_db():
    db = GakumasDB("gakumas_library.db")
    yield db
    db.close()


@pytest.fixture
def offline_agent():
    return AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))


def _pick_test_cards(db: GakumasDB, plan: str):
    """Pick 1 limit_count=1 card (if available) and 2 repeatable cards with icons for plan."""
    all_cards = [
        c for c in db.get_cards_for_plan(plan)
        if c.get("icon_path") and Path(c["icon_path"]).exists()
    ]
    limit_cards = [c for c in all_cards if c.get("limit_count") == 1]
    normal_cards = [c for c in all_cards if not c.get("limit_count")]
    chosen = []
    if limit_cards:
        chosen.append(limit_cards[0])
    chosen.extend(normal_cards[: max(0, 3 - len(chosen))])
    return chosen


@pytest.mark.parametrize("plan,buffs", [
    ("sense", {"goodConditionTurns": 4, "concentration": 3}),
    ("logic", {"goodImpressionTurns": 5, "motivation": 4}),
    ("anomaly", {"fullPowerCharge": 6}),
])
def test_e2e_battle_multiturn_all_three_plans(shared_db, offline_agent, plan, buffs):
    """Run multi-turn battle loop for sense, logic, and anomaly plans."""
    cards = _pick_test_cards(shared_db, plan)
    drinks = [
        d for d in shared_db.get_drinks_for_plan(plan)
        if d.get("icon_path") and Path(d["icon_path"]).exists()
    ][:2]

    ctrl = MockController(db=shared_db, plan=plan, auto_progress=True)
    ctrl.set_scenario(
        "battle_turn",
        cards=cards,
        drinks=drinks,
        turn=1,
        turns_remaining=6,
        stamina=30,
        max_stamina=30,
        genki=6,
        score=200,
        buffs=buffs,
    )

    prompts_seen = []

    def _on_prompt(prompt_txt, actions):
        prompts_seen.append((prompt_txt, actions))

    runner = GakumasRunner(
        mode="mock",
        controller=ctrl,
        agent=offline_agent,
        db=shared_db,
        plan=plan,
        on_prompt_built=_on_prompt,
    )
    results = runner.run_loop(max_steps=4, delay_between_steps=0.0)

    assert len(results) == 4
    assert all(r.success for r in results)
    assert all(r.scene_type == "battle_turn" for r in results)
    assert len(ctrl.click_history) == 4
    for prompt_txt, actions in prompts_seen:
        assert len(actions) >= 1
        assert "一、剧本流程与当前周目" in prompt_txt
        assert "七、合法动作列表与决策输出" in prompt_txt


def test_e2e_limit_count_1_card_blocked_on_subsequent_turn(shared_db):
    """Verify a played limit_count=1 card is tracked and marked unplayable on next turn."""
    limit_cards = [
        c for c in shared_db.get_cards_for_plan("sense")
        if c.get("limit_count") == 1
        and not (c.get("dsl_conditions") or "").strip()
        and c.get("type") != "trouble"
        and c.get("icon_path")
        and Path(c["icon_path"]).exists()
    ]
    assert len(limit_cards) > 0
    limit_card = limit_cards[0]

    tracker = GameStateTracker(plan="sense")
    prompt_builder = PromptBuilder(db=shared_db)
    ctrl = MockController(db=shared_db, plan="sense", auto_progress=False)

    # Turn 1: fill 3-card hand with this limit_count=1 card
    frame1, _ = ctrl.generator.generate_battle_frame(
        cards=[limit_card, limit_card, limit_card],
        drinks=[],
        turn=1,
        turns_remaining=8,
        stamina=30,
        genki=10,
        plan="sense",
    )
    ctrl.inject_frame(frame1)

    agent = AgentDecisionMaker(AgentConfig(api_key=None, fallback_on_error=True))
    runner = GakumasRunner(
        mode="mock",
        controller=ctrl,
        agent=agent,
        db=shared_db,
        tracker=tracker,
        prompt_builder=prompt_builder,
        plan="sense",
    )
    runner.start()
    res1 = runner.run_once()
    assert res1.success is True
    assert res1.action_taken is not None
    assert res1.action_taken.action_type == "play_card"
    assert limit_card["id"] in tracker.used_limit_cards

    # Turn 2: same limit_count=1 card appears again; all copies blocked -> only end_turn legal -> rule_skip
    frame2, _ = ctrl.generator.generate_battle_frame(
        cards=[limit_card, limit_card, limit_card],
        drinks=[],
        turn=2,
        turns_remaining=7,
        stamina=28,
        genki=10,
        plan="sense",
    )
    ctrl.inject_frame(frame2)
    res2 = runner.run_once()

    assert res2.success is True
    assert len(res2.legal_actions) == 1
    assert res2.legal_actions[0].action_type == "end_turn"
    assert res2.decision is not None
    assert res2.decision.source == "rule_skip"
    assert res2.decision.prompt_tokens == 0


def test_cli_app_mock_mode_all_plans():
    """Verify CLI `demo/app.py --mode mock --plan all` executes cleanly with exit code 0."""
    exit_code = app_main(["--mode", "mock", "--plan", "all", "--steps", "2", "--delay", "0"])
    assert exit_code == 0


def test_cli_app_adb_mode_with_fallback(tmp_path):
    """Verify CLI `demo/app.py --mode adb --host 192.168.0.115 --port 5555` executes cleanly."""
    exit_code = app_main([
        "--mode", "adb",
        "--host", "192.168.0.115",
        "--port", "5555",
        "--steps", "1",
        "--delay", "0",
        "--checkpoint", str(tmp_path / "cli_adb_ckpt.json"),
        "--fresh",
    ])
    assert exit_code == 0


def test_live_xai_llm_decision_e2e(shared_db):
    """Verify live xAI LLM decision integration via AgentConfig.from_env_or_xai()."""
    cfg = AgentConfig.from_env_or_xai()
    assert cfg.api_key, "Expected xAI API key from env or c:\\Users\\josha\\.ssh\\XAIapikey.txt"
    agent = AgentDecisionMaker(cfg)

    cards = _pick_test_cards(shared_db, "sense")
    drinks = [
        d for d in shared_db.get_drinks_for_plan("sense")
        if d.get("icon_path") and Path(d["icon_path"]).exists()
    ][:1]

    ctrl = MockController(db=shared_db, plan="sense", auto_progress=False)
    ctrl.set_scenario(
        "battle_turn",
        cards=cards,
        drinks=drinks,
        turn=1,
        turns_remaining=8,
        stamina=30,
        genki=4,
        score=100,
        buffs={"goodConditionTurns": 3},
    )
    runner = GakumasRunner(
        mode="mock",
        controller=ctrl,
        agent=agent,
        db=shared_db,
        plan="sense",
    )
    runner.start()
    res = runner.run_once()
    assert res.success is True
    assert res.decision is not None
    assert res.decision.source == "llm"
    assert res.decision.prompt_tokens > 0
    assert 0 <= res.decision.choice_id < len(res.legal_actions)


def test_root_assets_read_only_integrity():
    """Verify root asset directories and files remain intact and unmodified."""
    root = Path(".").resolve()
    for rel in ("cards", "drinks", "reference", "gakumas_library.db", "manifest.json", "README.md"):
        assert (root / rel).exists(), f"Missing root asset: {rel}"

    conn = sqlite3.connect((root / "gakumas_library.db").as_uri() + "?mode=ro", uri=True)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0] == 926
        assert conn.execute("SELECT COUNT(*) FROM drinks").fetchone()[0] == 28
    finally:
        conn.close()
