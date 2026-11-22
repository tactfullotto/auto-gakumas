"""
Unit tests for demo/agent/client.py.
Verifies AgentConfig, safe initialization without credentials, robust JSON parsing,
synchronous decide(), asynchronous adecide(), multi-round retries on invalid JSON or
out-of-bounds choice_id, and automatic heuristic fallback on errors or exhausted retries.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from demo.schemas import BoundingBox, DecisionResult, LegalAction, ScreenObservation
from demo.agent.client import AgentConfig, AgentDecisionMaker, parse_decision_json
from demo.agent.fallback import HeuristicDecisionEngine


def make_dummy_actions(count: int = 3) -> list[LegalAction]:
    actions = []
    for i in range(count):
        actions.append(
            LegalAction(
                choice_id=i,
                action_type="play_card" if i < 2 else "end_turn",
                description=f"动作 {i}",
                bbox=BoundingBox(xmin=100 * i, ymin=100, xmax=100 * i + 80, ymax=200),
                payload={"hp_loss": i * 5, "name": f"Card_{i}"},
            )
        )
    return actions


class TestAgentConfig:
    def test_default_config(self):
        cfg = AgentConfig()
        assert cfg.model == "gpt-4o-mini"
        assert cfg.timeout == 15.0
        assert cfg.max_retries == 2
        assert cfg.enable_rule_skip is True
        assert cfg.fallback_on_error is True

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://api.my-llm.com/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-custom-secret-key")
        monkeypatch.setenv("LLM_MODEL", "deepseek-coder")
        monkeypatch.setenv("LLM_TIMEOUT", "30.0")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        monkeypatch.setenv("LLM_TEMPERATURE", "0.5")

        cfg = AgentConfig()
        assert cfg.base_url == "https://api.my-llm.com/v1"
        assert cfg.api_key == "sk-custom-secret-key"
        assert cfg.model == "deepseek-coder"
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 5
        assert cfg.temperature == 0.5

    def test_explicit_constructor_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL", "env-model")
        cfg = AgentConfig(model="override-model", timeout=5.0)
        assert cfg.model == "override-model"
        assert cfg.timeout == 5.0


class TestJSONParser:
    def test_standard_json(self):
        raw = '{"choice_id": 2, "reason": "打出核心卡"}'
        cid, reason = parse_decision_json(raw)
        assert cid == 2
        assert reason == "打出核心卡"

    def test_markdown_code_fences(self):
        raw = '```json\n{"choice_id": 0, "reason": "code fence test"}\n```'
        cid, reason = parse_decision_json(raw)
        assert cid == 0
        assert reason == "code fence test"

    def test_embedded_json(self):
        raw = '思考过程：当前好调层数高，建议爆发。\n{"choice_id": 1, "reason": "爆发"}\n总结完毕。'
        cid, reason = parse_decision_json(raw)
        assert cid == 1
        assert reason == "爆发"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError):
            parse_decision_json("plain text without json")

    def test_missing_choice_id_raises(self):
        with pytest.raises(ValueError):
            parse_decision_json('{"action": 1, "reason": "no choice id"}')

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_decision_json("")

    def test_infinity_choice_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid choice_id"):
            parse_decision_json('{"choice_id": Infinity, "reason": "overflow test"}')

    def test_negative_infinity_choice_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid choice_id"):
            parse_decision_json('{"choice_id": -Infinity, "reason": "negative overflow"}')

    def test_nan_choice_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid choice_id"):
            parse_decision_json('{"choice_id": NaN, "reason": "nan test"}')

    def test_string_non_numeric_choice_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid choice_id"):
            parse_decision_json('{"choice_id": "not_a_number", "reason": "string invalid"}')


class TestAgentDecisionMaker:
    def test_init_without_api_key_safe(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        maker = AgentDecisionMaker(config=AgentConfig(api_key=None, base_url=None))
        assert maker.is_client_available is False
        assert maker.client is None
        assert maker.async_client is None

    def test_zero_actions_fallback(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key=None, base_url=None))
        res = maker.decide(prompt="test", legal_actions=[])
        assert res.choice_id == 0
        assert res.source == "fallback"
        assert "无可用合法动作" in res.reason

    def test_rule_skip_single_action(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-dummy"))
        actions = make_dummy_actions(1)
        res = maker.decide(prompt="test", legal_actions=actions)
        assert res.choice_id == actions[0].choice_id
        assert res.source == "rule_skip"
        assert res.prompt_tokens == 0
        assert res.completion_tokens == 0
        assert res.latency_ms == 0.0

    def test_offline_fallback_multiple_actions(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key=None, base_url=None))
        actions = make_dummy_actions(3)
        obs = ScreenObservation(scene_type="battle_turn", stamina=30)
        res = maker.decide(prompt="test", legal_actions=actions, obs=obs)
        assert res.source == "fallback"
        assert res.choice_id in {0, 1, 2}
        assert "无LLM凭证" in res.reason

    def test_sync_decide_llm_success(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-test"))
        actions = make_dummy_actions(3)

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(message=MagicMock(content='{"choice_id": 1, "reason": "精准爆发"}'))
        ]
        mock_resp.usage = MagicMock(prompt_tokens=150, completion_tokens=30)

        with patch.object(maker.client.chat.completions, "create", return_value=mock_resp):
            res = maker.decide(prompt="test prompt", legal_actions=actions)
            assert res.choice_id == 1
            assert res.source == "llm"
            assert res.reason == "精准爆发"
            assert res.prompt_tokens == 150
            assert res.completion_tokens == 30
            assert res.latency_ms > 0

    def test_decide_retry_on_invalid_json(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-test", max_retries=2))
        actions = make_dummy_actions(3)

        mock_resp_fail = MagicMock()
        mock_resp_fail.choices = [MagicMock(message=MagicMock(content="Malformed text"))]
        mock_resp_fail.usage = MagicMock(prompt_tokens=50, completion_tokens=10)

        mock_resp_ok = MagicMock()
        mock_resp_ok.choices = [
            MagicMock(message=MagicMock(content='{"choice_id": 0, "reason": "重试成功"}'))
        ]
        mock_resp_ok.usage = MagicMock(prompt_tokens=70, completion_tokens=15)

        with patch.object(maker.client.chat.completions, "create", side_effect=[mock_resp_fail, mock_resp_ok]):
            res = maker.decide(prompt="test", legal_actions=actions)
            assert res.choice_id == 0
            assert res.source == "llm"
            assert res.prompt_tokens == 120  # Accumulated tokens (50 + 70)
            assert res.completion_tokens == 25

    def test_decide_retry_on_out_of_bounds_choice(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-test", max_retries=2))
        actions = make_dummy_actions(2)  # choice_ids are 0 and 1

        mock_resp_out_of_bounds = MagicMock()
        mock_resp_out_of_bounds.choices = [
            MagicMock(message=MagicMock(content='{"choice_id": 99, "reason": "超范围"}'))
        ]
        mock_resp_out_of_bounds.usage = MagicMock(prompt_tokens=50, completion_tokens=10)

        mock_resp_valid = MagicMock()
        mock_resp_valid.choices = [
            MagicMock(message=MagicMock(content='{"choice_id": 1, "reason": "合法选择"}'))
        ]
        mock_resp_valid.usage = MagicMock(prompt_tokens=60, completion_tokens=10)

        with patch.object(
            maker.client.chat.completions,
            "create",
            side_effect=[mock_resp_out_of_bounds, mock_resp_valid],
        ):
            res = maker.decide(prompt="test", legal_actions=actions)
            assert res.choice_id == 1
            assert res.source == "llm"
            assert res.prompt_tokens == 110

    def test_decide_fallback_on_network_error(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-test", max_retries=1))
        actions = make_dummy_actions(3)

        with patch.object(
            maker.client.chat.completions,
            "create",
            side_effect=Exception("Connection refused"),
        ):
            res = maker.decide(prompt="test", legal_actions=actions)
            assert res.source == "fallback"
            assert "LLM异常" in res.reason
            assert res.choice_id in {0, 1, 2}

    def test_decide_fallback_on_exhausted_retries(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-test", max_retries=1))
        actions = make_dummy_actions(3)

        mock_bad = MagicMock()
        mock_bad.choices = [MagicMock(message=MagicMock(content="not valid json"))]
        mock_bad.usage = MagicMock(prompt_tokens=30, completion_tokens=5)

        with patch.object(
            maker.client.chat.completions,
            "create",
            side_effect=[mock_bad, mock_bad],
        ):
            res = maker.decide(prompt="test", legal_actions=actions)
            assert res.source == "fallback"
            assert "JSON解析失败兜底" in res.reason

    @pytest.mark.anyio
    async def test_async_adecide_success(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-test"))
        actions = make_dummy_actions(3)

        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(message=MagicMock(content='{"choice_id": 2, "reason": "async pass"}'))
        ]
        mock_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=20)

        with patch.object(
            maker.async_client.chat.completions, "create", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_resp
            res = await maker.adecide(prompt="test async", legal_actions=actions)
            assert res.choice_id == 2
            assert res.source == "llm"
            assert res.prompt_tokens == 100
            assert res.completion_tokens == 20

    def test_decide_retry_on_infinity_choice_id(self):
        maker = AgentDecisionMaker(config=AgentConfig(api_key="sk-test", max_retries=2))
        actions = make_dummy_actions(3)

        mock_resp_fail = MagicMock()
        mock_resp_fail.choices = [MagicMock(message=MagicMock(content='{"choice_id": Infinity, "reason": "overflow"}'))]
        mock_resp_fail.usage = MagicMock(prompt_tokens=40, completion_tokens=10)

        mock_resp_ok = MagicMock()
        mock_resp_ok.choices = [MagicMock(message=MagicMock(content='{"choice_id": 1, "reason": "recovered"}'))]
        mock_resp_ok.usage = MagicMock(prompt_tokens=60, completion_tokens=15)

        with patch.object(maker.client.chat.completions, "create", side_effect=[mock_resp_fail, mock_resp_ok]):
            res = maker.decide(prompt="test infinity retry", legal_actions=actions)
            assert res.choice_id == 1
            assert res.source == "llm"
            assert res.reason == "recovered"
            assert res.prompt_tokens == 100
