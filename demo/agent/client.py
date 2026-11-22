"""
OpenAI-Compatible LLM Client and Decision Maker for Gakumas Autonomous Agent.
Supports synchronous (`decide`) and asynchronous (`adecide`) execution, zero-token
rule skip, strict JSON schema output enforcement, retries, and offline heuristic fallback.
demo/agent/client.py
"""

import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import openai
from pydantic import BaseModel, Field

from demo.schemas import DecisionResult, LegalAction, ScreenObservation
from demo.agent.prompts import DEFAULT_SYSTEM_PROMPT, get_session_system_prompt, get_system_prompt
from demo.agent.fallback import HeuristicDecisionEngine, HeuristicRuleEngine, check_single_legal_action

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """
    Configuration container for LLM Decision Maker with environment variable fallbacks.
    """
    model_config = {"extra": "ignore"}

    base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    )
    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    )
    timeout: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "15.0"))
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "2"))
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    enable_rule_skip: bool = True
    fallback_on_error: bool = True
    session_mode: bool = False

    @classmethod
    def from_env_or_xai(
        cls,
        key_file: str = r"c:\Users\josha\.ssh\XAIapikey.txt",
    ) -> "AgentConfig":
        """Load LLM config from environment variables, falling back to the local xAI key file if present."""
        env_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if env_key:
            return cls()
        p = Path(key_file)
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
            if key:
                return cls(
                    base_url=os.getenv("LLM_BASE_URL") or "https://api.x.ai/v1",
                    api_key=key,
                    model=os.getenv("LLM_MODEL") or "grok-build-0.1",
                    timeout=float(os.getenv("LLM_TIMEOUT", "45.0")),
                    max_retries=int(os.getenv("LLM_MAX_RETRIES", "1")),
                )
        return cls(api_key=None, fallback_on_error=True)


def parse_decision_json(content: str) -> Tuple[int, str]:
    """
    Robust JSON parser for LLM responses.
    Strips markdown code blocks and handles trailing or surrounding remarks.
    Returns (choice_id, reason).
    Raises ValueError on invalid formats, missing fields, or conversion errors.
    """
    if not content or not content.strip():
        raise ValueError("Empty response received from LLM")

    cleaned = content.strip()
    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Search for first embedded JSON object containing choice_id
        match = re.search(r"\{[^{}]*\"choice_id\"[^{}]*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Failed to parse JSON from response: {content[:100]}...")

    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON must be an object, got {type(data).__name__}")

    if "choice_id" not in data:
        raise ValueError("Missing required key 'choice_id' in response")

    try:
        choice_id = int(data["choice_id"])
    except (ValueError, TypeError, OverflowError) as err:
        raise ValueError(f"Invalid choice_id type, expected int, got: {data['choice_id']}") from err

    reason = str(data.get("reason", ""))
    return choice_id, reason


class AgentDecisionMaker:
    """
    Unified LLM Decision Maker supporting sync (`decide`) and async (`adecide`) entrypoints.
    Handles client initialization safely without raising when api_key is missing.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        fallback_engine: Optional[HeuristicDecisionEngine] = None,
    ):
        self.config = config or AgentConfig()
        self.fallback_engine = fallback_engine or HeuristicDecisionEngine()
        self.session_messages: List[Dict[str, str]] = []
        self._session_plan: Optional[str] = None
        self._session_scenario_id: Optional[str] = None
        self._session_p_idol_id: Optional[int] = None
        self._session_turns: int = 0

        # Determine client availability
        api_key_str = self.config.api_key.strip() if self.config.api_key else ""
        self.is_client_available = bool(api_key_str or self.config.base_url)

        self.client: Optional[openai.OpenAI] = None
        self.async_client: Optional[openai.AsyncOpenAI] = None
        self._request_timeout = openai.Timeout(
            self.config.timeout,
            connect=min(10.0, self.config.timeout),
        )

        if self.is_client_available:
            try:
                effective_key = api_key_str if api_key_str else "EMPTY"
                self.client = openai.OpenAI(
                    api_key=effective_key,
                    base_url=self.config.base_url,
                    timeout=self._request_timeout,
                    max_retries=0,
                )
                self.async_client = openai.AsyncOpenAI(
                    api_key=effective_key,
                    base_url=self.config.base_url,
                    timeout=self._request_timeout,
                    max_retries=0,
                )
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client: %s. Using offline mode.", e)
                self.client = None
                self.async_client = None
                self.is_client_available = False

    def decide(
        self,
        prompt: str,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation] = None,
        system_prompt: Optional[str] = None,
        use_session: Optional[bool] = None,
        scenario_id: Optional[str] = None,
        p_idol_id: Optional[int] = None,
    ) -> DecisionResult:
        """
        Synchronous decision entrypoint. When `use_session=True` (or `config.session_mode=True`),
        maintains a persistent single-produce conversation history across steps.
        """
        # 1. Edge Case: No legal actions
        if not legal_actions:
            return DecisionResult(
                choice_id=0,
                reason="无可用合法动作",
                source="fallback",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # 2. Rule Skip Optimization (0 tokens, 0ms)
        if len(legal_actions) == 1 and self.config.enable_rule_skip:
            act = legal_actions[0]
            return DecisionResult(
                choice_id=act.choice_id,
                reason=f"唯一合法动作自动跳过LLM: {act.description}",
                source="rule_skip",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # 3. Offline Heuristic Fallback
        if not self.is_client_available or self.client is None:
            return self._run_fallback(
                legal_actions=legal_actions,
                obs=obs,
                reason_prefix="无LLM凭证，启用离线启发式规则",
            )

        # 4. LLM Query Loop with Retries
        session_active = self.config.session_mode if use_session is None else bool(use_session)
        valid_choice_ids = {a.choice_id for a in legal_actions}
        messages = self._prepare_messages(prompt, obs, system_prompt, session_active, scenario_id, p_idol_id)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        start_time = time.perf_counter()

        for attempt in range(self.config.max_retries + 1):
            try:
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                        response_format={"type": "json_object"},
                        timeout=self._request_timeout,
                    )
                except openai.BadRequestError as bad_req:
                    trimmed = self._trim_session_on_context_overflow(messages, bad_req) if session_active else None
                    if trimmed is not None:
                        messages = trimmed
                    # Retry without response_format if provider doesn't support json_object or after trim
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                        timeout=self._request_timeout,
                    )
            except Exception as e:
                logger.warning("LLM call attempt %d failed: %s", attempt, e)
                if attempt < self.config.max_retries:
                    continue
                if self.config.fallback_on_error:
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    res = self._run_fallback(legal_actions, obs, f"LLM异常({type(e).__name__})兜底")
                    res.latency_ms = round(elapsed, 2)
                    res.prompt_tokens = total_prompt_tokens
                    res.completion_tokens = total_completion_tokens
                    return res
                raise

            # Track tokens
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens or 0
                total_completion_tokens += response.usage.completion_tokens or 0

            raw_text = response.choices[0].message.content or ""

            # Parse JSON
            try:
                choice_id, reason = parse_decision_json(raw_text)
            except ValueError as parse_err:
                logger.warning("JSON parse failed on attempt %d: %s", attempt, parse_err)
                if attempt < self.config.max_retries:
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({
                        "role": "user",
                        "content": f"输出格式错误: 无法解析为有效 JSON ({parse_err})。请严格只输出标准 JSON 格式: {{\"choice_id\": <合法编号>, \"reason\": \"<简短理由>\"}}"
                    })
                    continue
                if self.config.fallback_on_error:
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    res = self._run_fallback(legal_actions, obs, f"JSON解析失败兜底: {parse_err}")
                    res.latency_ms = round(elapsed, 2)
                    res.prompt_tokens = total_prompt_tokens
                    res.completion_tokens = total_completion_tokens
                    return res
                raise

            # Validate choice_id
            if choice_id not in valid_choice_ids:
                logger.warning("choice_id=%d out of bounds on attempt %d (valid: %s)", choice_id, attempt, valid_choice_ids)
                if attempt < self.config.max_retries:
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({
                        "role": "user",
                        "content": f"错误: choice_id={choice_id} 不在当前合法动作编号列表 {sorted(valid_choice_ids)} 中。请重新从合法编号中选择，并严格输出 JSON 格式: {{\"choice_id\": <int>, \"reason\": \"<str>\"}}"
                    })
                    continue
                if self.config.fallback_on_error:
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    res = self._run_fallback(legal_actions, obs, f"越界编号 choice_id={choice_id} 兜底")
                    res.latency_ms = round(elapsed, 2)
                    res.prompt_tokens = total_prompt_tokens
                    res.completion_tokens = total_completion_tokens
                    return res
                raise ValueError(f"Choice ID {choice_id} is not in legal actions {sorted(valid_choice_ids)}")

            # Successfully obtained valid decision; commit clean exchange if in persistent session
            if session_active:
                self._commit_session_turn(
                    prompt,
                    choice_id,
                    reason,
                    system_msg=messages[0]["content"],
                    plan=obs.plan if obs else "logic",
                    scenario_id=scenario_id or (obs.scenario_id if obs else "hajime_regular"),
                    p_idol_id=p_idol_id,
                )
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return DecisionResult(
                choice_id=choice_id,
                reason=reason,
                source="llm",
                latency_ms=round(elapsed, 2),
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )

    async def adecide(
        self,
        prompt: str,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation] = None,
        system_prompt: Optional[str] = None,
        use_session: Optional[bool] = None,
        scenario_id: Optional[str] = None,
        p_idol_id: Optional[int] = None,
    ) -> DecisionResult:
        """
        Asynchronous decision entrypoint.
        """
        # 1. Edge Case: No legal actions
        if not legal_actions:
            return DecisionResult(
                choice_id=0,
                reason="无可用合法动作",
                source="fallback",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # 2. Rule Skip Optimization (0 tokens, 0ms)
        if len(legal_actions) == 1 and self.config.enable_rule_skip:
            act = legal_actions[0]
            return DecisionResult(
                choice_id=act.choice_id,
                reason=f"唯一合法动作自动跳过LLM: {act.description}",
                source="rule_skip",
                latency_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
            )

        # 3. Offline Heuristic Fallback
        if not self.is_client_available or self.async_client is None:
            return self._run_fallback(
                legal_actions=legal_actions,
                obs=obs,
                reason_prefix="无LLM凭证，启用离线启发式规则",
            )

        # 4. LLM Query Loop with Retries
        session_active = self.config.session_mode if use_session is None else bool(use_session)
        valid_choice_ids = {a.choice_id for a in legal_actions}
        messages = self._prepare_messages(prompt, obs, system_prompt, session_active, scenario_id, p_idol_id)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        start_time = time.perf_counter()

        for attempt in range(self.config.max_retries + 1):
            try:
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                        response_format={"type": "json_object"},
                        timeout=self._request_timeout,
                    )
                except openai.BadRequestError as bad_req:
                    trimmed = self._trim_session_on_context_overflow(messages, bad_req) if session_active else None
                    if trimmed is not None:
                        messages = trimmed
                    response = await self.async_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=self.config.temperature,
                        timeout=self._request_timeout,
                    )
            except Exception as e:
                logger.warning("Async LLM call attempt %d failed: %s", attempt, e)
                if attempt < self.config.max_retries:
                    continue
                if self.config.fallback_on_error:
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    res = self._run_fallback(legal_actions, obs, f"LLM异常({type(e).__name__})兜底")
                    res.latency_ms = round(elapsed, 2)
                    res.prompt_tokens = total_prompt_tokens
                    res.completion_tokens = total_completion_tokens
                    return res
                raise

            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens or 0
                total_completion_tokens += response.usage.completion_tokens or 0

            raw_text = response.choices[0].message.content or ""

            try:
                choice_id, reason = parse_decision_json(raw_text)
            except ValueError as parse_err:
                logger.warning("Async JSON parse failed on attempt %d: %s", attempt, parse_err)
                if attempt < self.config.max_retries:
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({
                        "role": "user",
                        "content": f"输出格式错误: 无法解析为有效 JSON ({parse_err})。请严格只输出标准 JSON 格式: {{\"choice_id\": <合法编号>, \"reason\": \"<简短理由>\"}}"
                    })
                    continue
                if self.config.fallback_on_error:
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    res = self._run_fallback(legal_actions, obs, f"JSON解析失败兜底: {parse_err}")
                    res.latency_ms = round(elapsed, 2)
                    res.prompt_tokens = total_prompt_tokens
                    res.completion_tokens = total_completion_tokens
                    return res
                raise

            if choice_id not in valid_choice_ids:
                logger.warning("Async choice_id=%d out of bounds on attempt %d (valid: %s)", choice_id, attempt, valid_choice_ids)
                if attempt < self.config.max_retries:
                    messages.append({"role": "assistant", "content": raw_text})
                    messages.append({
                        "role": "user",
                        "content": f"错误: choice_id={choice_id} 不在当前合法动作编号列表 {sorted(valid_choice_ids)} 中。请重新从合法编号中选择，并严格输出 JSON 格式: {{\"choice_id\": <int>, \"reason\": \"<str>\"}}"
                    })
                    continue
                if self.config.fallback_on_error:
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                    res = self._run_fallback(legal_actions, obs, f"越界编号 choice_id={choice_id} 兜底")
                    res.latency_ms = round(elapsed, 2)
                    res.prompt_tokens = total_prompt_tokens
                    res.completion_tokens = total_completion_tokens
                    return res
                raise ValueError(f"Choice ID {choice_id} is not in legal actions {sorted(valid_choice_ids)}")

            if session_active:
                self._commit_session_turn(
                    prompt,
                    choice_id,
                    reason,
                    system_msg=messages[0]["content"],
                    plan=obs.plan if obs else "logic",
                    scenario_id=scenario_id or (obs.scenario_id if obs else "hajime_regular"),
                    p_idol_id=p_idol_id,
                )
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return DecisionResult(
                choice_id=choice_id,
                reason=reason,
                source="llm",
                latency_ms=round(elapsed, 2),
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )

    def reset_session(self) -> None:
        """Clear persistent Produce conversation history so the next run bootstraps cleanly."""
        self.session_messages.clear()
        self._session_plan = None
        self._session_scenario_id = None
        self._session_p_idol_id = None
        self._session_turns = 0

    def get_session_stats(self) -> Dict[str, Any]:
        """Return persistent session conversation metrics for logs and UI."""
        return {
            "active": bool(self.session_messages),
            "turns": self._session_turns,
            "messages": len(self.session_messages),
            "plan": self._session_plan,
            "scenario_id": self._session_scenario_id,
            "p_idol_id": self._session_p_idol_id,
        }

    def export_session(self) -> Dict[str, Any]:
        """Serialize persistent conversation history for process-restart checkpoint recovery."""
        return {
            "messages": [{"role": str(m["role"]), "content": str(m["content"])} for m in self.session_messages],
            "plan": self._session_plan,
            "scenario_id": self._session_scenario_id,
            "p_idol_id": self._session_p_idol_id,
            "turns": self._session_turns,
        }

    def load_session(self, data: Dict[str, Any]) -> bool:
        """Restore persistent conversation history from checkpoint dict if structurally valid."""
        if not isinstance(data, dict):
            return False
        raw_msgs = data.get("messages")
        if not isinstance(raw_msgs, list):
            return False
        cleaned: List[Dict[str, str]] = []
        for msg in raw_msgs:
            if isinstance(msg, dict) and isinstance(msg.get("role"), str) and isinstance(msg.get("content"), str):
                cleaned.append({"role": msg["role"], "content": msg["content"]})
        if not cleaned:
            self.reset_session()
            return False
        self.session_messages = cleaned
        self._session_plan = str(data["plan"]).lower() if data.get("plan") else None
        self._session_scenario_id = str(data["scenario_id"]) if data.get("scenario_id") else None
        raw_pid = data.get("p_idol_id")
        self._session_p_idol_id = int(raw_pid) if raw_pid is not None else None
        self._session_turns = int(data.get("turns", max(0, (len(cleaned) - 1) // 2)))
        total_chars = sum(len(m.get("content", "")) for m in self.session_messages)
        if len(self.session_messages) > 75 or total_chars > 130_000:
            self._compact_session_preserving_archives(keep_tail_pairs=8)
        return True

    def _compact_session_preserving_archives(self, keep_tail_pairs: int = 8) -> None:
        """Trim middle session turns while keeping system, initial bootstrap, archive-unlock turns, and recent tail."""
        tail_msgs = max(2, keep_tail_pairs * 2)
        if len(self.session_messages) <= 3 + tail_msgs:
            return
        head = list(self.session_messages[:3])
        mid_end = len(self.session_messages) - tail_msgs
        archived_mid: List[Dict[str, str]] = []
        idx = 3
        while idx + 1 < mid_end:
            u_msg = self.session_messages[idx]
            a_msg = self.session_messages[idx + 1]
            if (
                u_msg.get("role") == "user"
                and a_msg.get("role") == "assistant"
                and "档案收录】" in u_msg.get("content", "")
            ):
                archived_mid.extend((u_msg, a_msg))
            idx += 2
        tail = list(self.session_messages[-tail_msgs:])
        compacted = head + archived_mid + tail
        if sum(len(m.get("content", "")) for m in compacted) > 140_000:
            compacted = head + tail
        self.session_messages = compacted
        logger.info("Proactively compacted persistent session history to %d messages.", len(compacted))

    def _prepare_messages(
        self,
        prompt: str,
        obs: Optional[ScreenObservation],
        system_prompt: Optional[str],
        use_session: bool,
        scenario_id: Optional[str],
        p_idol_id: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        if not use_session:
            sys_prompt = system_prompt or self._resolve_system_prompt(obs, p_idol_id=p_idol_id)
            return [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]
        plan = obs.plan if obs else "logic"
        sc_id = scenario_id or (obs.scenario_id if obs else "hajime_regular")
        if not self.session_messages or (self._session_plan and self._session_plan != plan):
            sys_p = system_prompt or get_session_system_prompt(plan=plan, scenario_id=sc_id, p_idol_id=p_idol_id)
            return [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt},
            ]
        return list(self.session_messages) + [{"role": "user", "content": prompt}]

    def _commit_session_turn(
        self,
        user_prompt: str,
        choice_id: int,
        reason: str,
        system_msg: Optional[str] = None,
        plan: Optional[str] = None,
        scenario_id: Optional[str] = None,
        p_idol_id: Optional[int] = None,
    ) -> None:
        if not self.session_messages or (plan and self._session_plan and self._session_plan != plan):
            sys_p = system_msg or get_session_system_prompt(plan=plan, scenario_id=scenario_id, p_idol_id=p_idol_id)
            self.session_messages = [{"role": "system", "content": sys_p}]
            self._session_plan = plan
            self._session_scenario_id = scenario_id
            self._session_p_idol_id = p_idol_id
            self._session_turns = 0
        self.session_messages.append({"role": "user", "content": user_prompt})
        self.session_messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"choice_id": choice_id, "reason": reason}, ensure_ascii=False),
            }
        )
        self._session_turns += 1
        total_chars = sum(len(m.get("content", "")) for m in self.session_messages)
        if len(self.session_messages) > 75 or total_chars > 130_000:
            self._compact_session_preserving_archives(keep_tail_pairs=8)

    def _trim_session_on_context_overflow(
        self,
        messages: List[Dict[str, str]],
        err: Exception,
    ) -> Optional[List[Dict[str, str]]]:
        body_str = str(getattr(err, "body", "") or "")
        resp_txt = getattr(getattr(err, "response", None), "text", None)
        resp_str = resp_txt if isinstance(resp_txt, str) else ""
        err_text = f"{err} {body_str} {resp_str}".lower()
        if not any(tok in err_text for tok in ("context", "token", "length", "maximum", "too_large", "too long")):
            return None
        if len(self.session_messages) <= 5:
            return None
        # Preserve [system, bootstrap_user, bootstrap_assistant] plus most recent 6 pairs
        kept = self.session_messages[:3] + self.session_messages[-12:]
        self.session_messages = kept
        logger.warning("Trimmed persistent session history to %d messages due to context limit.", len(kept))
        return list(self.session_messages) + [messages[-1]]

    def _resolve_system_prompt(
        self,
        obs: Optional[ScreenObservation],
        p_idol_id: Optional[int] = None,
    ) -> str:
        if obs:
            return get_system_prompt(plan=obs.plan, scene_type=obs.scene_type, compact=False, p_idol_id=p_idol_id)
        return DEFAULT_SYSTEM_PROMPT

    def _run_fallback(
        self,
        legal_actions: List[LegalAction],
        obs: Optional[ScreenObservation] = None,
        reason_prefix: str = "",
    ) -> DecisionResult:
        res = self.fallback_engine.decide(legal_actions=legal_actions, obs=obs)
        if reason_prefix:
            res.reason = f"{reason_prefix} | {res.reason}"
        return res
