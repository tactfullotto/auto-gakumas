"""
Agent module for Gakumas Autonomous Decision and Automation System.
Provides LLM decision-making, prompt templates, and deterministic heuristic fallback.
"""

from demo.agent.client import (
    AgentConfig,
    AgentDecisionMaker,
    parse_decision_json,
)
from demo.agent.fallback import (
    HeuristicDecisionEngine,
    HeuristicRuleEngine,
    check_single_legal_action,
)
from demo.agent.prompts import (
    CARD_REWARD_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DETAILED_SYSTEM_PROMPT,
    ANOMALY_SYSTEM_PROMPT,
    LOGIC_SYSTEM_PROMPT,
    PRODUCE_SCHEDULE_PROMPT,
    SENSE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    estimate_tokens,
    get_system_prompt,
)

__all__ = [
    "AgentConfig",
    "AgentDecisionMaker",
    "parse_decision_json",
    "HeuristicDecisionEngine",
    "HeuristicRuleEngine",
    "check_single_legal_action",
    "SYSTEM_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "DETAILED_SYSTEM_PROMPT",
    "SENSE_SYSTEM_PROMPT",
    "LOGIC_SYSTEM_PROMPT",
    "ANOMALY_SYSTEM_PROMPT",
    "PRODUCE_SCHEDULE_PROMPT",
    "CARD_REWARD_PROMPT",
    "get_system_prompt",
    "estimate_tokens",
]
