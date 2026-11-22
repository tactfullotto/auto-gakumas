"""Headless Gakumas battle simulator package (lazy exports to avoid runpy warnings)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ActionSpec",
    "BattleConfig",
    "BattleResult",
    "CardConfig",
    "DrinkConfig",
    "GakumasActionCodec",
    "GakumasBattleEnv",
    "GakumasFeatureEncoder",
    "HeadlessBattleSimulator",
    "ISMCTSAgent",
    "ISMCTSConfig",
    "MaskablePPOConfig",
    "MaskablePPOTrainer",
    "PItemConfig",
    "PolicyValueNet",
    "TurnMultiplierSpec",
    "build_last_run_hif_preset",
    "describe_loaded_config",
    "generate_stage_turn_schedule",
]

_EXPORT_MODULES = {
    "ActionSpec": "demo.sim.rl_env",
    "GakumasActionCodec": "demo.sim.rl_env",
    "GakumasBattleEnv": "demo.sim.rl_env",
    "GakumasFeatureEncoder": "demo.sim.rl_env",
    "ISMCTSAgent": "demo.sim.ismcts",
    "ISMCTSConfig": "demo.sim.ismcts",
    "MaskablePPOConfig": "demo.sim.maskable_ppo",
    "MaskablePPOTrainer": "demo.sim.maskable_ppo",
    "PolicyValueNet": "demo.sim.maskable_ppo",
}


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod_name = _EXPORT_MODULES.get(name, "demo.sim.battle_simulator")
        mod = import_module(mod_name)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
