# auto-gakumas

[中文](README.zh.md) | [**English**](README.md) | [日本語](README.ja.md)

> **Zero-LLM** end-to-end automation for the **学園アイドルマスター (Gakumas) H.I.F 本戦** card game — from produce schedule to final battle, fully automated via ADB + local vision + PPO/ISMCTS planner.

---

## Overview

`auto-gakumas` drives MuMu emulator through an entire Gakumas produce cycle:

1. **Schedule & Lessons** — follows a fixed weekly schedule, selects lessons and rest.
2. **Card Rewards** — evaluates and picks card rewards; auto-discards banned/low-value cards; retries shop refresh if unremovable M-cards appear in swap screen; protects customized cards on mandatory swaps.
3. **Shop** — upgrades cards, buys drinks, spends remaining P-points before final departure; refreshes shop to maximize spend in the last visit.
4. **H.I.F 本戦 (Rounds 1 & 2)** — plays both battle rounds using a local `PPO + SO-ISMCTS` decision engine with no cloud API calls.

The entire pipeline runs offline. No LLM or remote inference is required at runtime.

---

## Architecture

```
ADB (MuMu 127.0.0.1:16384)
        │
        ▼
LiveScreenRecognizer          ← OpenCV + Tesseract OCR
  • scene detection & L-corner deck selection
  • card/drink/score OCR & customization badge check
  • P-point sandwich OCR (outlined white digits, anti-occlusion)
  • italic-deshear turn-ring OCR + 2-frame drop recovery
  • battle hand animation stabiliser & stuck-selection retry
        │
        ▼
LocalHybridDecisionAgent      ← rule engine + local ML
  ├── ScheduleAgent            (lesson / rest selection)
  ├── CardRewardAgent          (card pick, swap protection & discard rules)
  ├── ShopAgent                (upgrade / buy / refresh + visit tracking)
  └── LiveBattlePlanner        ← PPO value network + SO-ISMCTS + step self-healing
            │
            └── HeadlessBattleSimulator  (fast local rollouts & P-item triggers)
```

### Key modules

| Path | Description |
|---|---|
| `demo/run_hif_local.py` | Main entry point; supports `on_step_callback` / `on_frame_callback` hooks |
| `demo/agent/local_agent.py` | `LocalHybridDecisionAgent` orchestrator; P-idol stat fallback; shop visit state machine; mandatory swap protection |
| `demo/maa_bridge/runner.py` | Step loop; battle hand animation wait; 5-card fanned tap clamping & `選択中` stuck-card retry; upgrade popup disambiguation |
| `demo/vision/live_adb_recognizer.py` | `LiveScreenRecognizer`; P-point sandwich OCR; italic-deshear turn-ring OCR (dimmed-HUD fallback); shop icon cache |
| `demo/sim/live_battle_planner.py` | `LiveBattlePlanner`; finisher combo gating (好調 ≥ 13); resource-zeroing finisher rules; step-feedback self-healing & turn reconciliation |
| `demo/sim/battle_simulator.py` | `HeadlessBattleSimulator`; 5-card hand cap; resource-zeroing finisher & 2x-forbid rules; Sense P-items (`#27092 届けたい想い`, etc.) |
| `demo/sim/ismcts.py` | SO-ISMCTS; anti-SKIP pruning; trouble-exile fallback; zeroing-finisher pruning |
| `demo/sim/rl_env.py` | `GakumasActionCodec`; legal action masking & dominated-action pruning |
| `demo/core/idol_guides.py` | P-idol default stat lookup & stat-axis ordering |
| `demo/core/hif_catalog.py` | `ProducePreferences`, HIF memory ability templates, plan metadata |
| `demo/core/tracker.py` | In-run state tracker (deck, hand, P-items, produce step) |
| `demo/web/hif_session.py` | Web workbench session manager (SSE, loadout config, live ADB preview) |
| `demo/web/static/hif_workbench.js` | 3-profession configurator + multi-filter card preference editor |
| `gakumas_library.db` | SQLite card & drink database (926 cards, 28 drinks) |

---

## H.I.F Workbench Web UI

A local web interface for configuring and launching produce runs without touching the CLI.

**Launch:**
```bash
python -m demo.web.server --host 127.0.0.1 --port 8765 --open
```

**Features:**

- **3-profession selector** (Sense / Logic / Anomaly) with P-Idol catalog and stat preview
- **Loadout modes:**
  - `auto` — fully rely on ADB scan to detect deck/P-items at run start
  - `manual` — skip scan; inject a hand-crafted deck + P-item list directly
  - `hybrid` — ADB scan + manually specified H.I.F memory hooks merged at start
- **Manual card/P-item editor** — build your deck card by card from the full DB
- **H.I.F Memory editor** — 4-slot memory cards, passive bonuses, and wiki preset groups
- **Produce Preferences panel** — multi-filter card selector (type / rarity / upgraded / effect keyword / assignment status); set priority/avoid/keep card lists; banned drink IDs
- **Live ADB preview** — real-time screen mirror during a run
- **Library JSON import/export** — save and reload full loadout configs
- **SSE progress stream** — live step-by-step automation log in the browser

---

## Special Rules

### Drink #19 — 初星スペシャル青汁 (banned)
Never select, buy, or use. Auto-discarded via `捨てる` if received.

### Drink #25 — 特製ハツボシエキス (strict gating)
Only consumed when **all three** conditions hold:
1. A non-Legend Active card (excluding resource-zeroing finishers) is already in hand.
2. The active card is playable immediately after the drink effect.
3. Commits to playing that Active card the same turn.

### Shop — spend all P-points
On the final shop visit, refresh the store until all P-points are exhausted. Upgrades prioritised over new cards.

### Card swap — M-card refresh & customization protection
- If an unremovable M-card appears in the early-game card exchange, click refresh to reroll.
- On mandatory H.I.F `チェンジ` screens, customized cards (`カスタマイズ済`) are protected by immediately switching to a non-customized slot instead of looping on cancel.

### Battle — finisher combo & resource-zeroing finisher rules
- **Scaling finisher (`#162/#163 → #813/#814`)**: combo is only allowed when tracked `goodConditionTurns ≥ 13` (or OCR is occluded in the last 2 turns while the finisher is visually playable).
- **Resource-zeroing solo finishers (`#618/#619 世界一の勇気`, `#152/#153 届いて`)**: held strictly until the final turn (`turns_remaining == 1`), never paired with armed `2x` (`#162/#163` or `#25`) so the second trigger doesn't fire on a zeroed stat, and tutored only on the final turn if missing from hand.

### Battle — tap reliability & simulator self-healing
- **Hand animation stabiliser**: waits for hand-area motion < 6.0 mean-pixel-delta (up to 3 probes × 0.85 s) before reading cards.
- **5-card fanned tap clamping & retry**: clamps outer card taps inward (`13.5%–86.5%` width), confirms on the lifted hitbox, and retries automatically if a card remains stuck in `選択中`.
- **Step-feedback self-healing**: if a `limit=1` card fails to fire on device and stays in hand, rolls back phantom simulator state (`double_card_effect_cards`, `cards_used`, premature turn advance) and reconciles remaining turns against hand changes.

---

## Requirements

- Python 3.11
- MuMu Player (Android emulator) with ADB at `127.0.0.1:16384` (**Recommended display config: `720×1600` resolution, `320 DPI`, portrait**)
- Gakumas installed in emulator
- `gakumas_library.db` (included in repo)

> **Recommended Emulator Settings (`720×1600`, `DPI 320`)**
> All vision ROIs (shop grids, card/drink icons, L-corner selection brackets, HUD P-points, and fanned hand tap regions) are calibrated for portrait **`720×1600` @ `320 DPI`**. Using this exact display configuration is strongly recommended for reliable OCR and tap accuracy.

```bash
pip install -r demo/requirements.txt
```

Also requires `tesseract` and `adb` on PATH.

---

## Usage

### CLI (headless)
```bash
python -m demo.run_hif_local \
    --adb 127.0.0.1:16384 \
    --idol sense \
    --log-prefix live8
```

### Web UI
```bash
python -m demo.web.server --host 127.0.0.1 --port 8765 --open
# Opens http://127.0.0.1:8765  →  H.I.F Workbench tab
```

---

## Tests

```bash
python -m pytest demo/tests/ -k "not test_live_xai_llm_decision_e2e" -q
```

342 offline unit & integration tests (excluding live-device E2E).

---

## Live Results

| Run | Rounds | Round 2 Score |
|-----|--------|---------------|
| live7 | 2/2 ✅ | +1,594,997 |

---

## Data Sources

Card and drink data from [gakumas-tools/gakumas-tools](https://github.com/gakumas-tools/gakumas-tools) (commit `a82c3a3`, 2026-11-15).

---

## License

Research / personal use. Game assets © Bandai Namco Entertainment.
