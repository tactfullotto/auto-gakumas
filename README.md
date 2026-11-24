# auto-gakumas

[中文](README.zh.md) | [**English**](README.md) | [日本語](README.ja.md)

> **Zero-LLM** end-to-end automation for the **学園アイドルマスター (Gakumas) H.I.F 本戦** card game — from produce schedule to final battle, fully automated via ADB + local vision + PPO/ISMCTS planner.

---

## Overview

`auto-gakumas` drives MuMu emulator through an entire Gakumas produce cycle:

1. **Schedule & Lessons** — follows a fixed weekly schedule, selects lessons and rest.
2. **Card Rewards** — evaluates and picks card rewards; auto-discards banned/low-value cards; retries shop refresh if unremovable M-cards appear in swap screen.
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
  • scene detection
  • card/drink/score OCR
  • P-point sandwich OCR (outlined white digits, anti-occlusion)
  • battle hand animation stabiliser
        │
        ▼
LocalHybridDecisionAgent      ← rule engine + local ML
  ├── ScheduleAgent            (lesson / rest selection)
  ├── CardRewardAgent          (card pick & discard rules)
  ├── ShopAgent                (upgrade / buy / refresh + visit tracking)
  └── LiveBattlePlanner        ← PPO value network + SO-ISMCTS
            │
            └── HeadlessBattleSimulator  (fast local rollouts)
```

### Key modules

| Path | Description |
|---|---|
| `demo/run_hif_local.py` | Main entry point; supports `on_step_callback` / `on_frame_callback` hooks |
| `demo/agent/local_agent.py` | `LocalHybridDecisionAgent` orchestrator; P-idol stat fallback; shop visit state machine |
| `demo/maa_bridge/runner.py` | Step loop; battle hand animation wait; card-upgrade popup disambiguation |
| `demo/vision/live_adb_recognizer.py` | `LiveScreenRecognizer`; P-point sandwich OCR; shop icon cache |
| `demo/sim/live_battle_planner.py` | `LiveBattlePlanner`; finisher combo gating (好調 ≥ 13); free-action & SB-drink detection |
| `demo/sim/battle_simulator.py` | `HeadlessBattleSimulator`; hand cap enforcement |
| `demo/sim/ismcts.py` | SO-ISMCTS; anti-SKIP pruning; trouble-exile fallback |
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
python demo/app.py
# Open http://localhost:5000
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
1. A non-Legend Active card is already in hand.
2. The active card is playable immediately after the drink effect.
3. Commits to playing that Active card the same turn.

### Shop — spend all P-points
On the final shop visit, refresh the store until all P-points are exhausted. Upgrades prioritised over new cards.

### Card swap — M-card refresh
If an unremovable M-card appears in the early-game card exchange, click refresh to reroll.

### Battle — finisher combo gating
Cards `#162/#163 → #813/#814` combo is only allowed when tracked `goodConditionTurns ≥ 13` (or OCR is occluded in the last 2 turns while the finisher is visually playable).

### Battle — hand animation stabiliser
Before reading cards, the runner waits for hand-area motion < 6.0 mean-pixel-delta (up to 3 probes × 0.85 s) to avoid misreads during draw animations.

---

## Requirements

- Python 3.11
- MuMu Player (Android emulator) with ADB at `127.0.0.1:16384`
- Gakumas installed in emulator
- `gakumas_library.db` (included in repo)

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
python demo/app.py
# Open http://localhost:5000  →  H.I.F Workbench tab
```

---

## Tests

```bash
python -m pytest demo/tests/ -k "not test_live_xai_llm_decision_e2e" -q
```

Expected: **298 passed, 2 skipped** in ~91s.

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
