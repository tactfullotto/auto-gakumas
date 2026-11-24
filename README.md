# auto-gakumas

[中文](README.zh.md) | **English** | [日本語](README.ja.md)

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
        │
        ▼
LocalHybridDecisionAgent      ← rule engine + local ML
  ├── ScheduleAgent            (lesson / rest selection)
  ├── CardRewardAgent          (card pick & discard rules)
  ├── ShopAgent                (upgrade / buy / refresh logic)
  └── LiveBattlePlanner        ← PPO value network + SO-ISMCTS
            │
            └── HeadlessBattleSimulator  (fast local rollouts)
```

### Key modules

| Path | Description |
|---|---|
| `demo/run_hif_local.py` | Main entry point; supports `on_step_callback` / `on_frame_callback` hooks |
| `demo/agent/local_agent.py` | `LocalHybridDecisionAgent` orchestrator |
| `demo/maa_bridge/adb_controller.py` | ADB tap/swipe/screenshot |
| `demo/vision/live_adb_recognizer.py` | `LiveScreenRecognizer`, OCR helpers |
| `demo/sim/` | `HeadlessBattleSimulator`, SO-ISMCTS planner |
| `demo/core/rules.py` | Card/drink selection rules |
| `demo/core/hif_catalog.py` | `ProducePreferences`, HIF memory ability templates, plan metadata |
| `demo/core/tracker.py` | In-run state tracker (deck, hand, P-items, produce step) |
| `demo/web/hif_session.py` | Web workbench session manager (SSE, loadout config, live ADB preview) |
| `demo/web/static/hif_workbench.js` | 3-profession configurator + manual card/memory editor front-end |
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
  - `auto` — fully rely on the ADB scan to detect deck/P-items at run start
  - `manual` — skip scan; inject a hand-crafted deck + P-item list directly into the simulator
  - `hybrid` — ADB scan + manually specified H.I.F memory hooks merged at start
- **Manual card/P-item editor** — build your deck card by card from the full DB
- **H.I.F Memory editor** — 4-slot memory cards, passive bonuses, and wiki preset groups (exile-trouble, extra-use, draw triggers)
- **Produce Preferences panel** — set priority/avoid/keep card lists, banned drink IDs, score delta thresholds
- **Live ADB preview** — real-time screen mirror during a run
- **Library JSON import/export** — save and reload full loadout configs
- **SSE progress stream** — live step-by-step automation log in the browser

---

## Special Rules

### Drink #19 — 初星スペシャル青汁 (banned)
Never select, buy, or use. Auto-discarded via `捨てる` if received. Too random, net-negative in most hands.

### Drink #25 — 特製ハツボシエキス (strict gating)
Only consumed when **all three** conditions hold:
1. A non-Legend Active card is already in hand.
2. The active card is playable immediately after the drink effect.
3. Commits to playing that Active card the same turn.

If a non-L Active card hasn't appeared yet, do **not** drink in advance.

### Shop — spend all P-points
On the final shop visit, refresh the store until all P-points are exhausted. Upgrades are prioritized over new cards.

### Card swap — M-card refresh
During the early-game card exchange, if an unremovable M-card appears, click refresh to reroll the swap pool.

---

## Requirements

- Python 3.11
- MuMu Player (Android emulator) with ADB at `127.0.0.1:16384`
- Gakumas installed in emulator
- `gakumas_library.db` (included in repo)
- Python packages: see `demo/requirements.txt`

```bash
pip install -r demo/requirements.txt
```

Also requires:
- `tesseract` on PATH (for OCR)
- `adb` on PATH or bundled with MuMu

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

Logs are written to `live<N>_run_log.csv`. Debug frames are saved under `debug_frames_live<N>/`.

---

## Tests

```bash
# Run all unit/integration tests (skip live-device tests)
python -m pytest demo/tests/ -k "not test_live_xai_llm_decision_e2e" -q
```

Expected: **176+ passed** in ~260s (offline suite, no live-device tests).

---

## Live Results

| Run | Rounds | Round 2 Score |
|-----|--------|---------------|
| live7 | 2/2 ✅ | +1,594,997 |

---

## Data Sources

Card and drink data extracted from [gakumas-tools/gakumas-tools](https://github.com/gakumas-tools/gakumas-tools) (commit `a82c3a3`, 2026-11-15).  
DSL effect schema from [gakumas-data](https://github.com/pnpm/gakumas-data).

---

## License

Research / personal use. Game assets © Bandai Namco Entertainment.
