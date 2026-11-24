# auto-gakumas

[中文](README.zh.md) | [English](README.md) | [**日本語**](README.ja.md)

> **LLM 不使用**のエンドツーエンド自動化ツール。**学園アイドルマスター H.I.F 本戦**カードゲームを、ADB + ローカルビジョン + PPO/ISMCTS プランナーで、プロデュース日程から最終バトルまで完全自動化します。

---

## 概要

`auto-gakumas` は MuMu エミュレーターを通じて、Gakumas のプロデュースサイクル全体を自動操作します：

1. **スケジュール & レッスン** — 固定週次スケジュールに従い、レッスン・休憩を選択。
2. **カード報酬** — スキルカード報酬を評価・選択。禁止/低価値カードは自動廃棄。交換画面に除外不可の M カードが表示された場合はリフレッシュ。
3. **ショップ** — カード強化・ドリンク購入・P ポイントの完全消費（最終訪問時は P ポイントが尽きるまでリフレッシュ）。
4. **H.I.F 本戦（第 1・第 2 ラウンド）** — ローカルの `PPO + SO-ISMCTS` 決定エンジンで対戦。クラウド API 不要。

すべてオフラインで動作。実行時に LLM は不要です。

---

## アーキテクチャ

```
ADB (MuMu 127.0.0.1:16384)
        │
        ▼
LiveScreenRecognizer          ← OpenCV + Tesseract OCR
  • シーン検出
  • カード/ドリンク/スコア OCR
  • Pポイント「サンドイッチ法」OCR（白縁数字、HUD 遮蔽対策）
  • バトル手札アニメーション安定待機
        │
        ▼
LocalHybridDecisionAgent      ← ルールエンジン + ローカル ML
  ├── ScheduleAgent            （レッスン / 休憩選択）
  ├── CardRewardAgent          （カード選択・廃棄ルール）
  ├── ShopAgent                （強化 / 購入 / リフレッシュ + 訪問カウント）
  └── LiveBattlePlanner        ← PPO価値ネットワーク + SO-ISMCTS
            │
            └── HeadlessBattleSimulator  （高速ローカルロールアウト）
```

### 主要モジュール

| パス | 説明 |
|---|---|
| `demo/run_hif_local.py` | メインエントリ；`on_step_callback` / `on_frame_callback` フック対応 |
| `demo/agent/local_agent.py` | `LocalHybridDecisionAgent` オーケストレーター；P-idol ステータス フォールバック；ショップ訪問ステートマシン |
| `demo/maa_bridge/runner.py` | ステップループ；手札アニメーション待機；カード強化ポップアップ判別 |
| `demo/vision/live_adb_recognizer.py` | `LiveScreenRecognizer`；Pポイント サンドイッチ OCR；ショップアイコンキャッシュ |
| `demo/sim/live_battle_planner.py` | `LiveBattlePlanner`；フィニッシャーコンボ制御（好調 ≥ 13）；フリーアクション / SB ドリンク検出 |
| `demo/sim/battle_simulator.py` | `HeadlessBattleSimulator`；手札 5 枚上限強制 |
| `demo/sim/ismcts.py` | SO-ISMCTS；アンチ SKIP 枝刈り；トラブルカード除外フォールバック |
| `demo/core/idol_guides.py` | P-idol デフォルトステータス参照 & ステータス軸順序 |
| `demo/core/hif_catalog.py` | `ProducePreferences`、HIF メモリアビリティテンプレート、プラン メタデータ |
| `demo/core/tracker.py` | 局内状態トラッカー（デッキ・手札・P アイテム・プロデュースステップ） |
| `demo/web/hif_session.py` | Web ワークベンチ バックエンド（SSE・ローディング設定・ライブ ADB プレビュー） |
| `demo/web/static/hif_workbench.js` | 3 職業コンフィギュレーター + 多次元カード優先度フィルタエディタ |
| `gakumas_library.db` | SQLite データベース（926 枚カード・28 種ドリンク） |

---

## H.I.F Workbench Web UI

CLI を使わずにプロデュースを設定・起動できるローカル Web インターフェース。

**起動：**
```bash
python demo/app.py
# ブラウザで http://localhost:5000 を開く
```

**機能：**

- **3 職業セレクター**（Sense / Logic / Anomaly）P-Idol カタログ & ステータスプレビュー付き
- **ローディングモード：**
  - `auto` — ADB スキャンでデッキ / P アイテムを自動検出
  - `manual` — スキャンをスキップ；手入力デッキ + P アイテムリストを直接注入
  - `hybrid` — ADB スキャン + 手動指定 HIF メモリフックを開幕時にマージ
- **手動カード / P アイテムエディタ** — 全 DB からカードを 1 枚ずつデッキ構築
- **HIF メモリエディタ** — 4 スロットメモリカード・パッシブボーナス・Wiki プリセットグループ
- **プロデュース設定パネル** — 多次元フィルタ（タイプ / レア度 / 強化済み / 効果キーワード / 割当ステータス）；優先 / 避ける / キープカードリスト；禁止ドリンク ID
- **ライブ ADB プレビュー** — 実行中のリアルタイム画面ミラー
- **ライブラリ JSON インポート / エクスポート** — 設定の保存と再読込
- **SSE 進行ストリーム** — ブラウザ内リアルタイムログ

---

## 特別ルール

### ドリンク #19 — 初星スペシャル青汁（禁止）
選択・購入・使用を一切しない。受け取った場合は `捨てる` で自動廃棄。

### ドリンク #25 — 特製ハツボシエキス（厳格制御）
以下の**3 条件すべて**が揃った場合のみ使用：
1. 手札に非 Legend の Active カードがある。
2. ドリンク効果後にその Active カードを即座にプレイできる。
3. 同ターン内にその Active カードをプレイすることを確約。

非 L レベルの Active カードが手札にない場合は事前使用不可。

### ショップ — P ポイント完全消費
最終訪問時は P ポイントが尽きるまでリフレッシュ。強化を購入より優先。

### カード交換 — M カードリフレッシュ
前半のカード交換で除外不可の M カードが出た場合、リフレッシュで引き直す。

### バトル — フィニッシャーコンボ制御
`#162/#163 → #813/#814` コンボは、追跡中の `goodConditionTurns ≥ 13` の場合のみ許可（または残り 2 ターン以内で OCR が遮蔽され #814 が視覚的にプレイ可能な場合も許可）。

### バトル — 手札アニメーション安定待機
カード読み取り前に、手札エリアの平均ピクセル差が 6.0 未満になるまで待機（最大 3 回 × 0.85 秒）。ドロウアニメーション中の誤読を防止。

---

## 環境要件

- Python 3.11
- MuMu Player（Android エミュレーター）ADB アドレス `127.0.0.1:16384`
- エミュレーター内に Gakumas インストール済み
- `gakumas_library.db`（リポジトリに同梱）

```bash
pip install -r demo/requirements.txt
```

`tesseract` と `adb` を PATH に追加する必要があります。

---

## 使用方法

### CLI（ヘッドレス）
```bash
python -m demo.run_hif_local \
    --adb 127.0.0.1:16384 \
    --idol sense \
    --log-prefix live8
```

### Web UI
```bash
python demo/app.py
# http://localhost:5000 → H.I.F Workbench タブを開く
```

---

## テスト

```bash
python -m pytest demo/tests/ -k "not test_live_xai_llm_decision_e2e" -q
```

期待結果：**298 passed, 2 skipped**（約 91 秒）。

---

## 実測結果

| 実行 | ラウンド数 | 第2ラウンド スコア |
|------|-----------|-------------------|
| live7 | 2/2 ✅ | +1,594,997 |

---

## データソース

カード・ドリンクデータは [gakumas-tools/gakumas-tools](https://github.com/gakumas-tools/gakumas-tools)（commit `a82c3a3`、2026-11-15）から抽出。

---

## ライセンス

個人研究・個人利用のみ。ゲームアセットの著作権は © Bandai Namco Entertainment に帰属します。
