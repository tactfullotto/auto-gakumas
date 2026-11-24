# auto-gakumas

[中文](README.zh.md) | [English](README.md) | [**日本語**](README.ja.md)

> **LLM 不使用**のエンドツーエンド自動化ツール。**学園アイドルマスター H.I.F 本戦**カードゲームを、ADB + ローカルビジョン + PPO/ISMCTS プランナーで、プロデュース日程から最終バトルまで完全自動化します。

---

## 概要

`auto-gakumas` は ADB 経由で MuMu エミュレーターを操作し、Gakumas のプロデュースサイクル全体を自動実行します：

1. **スケジュール & レッスン** — 固定週次スケジュールに従い、レッスンと休息を選択。
2. **カード報酬** — カード報酬を評価・選択。禁止/低価値カードを自動破棄。スワップ画面に削除不可の M カードが出た場合はショップをリフレッシュ。
3. **ショップ** — カードのアップグレード、ドリンク購入、最終出発前に P ポイントを使い切る。最後の訪問でリフレッシュして支出を最大化。
4. **H.I.F 本戦（第 1 & 2 戦）** — ローカルの `PPO + SO-ISMCTS` 決定エンジンで両バトルをプレイ。クラウド API 呼び出し不要。

全パイプラインはオフラインで動作します。実行時に LLM やリモート推論は不要です。

---

## アーキテクチャ

```
ADB (MuMu 127.0.0.1:16384)
        │
        ▼
LiveScreenRecognizer          ← OpenCV + Tesseract OCR
  • シーン検出
  • カード/ドリンク/スコア OCR
        │
        ▼
LocalHybridDecisionAgent      ← ルールエンジン + ローカル ML
  ├── ScheduleAgent            （レッスン / 休息の選択）
  ├── CardRewardAgent          （カード選択 & 破棄ルール）
  ├── ShopAgent                （アップグレード / 購入 / リフレッシュロジック）
  └── LiveBattlePlanner        ← PPO 価値ネットワーク + SO-ISMCTS
            │
            └── HeadlessBattleSimulator  （高速ローカルロールアウト）
```

### 主要モジュール

| パス | 説明 |
|---|---|
| `demo/run_hif_local.py` | メインエントリーポイント。`on_step_callback` / `on_frame_callback` フック対応 |
| `demo/agent/local_agent.py` | `LocalHybridDecisionAgent` オーケストレーター |
| `demo/maa_bridge/adb_controller.py` | ADB タップ/スワイプ/スクリーンショット |
| `demo/vision/live_adb_recognizer.py` | `LiveScreenRecognizer`、OCR ヘルパー |
| `demo/sim/` | `HeadlessBattleSimulator`、SO-ISMCTS プランナー |
| `demo/core/rules.py` | カード/ドリンク選択ルール |
| `demo/core/hif_catalog.py` | `ProducePreferences`、HIF メモリアビリティテンプレート、プランメタデータ |
| `demo/core/tracker.py` | 実行中状態トラッカー（デッキ、手札、P アイテム、プロデュースステップ） |
| `demo/web/hif_session.py` | Web ワークベンチセッションマネージャー（SSE、ロードアウト設定、ADB ライブプレビュー） |
| `demo/web/static/hif_workbench.js` | 3 職業コンフィギュレーター + 手動カード/メモリエディターフロントエンド |
| `gakumas_library.db` | SQLite カード & ドリンクデータベース（926 カード、28 ドリンク） |

---

## H.I.F Workbench Web UI

CLI を使わずにプロデュースランの設定・起動ができるローカル Web インターフェースです。

**起動方法：**
```bash
python demo/app.py
# http://localhost:5000 を開く
```

**機能：**

- **3 職業セレクター**（Sense / Logic / Anomaly）P-Idol カタログとステータスプレビュー付き
- **ロードアウトモード：**
  - `auto` — ランスタート時に ADB スキャンでデッキ/P アイテムを自動検出
  - `manual` — スキャンをスキップ；手作りのデッキ + P アイテムリストをシミュレーターに直接注入
  - `hybrid` — ADB スキャン + 手動指定の H.I.F メモリフックをスタート時にマージ
- **手動カード/P アイテムエディター** — フル DB からカードを 1 枚ずつ組み合わせてデッキ構築
- **H.I.F メモリエディター** — 4 スロットメモリカード、パッシブボーナス、Wiki プリセットグループ（トラブル追放、追加使用、ドロートリガー）
- **プロデュース設定パネル** — 優先/回避/保持カードリスト、禁止ドリンク ID、スコア差閾値を設定
- **ADB ライブプレビュー** — ラン中のリアルタイム画面ミラー
- **ライブラリ JSON インポート/エクスポート** — 完全なロードアウト設定の保存と再読み込み
- **SSE 進捗ストリーム** — ブラウザでのステップごとの自動化ログ

---

## バトルルール

### 手札上限（5 枚）
手札に 5 枚ある場合、手札を増やすドリンク（#22 など）の使用を延期。満杯の手札でも使えるカード（トラブルカード除く）がなければターンスキップを許可。

### アンチ空ターン保護
行動ポイントがある場合、絶対にターンをスキップしない（手札満杯または全部トラブルカードの場合を除く）。スタミナ ≥ 6 で使えるカードがない場合、トラブルカードを追放して打開する。

### ドリンク #19 — 初星スペシャル青汁（禁止）
絶対に選択・購入・使用しない。受け取った場合は「捨てる」で自動破棄。ランダム性が高くほとんどの手札で純マイナス。

### ドリンク #25 — 特製ハツボシエキス（厳格管理）
以下の**3 つ**の条件がすべて満たされた場合のみ消費：
1. 手札に非レジェンドアクティブカードがある。
2. ドリンク効果後、そのアクティブカードをすぐにプレイ可能。
3. そのターン内にそのアクティブカードをプレイすることを確約。

非 L アクティブカードがまだ出ていない場合は**事前に飲まない**。

### ショップ — P ポイントを使い切る
最後のショップ訪問では、P ポイントがなくなるまでリフレッシュ。アップグレード優先、次いで新カード購入。

### カードスワップ — M カードリフレッシュ
序盤のカード交換で削除不可の M カードが出た場合、リフレッシュをクリックしてスワッププールを引き直す。

---

## 動作環境

- Python 3.11+
- MuMu Player（Android エミュレーター）、ADB アドレス `127.0.0.1:16384`
- エミュレーターに Gakumas がインストール済み
- `gakumas_library.db`（リポジトリに同梱）
- Python パッケージ：`demo/requirements.txt` 参照

```bash
pip install -r demo/requirements.txt
```

追加要件：
- `tesseract` が PATH に存在（OCR 用）
- `adb` が PATH に存在、または MuMu に同梱

---

## 使い方

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
# http://localhost:5000 を開く → H.I.F Workbench タブ
```

ログは `live<N>_run_log.csv` に書き込まれます。デバッグフレームは `debug_frames_live<N>/` 以下に保存されます。

---

## テスト

```bash
# 全ユニット/統合テストを実行（実機テスト除外）
python -m pytest demo/tests/ -k "not test_live_xai_llm_decision_e2e" -q
```

期待値：**176 以上 通過**、約 260 秒（オフラインスイート）。

---

## 実機テスト結果

| ラン | ラウンド | 第 2 ラウンドスコア |
|-----|--------|-----------------|
| sense139 run3 | 2/2 ✅ | +1,198,099 |

---

## データソース

カード・ドリンクデータは [gakumas-tools/gakumas-tools](https://github.com/gakumas-tools/gakumas-tools)（コミット `a82c3a3`、2026-11-15）から抽出。  
DSL エフェクトスキーマは [gakumas-data](https://github.com/pnpm/gakumas-data) より。

---

## ライセンス

研究/個人利用目的。ゲームアセットは © Bandai Namco Entertainment。
