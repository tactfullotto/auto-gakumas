# auto-gakumas

[**中文**](README.zh.md) | [English](README.md) | [日本語](README.ja.md)

> **零 LLM** 的端到端自动化方案，适用于**学園アイドルマスター（Gakumas）H.I.F 本戦**卡牌对战——从制作日程到最终对战，完全通过 ADB + 本地视觉 + PPO/ISMCTS 规划器自动完成。

---

## 概述

`auto-gakumas` 通过 MuMu 模拟器完整驱动一次 Gakumas 制作周期：

1. **日程 & 课程** — 按固定每周日程选择课程/休息。
2. **卡牌奖励** — 评估并选取技能卡奖励；自动丢弃被禁/低价值卡；若换牌界面出现不可除外 M 卡则点击刷新；强制换牌时自动保护已定制强化卡。
3. **商店** — 升级卡牌、购买饮料、在最终离开前耗尽 P 点；最后一次访问时刷新商店直至 P 点花尽。
4. **H.I.F 本戦（第 1、2 局）** — 使用本地 `PPO + SO-ISMCTS` 决策引擎，无需任何云端 API 调用。

整套流程离线运行，运行时无需 LLM。

---

## 架构

```
ADB (MuMu 127.0.0.1:16384)
        │
        ▼
LiveScreenRecognizer          ← OpenCV + Tesseract OCR
  • 场景检测 & 橙色 L 角选框识别
  • 卡牌/饮料/分数 OCR & 定制徽章检测
  • P点"三明治法"OCR（白色描边数字，抗HUD遮挡）
  • 斜体纠偏剩余回合环 OCR + 连续 2 帧跳变恢复
  • 战斗手牌动画稳定等待 & 選択中卡死重试
        │
        ▼
LocalHybridDecisionAgent      ← 规则引擎 + 本地 ML
  ├── ScheduleAgent            （课程/休息选择）
  ├── CardRewardAgent          （卡牌选取、换牌保护与丢弃规则）
  ├── ShopAgent                （升级/购买/刷新 + 访问次数追踪）
  └── LiveBattlePlanner        ← PPO价值网络 + SO-ISMCTS + 步进自愈同步
            │
            └── HeadlessBattleSimulator  （快速本地推演 & P道具触发）
```

### 核心模块

| 路径 | 说明 |
|---|---|
| `demo/run_hif_local.py` | 主入口；支持 `on_step_callback` / `on_frame_callback` 回调 |
| `demo/agent/local_agent.py` | `LocalHybridDecisionAgent` 总调度；P-idol 数值 fallback；商店访问状态机；强制换牌定制卡保护 |
| `demo/maa_bridge/runner.py` | 步进循环；战斗手牌动画等待；5 张扇形手牌点击收束与 `選択中` 卡死重试；强化弹窗去歧义 |
| `demo/vision/live_adb_recognizer.py` | `LiveScreenRecognizer`；P点三明治OCR；斜体纠偏回合环OCR（暗化HUD兜底）；商店图标缓存 |
| `demo/sim/live_battle_planner.py` | `LiveBattlePlanner`；finisher连击门控（好調≥13）；清零型终结卡规则；步进反馈自愈与回合同步 |
| `demo/sim/battle_simulator.py` | `HeadlessBattleSimulator`；5张手牌上限；清零型终结卡与禁双倍规则；感性 P 道具（`#27092 届けたい想い` 等） |
| `demo/sim/ismcts.py` | SO-ISMCTS；反 SKIP 剪枝；Trouble 牌除外兜底；清零型终结卡剪枝 |
| `demo/sim/rl_env.py` | `GakumasActionCodec`；合法动作掩码与劣势动作剪枝 |
| `demo/core/idol_guides.py` | P-idol 默认能力值查询与能力轴排序 |
| `demo/core/hif_catalog.py` | `ProducePreferences`、HIF 回忆能力模板、职业元数据 |
| `demo/core/tracker.py` | 局内状态追踪（牌组、手牌、P道具、制作步骤） |
| `demo/web/hif_session.py` | Web 工作台后端 Session 管理（SSE、配置、实时ADB预览） |
| `demo/web/static/hif_workbench.js` | 三职业配置器 + 多维度卡牌偏好过滤编辑器 |
| `gakumas_library.db` | SQLite 数据库（926 张卡牌、28 种饮料） |

---

## H.I.F Workbench Web UI

用于配置和启动制作流程的本地 Web 界面，无需使用命令行。

**启动：**
```bash
python demo/app.py
# 在浏览器打开 http://localhost:5000
```

**功能：**

- **三职业选择器**（Sense / Logic / Anomaly），含 P-Idol 目录与能力值预览
- **配置模式：**
  - `auto` — 完全依赖 ADB 扫描识别开局卡组/P道具
  - `manual` — 跳过扫描，直接注入手填卡组 + P道具列表
  - `hybrid` — ADB 扫描 + 手动指定 HIF 回忆钩子合并
- **手动卡组/P道具编辑器** — 从完整数据库逐张构建卡组
- **HIF 回忆编辑器** — 4格回忆卡、被动词条、Wiki 预设组（除外眠気/追加使用数等）
- **制作偏好面板** — 多维过滤器（类型/稀有度/强化/效果关键词/分配状态）；设置优先/避免/保留卡列表；禁用饮料ID
- **实时 ADB 预览** — 运行中实时屏幕镜像
- **库 JSON 导入/导出** — 保存并重载完整配置
- **SSE 进度流** — 浏览器内逐步自动化日志实时推送

---

## 特殊规则

### 饮料 #19 — 初星スペシャル青汁（已禁用）
永不选取、购买或使用。若收到则通过 `捨てる` 自动丢弃。

### 饮料 #25 — 特製ハツボシエキス（严格门控）
仅当以下**三条件同时满足**时才饮用：
1. 手中已有一张非 Legend 的 Active 卡（排除清零型终结卡）。
2. 饮用后该 Active 卡立即可打出。
3. 本回合随即打出该 Active 卡。

若非 L 级 Active 卡尚未入手，不可提前饮用。

### 商店 — 花光 P 点
最后一次访问商店时，持续刷新直至 P 点耗尽。优先升级卡牌，其次购买新卡。

### 换牌 — M 卡刷新与定制卡保护
- 前期换牌若出现不可除外的 M 卡，点击刷新重滚换牌池。
- 强制 H.I.F `チェンジ` 界面下，自动保护已定制强化卡（`カスタマイズ済`），直接切换至未定制槽位，避免取消返回第 1 页死循环。

### 战斗 — 终结卡连击与资源清零型终结卡规则
- **成长型终结连击（`#162/#163 → #813/#814`）**：仅在追踪到 `goodConditionTurns ≥ 13` 时触发（或 OCR 被遮挡且处于最后 2 回合时允许）。
- **资源清零型单发终结卡（`#618/#619 世界一の勇気`、`#152/#153 届いて`）**：严格留至最后 1 回合（`turns_remaining == 1`）；禁止消耗已就绪的 `2x`（`#162/#163` 或 `#25`）以免第二次触发打在已清零数值上；仅在最后 1 回合且缺手时检索。

### 战斗 — 出牌可靠性与模拟器自愈同步
- **手牌动画稳定**：读牌前等待手牌区域帧差均值低于 6.0（最多 3 次探测 × 0.85 秒），避免抽牌动画中误识别。
- **5 张扇形手牌收束点击与重试**：外侧槽位点击自动向内收束（`13.5%–86.5%` 宽度），二次确认点击抬起后位置；若卡牌卡在 `選択中` 则自动补点重试，未确认打出时不误记消耗。
- **步进反馈自愈**：若 `limit=1` 卡在设备上未真正触发而仍留在手中，自动回滚模拟器幻觉状态（`double_card_effect_cards`、`cards_used`、误进回合），并基于手牌变化对齐真实剩余回合数。

---

## 环境要求

- Python 3.11
- MuMu 模拟器，ADB 地址 `127.0.0.1:16384`
- 模拟器中已安装 Gakumas
- `gakumas_library.db`（已包含在仓库中）

```bash
pip install -r demo/requirements.txt
```

同时需要将 `tesseract` 和 `adb` 加入 PATH。

---

## 使用方法

### 命令行（无界面）
```bash
python -m demo.run_hif_local \
    --adb 127.0.0.1:16384 \
    --idol sense \
    --log-prefix live8
```

### Web UI
```bash
python demo/app.py
# 打开 http://localhost:5000 → H.I.F Workbench 标签
```

---

## 测试

```bash
python -m pytest demo/tests/ -k "not test_live_xai_llm_decision_e2e" -q
```

共 **342 项**离线单元与集成测试（排除真机 E2E）。

---

## 实测结果

| 运行 | 局数 | 第2局得分 |
|------|------|-----------|
| live7 | 2/2 ✅ | +1,594,997 |

---

## 数据来源

卡牌与饮料数据来自 [gakumas-tools/gakumas-tools](https://github.com/gakumas-tools/gakumas-tools)（commit `a82c3a3`，2026-11-15）。

---

## 许可

仅供个人研究使用。游戏资产版权归 © Bandai Namco Entertainment 所有。
