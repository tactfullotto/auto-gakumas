# auto-gakumas

[**中文**](README.zh.md) | [English](README.md) | [日本語](README.ja.md)

> **零 LLM** 的端到端自动化方案，适用于**学園アイドルマスター（Gakumas）H.I.F 本戦**卡牌对战——从制作日程到最终对战，完全通过 ADB + 本地视觉 + PPO/ISMCTS 规划器自动完成。

---

## 概述

`auto-gakumas` 通过 ADB 驱动 MuMu 模拟器，完成一整个 Gakumas 制作周期：

1. **日程与课程** — 按固定周计划选择课程与休息。
2. **卡牌奖励** — 评估并选取卡牌奖励；自动弃置禁用/低价值卡牌；若换牌界面出现不可移除的 M 卡，则自动刷新商店。
3. **商店** — 升级卡牌、购买饮料，在最终离开前用尽 P 点；最后一次商店访问时刷新以最大化消费。
4. **H.I.F 本戦（第 1 & 2 轮）** — 使用本地 `PPO + SO-ISMCTS` 决策引擎完成两轮对战，无需任何云端 API 调用。

整个流程完全离线运行，运行时无需 LLM 或远程推理。

---

## 架构

```
ADB (MuMu 127.0.0.1:16384)
        │
        ▼
LiveScreenRecognizer          ← OpenCV + Tesseract OCR
  • 场景检测
  • 卡牌/饮料/分数 OCR
        │
        ▼
LocalHybridDecisionAgent      ← 规则引擎 + 本地 ML
  ├── ScheduleAgent            （课程/休息选择）
  ├── CardRewardAgent          （卡牌选取与弃置规则）
  ├── ShopAgent                （升级/购买/刷新逻辑）
  └── LiveBattlePlanner        ← PPO 价值网络 + SO-ISMCTS
            │
            └── HeadlessBattleSimulator  （快速本地模拟）
```

### 核心模块

| 路径 | 描述 |
|---|---|
| `demo/run_hif_local.py` | 主入口；支持 `on_step_callback` / `on_frame_callback` 钩子 |
| `demo/agent/local_agent.py` | `LocalHybridDecisionAgent` 总控 |
| `demo/maa_bridge/adb_controller.py` | ADB 点击/滑动/截图 |
| `demo/vision/live_adb_recognizer.py` | `LiveScreenRecognizer`，OCR 辅助工具 |
| `demo/sim/` | `HeadlessBattleSimulator`，SO-ISMCTS 规划器 |
| `demo/core/rules.py` | 卡牌/饮料选择规则 |
| `demo/core/hif_catalog.py` | `ProducePreferences`，HIF 记忆能力模板，计划元数据 |
| `demo/core/tracker.py` | 运行内状态追踪（牌组、手牌、P 道具、制作步骤） |
| `demo/web/hif_session.py` | Web 工作台会话管理（SSE、配置、ADB 实时预览） |
| `demo/web/static/hif_workbench.js` | 三职业配置器 + 手动卡牌/记忆编辑器前端 |
| `gakumas_library.db` | SQLite 卡牌与饮料数据库（926 张卡，28 种饮料） |

---

## H.I.F Workbench Web UI

用于配置和启动制作运行的本地 Web 界面，无需触碰命令行。

**启动：**
```bash
python demo/app.py
# 打开 http://localhost:5000
```

**功能：**

- **三职业选择器**（Sense / Logic / Anomaly）含 P-Idol 目录和属性预览
- **配置模式：**
  - `auto` — 完全依赖 ADB 扫描在运行开始时检测牌组/P 道具
  - `manual` — 跳过扫描；直接向模拟器注入手工牌组 + P 道具列表
  - `hybrid` — ADB 扫描 + 手动指定的 H.I.F 记忆钩子，在开始时合并
- **手动卡牌/P 道具编辑器** — 从完整数据库逐张构建牌组
- **H.I.F 记忆编辑器** — 4 槽记忆卡、被动加成和维基预设组（流放干扰、额外使用、抽牌触发）
- **制作偏好面板** — 设置优先/回避/保留卡列表、禁用饮料 ID、分数差阈值
- **ADB 实时预览** — 运行期间的实时屏幕镜像
- **库 JSON 导入/导出** — 保存并重载完整配置
- **SSE 进度流** — 浏览器中的逐步自动化日志

---

## 对战规则

### 手牌上限（5 张）
手牌中已有 5 张时，推迟使用会增加手牌数的饮料（如 #22）；若手牌已满仍无法使用（非干扰卡），则允许跳过回合。

### 反空回合保护
绝不空过行动点数（除非手牌已满或全是干扰卡）。若行动点数 ≥ 6 且无可用牌，优先流放干扰卡。

### 饮料 #19 — 初星スペシャル青汁（禁用）
永不选择、购买或使用。若收到则自动通过「捨てる」弃置。过于随机，大多数手牌中净负收益。

### 饮料 #25 — 特製ハツボシエキス（严格管控）
仅当以下**三个**条件同时满足时才消耗：
1. 手牌中已有非传说 Active 卡。
2. 饮料效果生效后该 Active 卡立即可打出。
3. 本回合内承诺打出该 Active 卡。

若非 L Active 卡尚未出现，**不得**提前饮用。

### 商店 — 花光所有 P 点
最后一次商店访问时，刷新商店直至 P 点耗尽。优先升级，其次购买新卡。

### 换牌 — M 卡刷新
早期游戏换牌时，若出现不可移除的 M 卡，点击刷新以重新抽取换牌池。

---

## 环境要求

- Python 3.11+
- MuMu Player（Android 模拟器），ADB 地址 `127.0.0.1:16384`
- 已在模拟器中安装 Gakumas
- `gakumas_library.db`（已包含在仓库中）
- Python 包：见 `demo/requirements.txt`

```bash
pip install -r demo/requirements.txt
```

另需：
- `tesseract` 在 PATH 中（用于 OCR）
- `adb` 在 PATH 中或由 MuMu 捆绑提供

---

## 使用方法

### 命令行（无头模式）
```bash
python -m demo.run_hif_local \
    --adb 127.0.0.1:16384 \
    --idol sense \
    --log-prefix live8
```

### Web UI
```bash
python demo/app.py
# 打开 http://localhost:5000  →  H.I.F Workbench 标签
```

日志写入 `live<N>_run_log.csv`。调试帧保存在 `debug_frames_live<N>/` 下。

---

## 测试

```bash
# 运行所有单元/集成测试（跳过实体设备测试）
python -m pytest demo/tests/ -k "not test_live_xai_llm_decision_e2e" -q
```

预期：**176+ 通过**，约 260 秒（离线套件）。

---

## 实测结果

| 运行 | 轮次 | 第 2 轮分数 |
|-----|--------|-----------------|
| sense139 run3 | 2/2 ✅ | +1,198,099 |

---

## 数据来源

卡牌与饮料数据提取自 [gakumas-tools/gakumas-tools](https://github.com/gakumas-tools/gakumas-tools)（提交 `a82c3a3`，2026-11-15）。  
DSL 效果模式来自 [gakumas-data](https://github.com/pnpm/gakumas-data)。

---

## 许可证

研究/个人用途。游戏资产版权归 Bandai Namco Entertainment 所有。
