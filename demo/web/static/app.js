const THEME_STORAGE_KEY = "gakumas-color-scheme";
const SVG_NS = "http://www.w3.org/2000/svg";

const SCENE_LABELS = {
  produce_schedule: "培育日程",
  card_reward: "三选一奖励",
  battle_turn: "战斗打牌",
  shop: "商店购买",
  popup: "确认弹窗",
  skip: "过场跳过",
  idle: "等待中",
};

const PIPELINE_ITEMS = [
  { id: "produce_schedule", label: "培育日程选择" },
  { id: "card_reward", label: "三选一选卡" },
  { id: "battle_turn", label: "战斗智能出牌" },
  { id: "shop", label: "商店与饮料筛选" },
  { id: "popup", label: "弹窗自动确认" },
];

const state = {
  full: null,
  activeView: "tasks",
  lastFrameKey: "",
  showOverlay: true,
  refreshQueued: false,
  libraryCache: new Map(),
};

const els = {
  tablist: document.getElementById("instance-tablist"),
  btnAddInstance: document.getElementById("btn-add-instance"),
  headerModePill: document.getElementById("header-mode-pill"),
  headerLlmPill: document.getElementById("header-llm-pill"),
  btnThemeToggle: document.getElementById("btn-theme-toggle"),
  navButtons: Array.from(document.querySelectorAll(".nav-rail .nav-item")),
  views: {
    tasks: document.getElementById("view-tasks"),
    monitor: document.getElementById("view-monitor"),
    library: document.getElementById("view-library"),
    settings: document.getElementById("view-settings"),
  },
  cfgPlan: document.getElementById("cfg-plan"),
  cfgMode: document.getElementById("cfg-mode"),
  cfgTarget: document.getElementById("cfg-target"),
  cfgScenarioId: document.getElementById("cfg-scenario-id"),
  cfgWeek: document.getElementById("cfg-week"),
  btnJumpWeek: document.getElementById("btn-jump-week"),
  roadmapTitleSpan: document.getElementById("roadmap-title-span"),
  scenarioRoadmapList: document.getElementById("scenario-roadmap-list"),
  cfgPreset: document.getElementById("cfg-preset"),
  cfgLlmMode: document.getElementById("cfg-llm-mode"),
  cfgSteps: document.getElementById("cfg-steps"),
  cfgDelay: document.getElementById("cfg-delay"),
  chkOverlay: document.getElementById("chk-overlay"),
  chkAutoProgress: document.getElementById("chk-auto-progress"),
  chkFallbackMock: document.getElementById("chk-fallback-mock"),
  simStamina: document.getElementById("sim-stamina"),
  simGenki: document.getElementById("sim-genki"),
  simScore: document.getElementById("sim-score"),
  btnRefreshPreview: document.getElementById("btn-refresh-preview"),
  btnTestConn: document.getElementById("btn-test-conn"),
  btnApplyScenario: document.getElementById("btn-apply-scenario"),
  btnInjectBattle: document.getElementById("btn-inject-battle"),
  badgeScenario: document.getElementById("badge-scenario"),
  promptTokenLabel: document.getElementById("prompt-token-label"),
  kpiWeek: document.getElementById("kpi-week"),
  kpiScene: document.getElementById("kpi-scene"),
  kpiTurn: document.getElementById("kpi-turn"),
  kpiStamina: document.getElementById("kpi-stamina"),
  kpiGenki: document.getElementById("kpi-genki"),
  kpiScore: document.getElementById("kpi-score"),
  kpiLimits: document.getElementById("kpi-limits"),
  obsBuffsRow: document.getElementById("obs-buffs-row"),
  promptCode: document.getElementById("prompt-code"),
  runnerStatusPill: document.getElementById("runner-status-pill"),
  btnStep: document.getElementById("btn-step"),
  btnReset: document.getElementById("btn-reset"),
  btnToggleRun: document.getElementById("btn-toggle-run"),
  pipelineList: document.getElementById("pipeline-list"),
  decisionSourceTag: document.getElementById("decision-source-tag"),
  decisionMetrics: document.getElementById("decision-metrics"),
  decisionReason: document.getElementById("decision-reason"),
  actionCount: document.getElementById("action-count"),
  legalActionsList: document.getElementById("legal-actions-list"),
  liveDot: document.getElementById("live-dot"),
  liveBadge: document.getElementById("live-badge"),
  liveResolution: document.getElementById("live-resolution"),
  liveCanvasWrap: document.getElementById("live-canvas-wrap"),
  liveFrameImg: document.getElementById("live-frame-img"),
  liveOverlaySvg: document.getElementById("live-overlay-svg"),
  btnZoomFrame: document.getElementById("btn-zoom-frame"),
  logsContainer: document.getElementById("logs-container"),
  btnClearLogs: document.getElementById("btn-clear-logs"),
  monitorSummaryBadge: document.getElementById("monitor-summary-badge"),
  monitorZoom: document.getElementById("monitor-zoom"),
  btnMonitorRefresh: document.getElementById("btn-monitor-refresh"),
  monitorGrid: document.getElementById("monitor-grid"),
  libQuery: document.getElementById("lib-query"),
  libKind: document.getElementById("lib-kind"),
  libPlan: document.getElementById("lib-plan"),
  libCategoryWrap: document.getElementById("lib-category-wrap"),
  libCategory: document.getElementById("lib-category"),
  btnLibSearch: document.getElementById("btn-lib-search"),
  libCountLabel: document.getElementById("lib-count-label"),
  libraryGrid: document.getElementById("library-grid"),
  setAdbHost: document.getElementById("set-adb-host"),
  setAdbPort: document.getElementById("set-adb-port"),
  setAdbRoot: document.getElementById("set-adb-root"),
  setWin32Title: document.getElementById("set-win32-title"),
  btnSaveConn: document.getElementById("btn-save-conn"),
  btnDiagAdb: document.getElementById("btn-diag-adb"),
  diagAdbResult: document.getElementById("diag-adb-result"),
  setLlmMode: document.getElementById("set-llm-mode"),
  setLlmBase: document.getElementById("set-llm-base"),
  setLlmModel: document.getElementById("set-llm-model"),
  setLlmKey: document.getElementById("set-llm-key"),
  btnSaveLlm: document.getElementById("btn-save-llm"),
  btnDiagLlm: document.getElementById("btn-diag-llm"),
  diagLlmResult: document.getElementById("diag-llm-result"),
  systemThemeStatus: document.getElementById("system-theme-status"),
  themeRadios: Array.from(document.querySelectorAll('input[name="theme-pref"]')),
  cardDetailDialog: document.getElementById("card-detail-dialog"),
  cardDialogTitle: document.getElementById("card-dialog-title"),
  dialogCardImg: document.getElementById("dialog-card-img"),
  dialogCardTags: document.getElementById("dialog-card-tags"),
  dialogCardDesc: document.getElementById("dialog-card-desc"),
  dialogCardDsl: document.getElementById("dialog-card-dsl"),
  frameZoomDialog: document.getElementById("frame-zoom-dialog"),
  zoomCanvasWrap: document.getElementById("zoom-canvas-wrap"),
  zoomFrameImg: document.getElementById("zoom-frame-img"),
  zoomOverlaySvg: document.getElementById("zoom-overlay-svg"),
  toastRegion: document.getElementById("toast-region"),
};

function showToast(msg) {
  const item = document.createElement("div");
  item.className = "toast-item";
  item.textContent = msg;
  els.toastRegion.appendChild(item);
  window.setTimeout(() => item.remove(), 3400);
}

async function apiGet(url) {
  const resp = await fetch(url, { cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}

async function apiPost(url, payload = {}) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok || data.ok === false) {
    throw new Error(data.error || data.message || `HTTP ${resp.status}`);
  }
  return data;
}

/* ==================== Color Scheme Manager (per dark-mode-toggle) ==================== */
const darkMedia = window.matchMedia("(prefers-color-scheme: dark)");

function getSystemScheme() {
  return darkMedia.matches ? "dark" : "light";
}

function getStoredScheme() {
  try {
    const val = localStorage.getItem(THEME_STORAGE_KEY);
    return val === "light" || val === "dark" ? val : null;
  } catch (_) {
    return null;
  }
}

function applySchemePref(pref) {
  const meta = document.querySelector('meta[name="color-scheme"]');
  try {
    if (pref === "light" || pref === "dark") {
      localStorage.setItem(THEME_STORAGE_KEY, pref);
    } else {
      localStorage.removeItem(THEME_STORAGE_KEY);
    }
  } catch (_) {}
  if (meta) {
    meta.content = pref === "light" || pref === "dark" ? pref : "light dark";
  }
  syncThemeControls();
}

function syncThemeControls() {
  const stored = getStoredScheme();
  const sys = getSystemScheme();
  els.systemThemeStatus.textContent = sys === "dark" ? "当前深色" : "当前浅色";
  const radioTarget = stored || "system";
  for (const r of els.themeRadios) {
    r.checked = r.value === radioTarget;
  }
}

function toggleComfortTheme() {
  const sys = getSystemScheme();
  const rendered = getStoredScheme() ?? sys;
  const target = rendered === "dark" ? "light" : "dark";
  if (target === sys) {
    applySchemePref("system");
  } else {
    applySchemePref(target);
  }
}

darkMedia.addEventListener("change", syncThemeControls);
window.addEventListener("storage", (e) => {
  if (e.key === THEME_STORAGE_KEY) {
    const val = getStoredScheme();
    const meta = document.querySelector('meta[name="color-scheme"]');
    if (meta) meta.content = val ?? "light dark";
    syncThemeControls();
  }
});

/* ==================== Native Dialog Helpers ==================== */
function setupDialogs() {
  for (const dialog of [els.cardDetailDialog, els.frameZoomDialog]) {
    dialog.addEventListener("close", () => {
      dialog.returnValue = "";
    });
  }
  if (!("closedBy" in HTMLDialogElement.prototype)) {
    const outside = (e, dialog) => {
      const r = dialog.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };
    for (const dialog of document.querySelectorAll('dialog[closedby="any"]')) {
      let pressedOutside = false;
      dialog.addEventListener("pointerdown", (e) => {
        pressedOutside = outside(e, dialog);
      });
      dialog.addEventListener("pointerup", (e) => {
        if (pressedOutside && outside(e, dialog)) {
          dialog.close();
        }
      });
    }
  }
}

function openCardDetailModal(item) {
  els.cardDialogTitle.textContent = item.name ? `${item.name} (#${item.id || item.card_id || item.drink_id || ""})` : "详情";
  const imgUrl = item.details_url || item.icon_url || "";
  if (imgUrl) {
    els.dialogCardImg.src = imgUrl;
    els.dialogCardImg.alt = item.name || "卡牌详情图片";
  } else {
    els.dialogCardImg.removeAttribute("src");
  }

  els.dialogCardTags.replaceChildren();
  const tags = [
    item.plan ? `职业: ${item.plan}` : null,
    item.rarity ? `稀有度: ${item.rarity}` : null,
    item.category ? `分类: ${item.category}` : null,
    item.type ? `类型: ${item.type}` : null,
    item.upgraded ? "+ 已强化" : null,
    item.limit_count ? `限 ${item.limit_count} 次` : null,
  ].filter(Boolean);

  for (const t of tags) {
    const span = document.createElement("span");
    span.className = "badge-mini";
    span.textContent = t;
    els.dialogCardTags.appendChild(span);
  }

  els.dialogCardDesc.textContent = item.description || item.dsl_summary || "无效果文本";
  const dslPayload = {
    id: item.id ?? item.card_id ?? item.drink_id,
    dsl_cost: item.dsl_cost ?? null,
    dsl_conditions: item.dsl_conditions ?? null,
    dsl_actions: item.dsl_actions ?? null,
  };
  els.dialogCardDsl.textContent = JSON.stringify(dslPayload, null, 2);
  els.cardDetailDialog.showModal();
}

/* ==================== Navigation & View Switching ==================== */
function switchView(viewName) {
  state.activeView = viewName;
  for (const btn of els.navButtons) {
    const isCurrent = btn.dataset.view === viewName;
    if (isCurrent) {
      btn.setAttribute("aria-current", "page");
    } else {
      btn.removeAttribute("aria-current");
    }
  }
  for (const [k, section] of Object.entries(els.views)) {
    section.hidden = k !== viewName;
  }
  if (viewName === "library" && els.libraryGrid.childElementCount === 0) {
    loadLibrary();
  }
  if (viewName === "monitor" && state.full) {
    renderMonitorGrid(state.full);
  }
}

/* ==================== SVG Overlay Rendering ==================== */
function makeSvgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, String(v));
  }
  return el;
}

function renderOverlayToSvg(svgEl, active, clickCoords) {
  svgEl.replaceChildren();
  const meta = active.frame_meta || { width: 1080, height: 1920 };
  const w = meta.width || 1080;
  const h = meta.height || 1920;
  svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);

  if (!state.showOverlay) {
    return;
  }

  const obs = active.observation;
  const dec = active.decision;
  const chosenBbox = (() => {
    if (!dec) return null;
    const match = (active.legal_actions || []).find((a) => a.choice_id === dec.choice_id);
    return match ? match.bbox : null;
  })();

  const drawBox = (bbox, color, label, isChosen = false) => {
    if (!bbox) return;
    const bw = Math.max(4, bbox.xmax - bbox.xmin);
    const bh = Math.max(4, bbox.ymax - bbox.ymin);
    const rect = makeSvgEl("rect", {
      x: bbox.xmin,
      y: bbox.ymin,
      width: bw,
      height: bh,
      rx: 10,
      fill: isChosen ? "rgba(239, 68, 68, 0.14)" : "none",
      stroke: isChosen ? "#ef4444" : color,
      "stroke-width": isChosen ? 8 : 5,
    });
    svgEl.appendChild(rect);

    if (label) {
      const bg = makeSvgEl("rect", {
        x: bbox.xmin,
        y: Math.max(2, bbox.ymin - 32),
        width: Math.min(bw + 40, Math.max(110, label.length * 16)),
        height: 30,
        rx: 6,
        fill: isChosen ? "#ef4444" : color,
      });
      const text = makeSvgEl("text", {
        x: bbox.xmin + 8,
        y: Math.max(22, bbox.ymin - 10),
        fill: "#ffffff",
        "font-size": 20,
        "font-weight": "bold",
      });
      text.textContent = label;
      svgEl.appendChild(bg);
      svgEl.appendChild(text);
    }
  };

  if (obs) {
    (obs.cards || []).forEach((c, idx) => {
      const isChosen = Boolean(
        chosenBbox && c.bbox && chosenBbox.xmin === c.bbox.xmin && chosenBbox.ymin === c.bbox.ymin
      );
      drawBox(c.bbox, "#3b82f6", `#${idx} ${c.name || c.card_id}`, isChosen);
    });
    (obs.drinks || []).forEach((d, idx) => {
      const isChosen = Boolean(
        chosenBbox && d.bbox && chosenBbox.xmin === d.bbox.xmin && chosenBbox.ymin === d.bbox.ymin
      );
      drawBox(d.bbox, "#10b981", `D${idx} ${d.name || d.drink_id}`, isChosen);
    });
    (obs.options || []).forEach((opt, idx) => {
      const isChosen = Boolean(
        chosenBbox && opt.bbox && chosenBbox.xmin === opt.bbox.xmin && chosenBbox.ymin === opt.bbox.ymin
      );
      drawBox(opt.bbox, "#f59e0b", `[${idx}] ${opt.name || "Option"}`, isChosen);
    });
  }

  if (chosenBbox && (!obs || ((obs.cards || []).length === 0 && (obs.options || []).length === 0))) {
    drawBox(chosenBbox, "#ef4444", `[${dec.choice_id}]`, true);
  }

  if (Array.isArray(clickCoords) && clickCoords.length === 2) {
    const [cx, cy] = clickCoords;
    svgEl.appendChild(
      makeSvgEl("circle", {
        cx,
        cy,
        r: 32,
        fill: "rgba(239, 68, 68, 0.22)",
        stroke: "#ef4444",
        "stroke-width": 5,
      })
    );
    svgEl.appendChild(
      makeSvgEl("line", {
        x1: cx - 44,
        y1: cy,
        x2: cx + 44,
        y2: cy,
        stroke: "#ffffff",
        "stroke-width": 4,
      })
    );
    svgEl.appendChild(
      makeSvgEl("line", {
        x1: cx,
        y1: cy - 44,
        x2: cx,
        y2: cy + 44,
        stroke: "#ffffff",
        "stroke-width": 4,
      })
    );
  }
}

/* ==================== Main State Renderers ==================== */
function renderTabs(full) {
  els.tablist.replaceChildren();
  const instances = full.instances || [];
  instances.forEach((inst, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.role = "tab";
    btn.className = "instance-tab";
    const selected = inst.id === full.active_id;
    btn.setAttribute("aria-selected", selected ? "true" : "false");
    btn.tabIndex = selected ? 0 : -1;
    btn.dataset.instanceId = inst.id;

    const dot = document.createElement("span");
    dot.className = inst.running ? "tab-dot tab-dot--running" : "tab-dot";
    const text = document.createElement("span");
    text.textContent = inst.name;
    btn.append(dot, text);

    btn.addEventListener("click", async () => {
      if (inst.id !== full.active_id) {
        await apiPost("/api/instance/select", { instance_id: inst.id });
        await refreshState();
      }
    });

    btn.addEventListener("keydown", async (e) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      let nextIdx = idx;
      if (e.key === "ArrowRight") nextIdx = (idx + 1) % instances.length;
      if (e.key === "ArrowLeft") nextIdx = (idx - 1 + instances.length) % instances.length;
      if (e.key === "Home") nextIdx = 0;
      if (e.key === "End") nextIdx = instances.length - 1;
      const target = instances[nextIdx];
      await apiPost("/api/instance/select", { instance_id: target.id });
      await refreshState();
      const targetEl = els.tablist.querySelector(`[data-instance-id="${target.id}"]`);
      if (targetEl) targetEl.focus();
    });

    els.tablist.appendChild(btn);
  });
}

function renderPipeline(active) {
  els.pipelineList.replaceChildren();
  const currentScene = active.observation ? active.observation.scene_type : "idle";
  const enabledMap = active.config.tasks_enabled || {};
  for (const item of PIPELINE_ITEMS) {
    const li = document.createElement("li");
    const label = document.createElement("label");
    label.className = "pipeline-step";
    if (currentScene === item.id) {
      label.classList.add("is-active-scene");
    }
    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = enabledMap[item.id] !== false;
    chk.addEventListener("change", () => {
      apiPost("/api/config", {
        instance_id: active.id,
        tasks_enabled: { [item.id]: chk.checked },
      }).catch((err) => showToast(`保存失败: ${err.message}`));
    });
    const span = document.createElement("span");
    span.textContent = item.label;
    label.append(chk, span);
    li.appendChild(label);
    els.pipelineList.appendChild(li);
  }
}

function renderLegalActions(active) {
  const actions = active.legal_actions || [];
  const chosenId = active.decision ? active.decision.choice_id : null;
  els.actionCount.textContent = String(actions.length);
  els.legalActionsList.replaceChildren();

  if (actions.length === 0) {
    const empty = document.createElement("li");
    empty.className = "muted-note";
    empty.textContent = "暂无合法动作（请点击抓取预览或单步执行）";
    els.legalActionsList.appendChild(empty);
    return;
  }

  for (const act of actions) {
    const li = document.createElement("li");
    li.className = "action-item";
    if (chosenId !== null && act.choice_id === chosenId) {
      li.classList.add("is-chosen");
    }

    const thumbWrap = document.createElement("div");
    thumbWrap.className = "action-thumb-wrap";
    if (act.icon_url) {
      const img = document.createElement("img");
      img.src = act.icon_url;
      img.alt = act.description;
      img.width = 48;
      img.height = 48;
      thumbWrap.appendChild(img);
    } else {
      const idText = document.createElement("span");
      idText.className = "action-fallback-id";
      idText.textContent = `#${act.choice_id}`;
      thumbWrap.appendChild(idText);
    }

    const mainCol = document.createElement("div");
    mainCol.className = "action-main";
    const titleLine = document.createElement("div");
    titleLine.className = "action-title-line";
    const idxTag = document.createElement("span");
    idxTag.className = "badge-mini";
    idxTag.textContent = `[${act.choice_id}] ${act.action_type}`;
    titleLine.appendChild(idxTag);
    if (chosenId !== null && act.choice_id === chosenId) {
      const pickTag = document.createElement("span");
      pickTag.className = "badge-mini";
      pickTag.textContent = "已选择";
      titleLine.appendChild(pickTag);
    }

    const descEl = document.createElement("div");
    descEl.className = "action-desc-text";
    descEl.textContent = act.description;
    mainCol.append(titleLine, descEl);

    const rightCol = document.createElement("div");
    const payload = act.payload || {};
    if (payload.card_id || payload.drink_id) {
      const infoBtn = document.createElement("button");
      infoBtn.type = "button";
      infoBtn.className = "btn-secondary btn-compact";
      infoBtn.textContent = "详情";
      infoBtn.addEventListener("click", async () => {
        const obs = active.observation || {};
        const matched =
          (obs.cards || []).find((c) => c.card_id === payload.card_id) ||
          (obs.drinks || []).find((d) => d.drink_id === payload.drink_id) ||
          (obs.options || []).find((o) => o.card_id === payload.card_id || o.drink_id === payload.drink_id);
        if (matched) {
          openCardDetailModal(matched);
          return;
        }
        openCardDetailModal({
          name: act.description,
          id: payload.card_id || payload.drink_id,
          icon_url: act.icon_url,
          details_url: act.details_url,
        });
      });
      rightCol.appendChild(infoBtn);
    }

    li.append(thumbWrap, mainCol, rightCol);
    els.legalActionsList.appendChild(li);
  }
}

function renderLogs(active) {
  els.logsContainer.replaceChildren();
  const logs = active.logs || [];
  if (logs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "muted-note";
    empty.textContent = "日志空闲中...";
    els.logsContainer.appendChild(empty);
    return;
  }
  for (const log of logs.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "log-line";
    const t = document.createElement("span");
    t.className = "log-time";
    t.textContent = `[${log.time}]`;
    const tag = document.createElement("span");
    tag.className = `log-tag log-tag--${log.level}`;
    tag.textContent = `[${String(log.level || "info").toUpperCase()}]`;
    const msg = document.createElement("span");
    msg.textContent = log.message;
    row.append(t, tag, msg);
    els.logsContainer.appendChild(row);
  }
}

function renderMonitorGrid(full) {
  els.monitorGrid.replaceChildren();
  const monitors = full.monitors || [];
  const runningCount = monitors.filter((m) => m.running).length;
  if (els.monitorSummaryBadge) {
    els.monitorSummaryBadge.textContent = `${monitors.length} 实例 · ${runningCount} 运行中`;
  }
  for (const m of monitors) {
    const card = document.createElement("article");
    card.className = "glass-card monitor-card";

    const head = document.createElement("div");
    head.className = "panel-header";
    const title = document.createElement("strong");
    title.textContent = m.name;
    const badge = document.createElement("span");
    badge.className = "panel-badge";
    badge.textContent = m.running ? "RUNNING" : `${m.mode.toUpperCase()} · ${m.plan}`;
    head.append(title, badge);

    const img = document.createElement("img");
    img.className = "monitor-thumb";
    img.src = `/api/frame?instance_id=${encodeURIComponent(m.id)}&seq=${m.frame_seq}`;
    img.alt = `${m.name} 实时监视截图`;
    img.width = m.frame_meta.width || 1080;
    img.height = m.frame_meta.height || 1920;

    const metaLine = document.createElement("div");
    metaLine.className = "muted-note";
    metaLine.textContent = `场景: ${SCENE_LABELS[m.scene_type] || m.scene_type} | 回合 ${m.turn} | 体力 ${m.stamina}/${m.max_stamina} | 分数 ${m.score}`;

    const reasonLine = document.createElement("div");
    reasonLine.className = "decision-reason";
    reasonLine.textContent = m.last_choice !== null ? `[${m.last_choice}] (${m.last_source}) ${m.last_reason}` : m.last_reason;

    const switchBtn = document.createElement("button");
    switchBtn.type = "button";
    switchBtn.className = "btn-secondary btn-compact";
    switchBtn.textContent = m.id === full.active_id ? "当前正在编辑" : "切换到此实例";
    switchBtn.addEventListener("click", async () => {
      await apiPost("/api/instance/select", { instance_id: m.id });
      switchView("tasks");
      await refreshState();
    });

    card.append(head, img, metaLine, reasonLine, switchBtn);
    els.monitorGrid.appendChild(card);
  }
}

function renderScenarioRoadmap(active) {
  const roadmap = active.scenario_roadmap;
  if (!roadmap || !Array.isArray(roadmap.weeks)) return;
  const obs = active.observation;
  const currentW = (obs && obs.week) || active.config.current_week || 1;

  if (els.roadmapTitleSpan) {
    els.roadmapTitleSpan.textContent = `${roadmap.short_name} · 当前第${currentW}周`;
  }

  if (els.cfgWeek && document.activeElement !== els.cfgWeek) {
    els.cfgWeek.replaceChildren();
    for (const w of roadmap.weeks) {
      const opt = document.createElement("option");
      opt.value = String(w.week);
      opt.textContent = w.title;
      els.cfgWeek.appendChild(opt);
    }
    els.cfgWeek.value = String(currentW);
  }

  if (els.scenarioRoadmapList) {
    els.scenarioRoadmapList.replaceChildren();
    for (const w of roadmap.weeks) {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "roadmap-item-btn";
      if (w.week === currentW) {
        btn.classList.add("is-current-week");
      }
      const leftSpan = document.createElement("span");
      leftSpan.textContent = w.title;
      const rightSpan = document.createElement("span");
      rightSpan.className = "badge-mini";
      rightSpan.textContent = SCENE_LABELS[w.scene_type] || w.scene_type;
      btn.append(leftSpan, rightSpan);
      btn.addEventListener("click", async () => {
        await apiPost("/api/scenario", {
          scenario_id: roadmap.scenario_id,
          scenario_preset: "hajime_regular",
          week: w.week,
          plan: els.cfgPlan.value,
        });
        await refreshState();
        showToast(`已切换至: ${w.title}`);
      });
      li.appendChild(btn);
      els.scenarioRoadmapList.appendChild(li);
    }
  }
}

function renderActiveInstance(full) {
  const active = full.active;
  const cfg = active.config;

  // Header status pills
  els.headerModePill.textContent = `${cfg.mode.toUpperCase()} · ${cfg.plan.toUpperCase()}`;
  els.headerLlmPill.textContent = cfg.has_api_key ? `LLM: ${cfg.llm_model}` : "LLM: 本地规则回退";

  // Top & Left config inputs (only overwrite when not actively focused)
  if (document.activeElement !== els.cfgPlan) els.cfgPlan.value = cfg.plan;
  if (document.activeElement !== els.cfgMode) els.cfgMode.value = cfg.mode;
  if (document.activeElement !== els.cfgTarget) {
    els.cfgTarget.value = cfg.mode === "win32" ? cfg.win32_title : `${cfg.adb_host}:${cfg.adb_port}`;
  }
  if (els.cfgScenarioId && document.activeElement !== els.cfgScenarioId) {
    els.cfgScenarioId.value = cfg.scenario_id || "hajime_regular";
  }
  if (document.activeElement !== els.cfgPreset) els.cfgPreset.value = cfg.scenario_preset;
  if (document.activeElement !== els.cfgLlmMode) els.cfgLlmMode.value = cfg.llm_mode;
  if (document.activeElement !== els.cfgSteps) els.cfgSteps.value = String(cfg.steps);
  if (document.activeElement !== els.cfgDelay) els.cfgDelay.value = String(cfg.delay);
  els.chkAutoProgress.checked = Boolean(cfg.auto_progress);
  els.chkFallbackMock.checked = Boolean(cfg.fallback_to_mock);
  els.badgeScenario.textContent = `${(cfg.scenario_id || "hajime").toUpperCase()} · W${cfg.current_week || 1}`;

  renderScenarioRoadmap(active);

  // Settings view fields
  if (document.activeElement !== els.setAdbHost) els.setAdbHost.value = cfg.adb_host;
  if (document.activeElement !== els.setAdbPort) els.setAdbPort.value = String(cfg.adb_port);
  if (document.activeElement !== els.setAdbRoot) els.setAdbRoot.value = cfg.adb_root_pwd;
  if (document.activeElement !== els.setWin32Title) els.setWin32Title.value = cfg.win32_title;
  if (document.activeElement !== els.setLlmMode) els.setLlmMode.value = cfg.llm_mode;
  if (document.activeElement !== els.setLlmBase) els.setLlmBase.value = cfg.llm_base_url;
  if (document.activeElement !== els.setLlmModel) els.setLlmModel.value = cfg.llm_model;

  // Runner state & Start/Stop shimmer button
  els.runnerStatusPill.textContent = active.running ? `RUNNING · STEP ${active.step_count}` : `IDLE · STEP ${active.step_count}`;
  const runLabel = els.btnToggleRun.querySelector(".btn-run-label");
  if (active.running) {
    els.btnToggleRun.classList.add("is-running");
    if (runLabel) runLabel.textContent = "⏹ 停止任务";
    els.btnStep.disabled = true;
  } else {
    els.btnToggleRun.classList.remove("is-running");
    if (runLabel) runLabel.textContent = "▶ 开始任务";
    els.btnStep.disabled = false;
  }

  // Prompt & Token Count (uncapped full context)
  const tok = active.est_tokens || 0;
  els.promptTokenLabel.textContent = `${tok} tok · 完整上下文`;
  els.promptTokenLabel.className = "token-pill token-pill--ok";
  els.promptCode.textContent = active.prompt_text || "等待抓取观测画面...";

  // KPI Grid (context-aware by scene_type)
  const obs = active.observation;
  if (obs) {
    if (els.kpiWeek) {
      els.kpiWeek.textContent = `第${obs.week || cfg.current_week || 1}/${obs.total_weeks || 13}周`;
    }
    els.kpiScene.textContent = SCENE_LABELS[obs.scene_type] || obs.scene_type;
    if (obs.scene_type === "battle_turn") {
      els.kpiTurn.textContent = `${obs.turn} (剩${obs.turns_remaining})`;
      els.kpiStamina.textContent = `${obs.stamina}/${obs.max_stamina}`;
      els.kpiGenki.textContent = String(obs.genki);
      els.kpiScore.textContent = String(obs.score);
    } else if (obs.scene_type === "produce_schedule") {
      els.kpiTurn.textContent = "培育阶段";
      els.kpiStamina.textContent = `${obs.stamina}/${obs.max_stamina}`;
      els.kpiGenki.textContent = "—";
      els.kpiScore.textContent = `日程 ×${(obs.options || []).length}`;
    } else {
      els.kpiTurn.textContent = "非战斗";
      els.kpiStamina.textContent = "—";
      els.kpiGenki.textContent = "—";
      els.kpiScore.textContent = `选项 ×${(obs.options || []).length}`;
    }
  } else {
    els.kpiScene.textContent = "—";
    els.kpiTurn.textContent = "—";
    els.kpiStamina.textContent = "—";
    els.kpiGenki.textContent = "—";
    els.kpiScore.textContent = "—";
  }
  const limitCount = ((active.tracker || {}).played_limit_cards || []).length;
  els.kpiLimits.textContent = String(limitCount);

  els.obsBuffsRow.replaceChildren();
  if (obs && obs.buffs) {
    for (const [bk, bv] of Object.entries(obs.buffs)) {
      if (bv > 0) {
        const chip = document.createElement("span");
        chip.className = "buff-chip";
        chip.textContent = `${bk}: ${bv}`;
        els.obsBuffsRow.appendChild(chip);
      }
    }
  }

  // Latest decision banner
  const dec = active.decision;
  if (dec) {
    els.decisionSourceTag.textContent = dec.source.toUpperCase();
    els.decisionSourceTag.className = dec.source === "llm" ? "source-pill source-pill--llm" : "source-pill source-pill--rule";
    els.decisionMetrics.textContent = `选择 [${dec.choice_id}] · 耗时 ${Math.round(dec.latency_ms)}ms · Token ${dec.prompt_tokens}+${dec.completion_tokens}`;
    els.decisionReason.textContent = dec.reason || "";
  } else if (active.last_error) {
    els.decisionSourceTag.textContent = "ERROR";
    els.decisionSourceTag.className = "source-pill source-pill--rule";
    els.decisionMetrics.textContent = "执行异常";
    els.decisionReason.textContent = active.last_error;
  } else {
    els.decisionSourceTag.textContent = "READY";
    els.decisionSourceTag.className = "source-pill";
    els.decisionMetrics.textContent = "点击「单步执行」或「开始任务」驱动闭环";
    els.decisionReason.textContent = "等待当前步骤决策输出...";
  }

  renderPipeline(active);
  renderLegalActions(active);
  renderLogs(active);

  // Live frame & SVG coordinate overlay sync
  const meta = active.frame_meta || { width: 1080, height: 1920 };
  const fSource = active.frame_source || cfg.mode;
  if (active.running) {
    els.liveBadge.textContent = "RUNNING";
    if (els.liveDot) els.liveDot.className = "live-dot";
  } else if (fSource === "disconnected") {
    els.liveBadge.textContent = `${cfg.mode.toUpperCase()} 未连接`;
    if (els.liveDot) els.liveDot.className = "live-dot live-dot--offline";
  } else if (String(fSource).endsWith("_fallback")) {
    els.liveBadge.textContent = `${cfg.mode.toUpperCase()} (MOCK 回退)`;
    if (els.liveDot) els.liveDot.className = "live-dot live-dot--warn";
  } else {
    els.liveBadge.textContent = `${String(fSource).toUpperCase()} LIVE`;
    if (els.liveDot) els.liveDot.className = "live-dot";
  }
  els.liveResolution.textContent = `${meta.width}×${meta.height}`;
  els.liveCanvasWrap.style.setProperty("--frame-aspect", `${meta.width} / ${meta.height}`);
  els.zoomCanvasWrap.style.setProperty("--frame-aspect", `${meta.width} / ${meta.height}`);

  const frameKey = `${active.id}:${active.frame_seq}`;
  if (frameKey !== state.lastFrameKey) {
    state.lastFrameKey = frameKey;
    const url = `/api/frame?instance_id=${encodeURIComponent(active.id)}&seq=${active.frame_seq}`;
    els.liveFrameImg.src = url;
    els.liveFrameImg.width = meta.width;
    els.liveFrameImg.height = meta.height;
    els.zoomFrameImg.src = url;
    els.zoomFrameImg.width = meta.width;
    els.zoomFrameImg.height = meta.height;
  }

  renderOverlayToSvg(els.liveOverlaySvg, active, active.click_coords);
  renderOverlayToSvg(els.zoomOverlaySvg, active, active.click_coords);
}

async function refreshState() {
  try {
    const data = await apiGet("/api/state");
    state.full = data;
    renderTabs(data);
    renderActiveInstance(data);
    if (state.activeView === "monitor") {
      renderMonitorGrid(data);
    }
  } catch (err) {
    showToast(`状态同步失败: ${err.message}`);
  }
}

function scheduleRefresh() {
  if (state.refreshQueued) return;
  state.refreshQueued = true;
  window.setTimeout(async () => {
    state.refreshQueued = false;
    await refreshState();
  }, 80);
}

/* ==================== Library Explorer ==================== */
async function loadLibrary() {
  const kind = els.libKind.value;
  const plan = els.libPlan.value;
  const category = els.libCategory.value;
  const q = els.libQuery.value.trim();

  els.libCategoryWrap.hidden = kind === "drinks";
  els.libCountLabel.textContent = "检索本地数据库中...";
  try {
    const endpoint =
      kind === "drinks"
        ? `/api/library/drinks?plan=${encodeURIComponent(plan)}&q=${encodeURIComponent(q)}&limit=100`
        : `/api/library/cards?plan=${encodeURIComponent(plan)}&category=${encodeURIComponent(category)}&q=${encodeURIComponent(q)}&limit=100`;
    const data = await apiGet(endpoint);
    els.libCountLabel.textContent = `共匹配 ${data.total} 项（当前展示前 ${data.items.length} 项）`;
    els.libraryGrid.replaceChildren();

    for (const item of data.items) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "glass-card lib-item-card";

      const head = document.createElement("div");
      head.className = "lib-item-head";
      if (item.icon_url) {
        const img = document.createElement("img");
        img.className = "lib-item-icon";
        img.src = item.icon_url;
        img.alt = item.name || "";
        img.loading = "lazy";
        img.width = 56;
        img.height = 56;
        head.appendChild(img);
      }

      const titleCol = document.createElement("div");
      const title = document.createElement("h2");
      title.className = "lib-item-title";
      title.textContent = item.name || `#${item.id}`;
      const sub = document.createElement("span");
      sub.className = "muted-note";
      sub.textContent = `${item.plan || "common"} · ${item.rarity || ""}${item.upgraded ? " · +" : ""}`;
      titleCol.append(title, sub);
      head.appendChild(titleCol);

      const desc = document.createElement("p");
      desc.className = "lib-item-desc";
      desc.textContent = item.description || "";

      btn.append(head, desc);
      btn.addEventListener("click", () => openCardDetailModal(item));
      els.libraryGrid.appendChild(btn);
    }
  } catch (err) {
    els.libCountLabel.textContent = `加载失败: ${err.message}`;
  }
}

/* ==================== Event Bindings & SSE ==================== */
function bindEvents() {
  els.btnThemeToggle.addEventListener("click", toggleComfortTheme);
  for (const r of els.themeRadios) {
    r.addEventListener("change", () => {
      if (r.checked) applySchemePref(r.value);
    });
  }

  for (const btn of els.navButtons) {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  }

  els.btnAddInstance.addEventListener("click", async () => {
    const plan = els.cfgPlan.value || "sense";
    await apiPost("/api/instance/create", { plan });
    await refreshState();
    showToast("已新建独立配置实例");
  });

  const pushToolbarConfig = async () => {
    const mode = els.cfgMode.value;
    const payload = {
      plan: els.cfgPlan.value,
      mode,
      scenario_id: els.cfgScenarioId ? els.cfgScenarioId.value : "hajime_regular",
      current_week: els.cfgWeek ? Number(els.cfgWeek.value || 1) : 1,
      scenario_preset: els.cfgPreset.value,
      llm_mode: els.cfgLlmMode.value,
      steps: Number(els.cfgSteps.value || 5),
      delay: Number(els.cfgDelay.value || 0.5),
      auto_progress: els.chkAutoProgress.checked,
      fallback_to_mock: els.chkFallbackMock.checked,
    };
    const rawTarget = els.cfgTarget.value.trim();
    if (mode === "win32") {
      payload.win32_title = rawTarget;
    } else if (rawTarget.includes(":")) {
      const [h, p] = rawTarget.split(":");
      payload.adb_host = h.trim();
      if (Number(p) > 0) payload.adb_port = Number(p);
    } else if (rawTarget) {
      payload.adb_host = rawTarget;
    }
    try {
      await apiPost("/api/config", payload);
      await refreshState();
    } catch (err) {
      showToast(`配置应用失败: ${err.message}`);
    }
  };

  for (const ctrl of [
    els.cfgPlan,
    els.cfgMode,
    els.cfgPreset,
    els.cfgLlmMode,
    els.cfgSteps,
    els.cfgDelay,
    els.chkAutoProgress,
    els.chkFallbackMock,
  ]) {
    ctrl.addEventListener("change", pushToolbarConfig);
  }
  els.cfgTarget.addEventListener("change", pushToolbarConfig);

  els.chkOverlay.addEventListener("change", () => {
    state.showOverlay = els.chkOverlay.checked;
    if (state.full) {
      const active = state.full.active;
      renderOverlayToSvg(els.liveOverlaySvg, active, active.click_coords);
      renderOverlayToSvg(els.zoomOverlaySvg, active, active.click_coords);
    }
  });

  if (els.cfgScenarioId) {
    els.cfgScenarioId.addEventListener("change", async () => {
      await apiPost("/api/scenario", {
        scenario_id: els.cfgScenarioId.value,
        scenario_preset: "hajime_regular",
        week: 1,
        plan: els.cfgPlan.value,
      });
      await refreshState();
      showToast(`已切换剧本: ${els.cfgScenarioId.value}`);
    });
  }

  if (els.btnJumpWeek) {
    els.btnJumpWeek.addEventListener("click", async () => {
      const targetWeek = Number((els.cfgWeek && els.cfgWeek.value) || 1);
      await apiPost("/api/scenario", {
        scenario_id: els.cfgScenarioId ? els.cfgScenarioId.value : "hajime_regular",
        scenario_preset: "hajime_regular",
        week: targetWeek,
        plan: els.cfgPlan.value,
      });
      await refreshState();
      showToast(`已跳转至第 ${targetWeek} 周`);
    });
  }

  els.btnApplyScenario.addEventListener("click", async () => {
    try {
      await apiPost("/api/scenario", {
        plan: els.cfgPlan.value,
        scenario_id: els.cfgScenarioId ? els.cfgScenarioId.value : "hajime_regular",
        week: els.cfgWeek ? Number(els.cfgWeek.value || 1) : 1,
        scenario_preset: els.cfgPreset.value,
      });
      await refreshState();
      showToast(`已注入场景: ${els.cfgPreset.value}`);
    } catch (err) {
      showToast(`注入失败: ${err.message}`);
    }
  });

  els.btnInjectBattle.addEventListener("click", async () => {
    try {
      await apiPost("/api/scenario", {
        plan: els.cfgPlan.value,
        scenario_preset: "battle_turn",
        battle_overrides: {
          stamina: Number(els.simStamina.value || 22),
          genki: Number(els.simGenki.value || 0),
          score: Number(els.simScore.value || 0),
        },
      });
      await refreshState();
      showToast("已按自定义数值生成战斗合成帧");
    } catch (err) {
      showToast(`注入战斗帧失败: ${err.message}`);
    }
  });

  els.btnRefreshPreview.addEventListener("click", async () => {
    const origText = els.btnRefreshPreview.textContent;
    els.btnRefreshPreview.disabled = true;
    els.btnRefreshPreview.textContent = "抓取中...";
    try {
      const res = await apiPost("/api/preview");
      if (res.frame_source && String(res.frame_source).endsWith("_fallback")) {
        showToast("实机不可达，已自动回退至 Mock 合成帧");
      } else {
        showToast("已刷新当前预览帧与合法动作");
      }
    } catch (err) {
      showToast(`预览抓取失败: ${err.message}`);
    } finally {
      els.btnRefreshPreview.disabled = false;
      els.btnRefreshPreview.textContent = origText;
      await refreshState();
    }
  });

  els.btnTestConn.addEventListener("click", async () => {
    const origText = els.btnTestConn.textContent;
    els.btnTestConn.disabled = true;
    els.btnTestConn.textContent = "探测中...";
    try {
      let host = els.setAdbHost.value.trim();
      let port = Number(els.setAdbPort.value || 5555);
      const rawTarget = els.cfgTarget.value.trim();
      if (els.cfgMode.value !== "win32" && rawTarget) {
        if (rawTarget.includes(":")) {
          const [h, p] = rawTarget.split(":");
          if (h.trim()) host = h.trim();
          if (Number(p) > 0) port = Number(p);
        } else {
          host = rawTarget;
        }
      }
      const resp = await fetch("/api/test-adb", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          adb_host: host,
          adb_port: port,
          adb_root_pwd: els.setAdbRoot.value,
        }),
      });
      const res = await resp.json();
      showToast(res.message || (res.ok ? "连接成功" : "连接失败"));
    } catch (err) {
      showToast(`连接探测失败: ${err.message}`);
    } finally {
      els.btnTestConn.disabled = false;
      els.btnTestConn.textContent = origText;
    }
  });

  els.btnStep.addEventListener("click", async () => {
    els.btnStep.disabled = true;
    try {
      await apiPost("/api/step");
      await refreshState();
    } catch (err) {
      showToast(`单步执行失败: ${err.message}`);
    } finally {
      els.btnStep.disabled = false;
    }
  });

  els.btnReset.addEventListener("click", async () => {
    await apiPost("/api/reset");
    await refreshState();
    showToast("已重设状态追踪与控制器");
  });

  els.btnToggleRun.addEventListener("click", async () => {
    const running = state.full && state.full.active && state.full.active.running;
    try {
      if (running) {
        await apiPost("/api/stop");
        showToast("正在停止任务...");
      } else {
        await pushToolbarConfig();
        await apiPost("/api/start");
        showToast("已启动自动任务队列");
      }
      await refreshState();
    } catch (err) {
      showToast(`操作失败: ${err.message}`);
    }
  });

  els.btnZoomFrame.addEventListener("click", () => {
    els.frameZoomDialog.showModal();
  });

  els.btnClearLogs.addEventListener("click", async () => {
    await apiPost("/api/logs/clear");
    await refreshState();
  });

  els.monitorZoom.addEventListener("input", () => {
    els.monitorGrid.style.setProperty("--monitor-card-w", `${els.monitorZoom.value}px`);
  });

  els.btnMonitorRefresh.addEventListener("click", async () => {
    await apiPost("/api/preview", { all: true });
    await refreshState();
  });

  els.btnLibSearch.addEventListener("click", loadLibrary);
  for (const el of [els.libKind, els.libPlan, els.libCategory]) {
    el.addEventListener("change", loadLibrary);
  }
  els.libQuery.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      loadLibrary();
    }
  });

  els.btnSaveConn.addEventListener("click", async () => {
    try {
      await apiPost("/api/config", {
        adb_host: els.setAdbHost.value.trim(),
        adb_port: Number(els.setAdbPort.value || 5555),
        adb_root_pwd: els.setAdbRoot.value.trim(),
        win32_title: els.setWin32Title.value,
      });
      await refreshState();
      showToast("控制器连接参数已保存");
    } catch (err) {
      showToast(`保存失败: ${err.message}`);
    }
  });

  els.btnDiagAdb.addEventListener("click", async () => {
    els.diagAdbResult.textContent = "正在探测 ADB 连接...";
    try {
      const res = await fetch("/api/test-adb", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          adb_host: els.setAdbHost.value.trim(),
          adb_port: Number(els.setAdbPort.value || 5555),
          adb_root_pwd: els.setAdbRoot.value.trim(),
        }),
      }).then((r) => r.json());
      els.diagAdbResult.textContent = `${res.message} (${res.latency_ms}ms)`;
    } catch (err) {
      els.diagAdbResult.textContent = `探测异常: ${err.message}`;
    }
  });

  els.btnSaveLlm.addEventListener("click", async () => {
    try {
      const payload = {
        llm_mode: els.setLlmMode.value,
        llm_base_url: els.setLlmBase.value.trim(),
        llm_model: els.setLlmModel.value.trim(),
      };
      if (els.setLlmKey.value.trim()) {
        payload.llm_api_key = els.setLlmKey.value.trim();
        els.setLlmKey.value = "";
      }
      await apiPost("/api/config", payload);
      await refreshState();
      showToast("LLM 配置已保存");
    } catch (err) {
      showToast(`保存失败: ${err.message}`);
    }
  });

  els.btnDiagLlm.addEventListener("click", async () => {
    els.diagLlmResult.textContent = "正在测试 LLM 决策链路...";
    try {
      const res = await fetch("/api/test-llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }).then((r) => r.json());
      els.diagLlmResult.textContent = res.message;
    } catch (err) {
      els.diagLlmResult.textContent = `测试失败: ${err.message}`;
    }
  });
}

function startSse() {
  if (!("EventSource" in window)) return;
  const es = new EventSource("/api/events");
  es.addEventListener("state", scheduleRefresh);
  es.addEventListener("connected", scheduleRefresh);
}

/* ==================== Bootstrap ==================== */
setupDialogs();
syncThemeControls();
bindEvents();
refreshState();
startSse();
