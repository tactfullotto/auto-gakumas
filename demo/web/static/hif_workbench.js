/**
 * H.I.F 本戦 3-Profession Configurator, Manual Card/P-Item/Memory Editor,
 * Card/Drink Preference Tuner, and Live Automation Workbench.
 */

const LOADOUT_MODE_DESC = {
  hybrid:
    "混合模式：进入 H.I.F 本戦时自动长按扫描卡组与 P 道具，并自动合并第 3 页手动录入的 H.I.F 回忆效果与开局词条。",
  auto:
    "全自动扫描模式：完全依赖开局识图扫描器识别技能卡、P 道具与部分内置回忆（本页卡组仅供参考预览）。",
  manual:
    "纯手动录入模式：跳过开局扫牌，直接使用下方手填的技能卡组、P 道具及第 3 页回忆注入战斗模拟器与智能体。",
};

const LOADOUT_MODE_SHORT = {
  hybrid: "混合模式 (自动扫卡+手动回忆)",
  auto: "全自动扫描 (Auto Scan)",
  manual: "纯手动录入 (Manual)",
};

const WIKI_GROUP_TITLES = {
  exile_trouble_1: "1. 除外眠気（每场1次 · 使用后除外山札/弃牌中1张《眠気》）",
  extra_use_draw_1: "2. 追加使用数+1 & 抽1牌（每场1次 · 使用后+1使用数并抽1张牌）",
  extra_use_1: "3. 追加使用数+1（每场1次 · 使用后不耗行动点）",
  extra_use_2: "4. 追加使用数+1（每场2次 · 可多次免费连动）",
};

const hifState = {
  ctx: null,
  catalog: null,
  state: null,
  planCardsCache: new Map(), // plan -> Array<card>
  planDrinksCache: new Map(), // plan -> Array<drink>
  activeTab: "run",
  stage: "roster", // 'roster' | 'workbench'
  lastFrameSeq: -1,
  cardDraftById: new Map(), // local working state before saving when needed
  editingCustCardId: null,
};

const RARITY_RANK = {
  L: 6,
  SSR: 5,
  SR: 4,
  R: 3,
  N: 2,
  T: 1,
};

function isUsableForSelectedIdol(card, selectedPid) {
  if ((card.category || "") !== "signature") return true;
  if (!selectedPid) return true;
  const pid = Number(selectedPid);
  return Number(card.p_idol_id) === pid || Number(card.related_p_idol_id) === pid;
}

function sortCardPool(cards, sortMode = "rarity_desc") {
  const copy = [...cards];
  if (sortMode === "id_asc") {
    return copy.sort((a, b) => Number(a.id) - Number(b.id));
  }
  if (sortMode === "id_desc") {
    return copy.sort((a, b) => Number(b.id) - Number(a.id));
  }
  return copy.sort((a, b) => {
    const rDiff = (RARITY_RANK[b.rarity] || 0) - (RARITY_RANK[a.rarity] || 0);
    if (rDiff !== 0) return rDiff;
    const uDiff = Number(Boolean(b.upgraded)) - Number(Boolean(a.upgraded));
    if (uDiff !== 0) return uDiff;
    return Number(a.id) - Number(b.id);
  });
}

function byId(id) {
  return document.getElementById(id);
}

function makeEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined && text !== null) el.textContent = String(text);
  return el;
}

async function ensurePlanCards(plan) {
  const key = (plan || "logic").toLowerCase();
  if (hifState.planCardsCache.has(key)) {
    return hifState.planCardsCache.get(key);
  }
  const data = await hifState.ctx.apiGet(`/api/library/cards?plan=${encodeURIComponent(key)}&limit=1000`);
  const items = data.items || [];
  hifState.planCardsCache.set(key, items);
  return items;
}

async function ensurePlanDrinks(plan) {
  const key = (plan || "logic").toLowerCase();
  if (hifState.planDrinksCache.has(key)) {
    return hifState.planDrinksCache.get(key);
  }
  const data = await hifState.ctx.apiGet(`/api/library/drinks?plan=${encodeURIComponent(key)}&limit=100`);
  const items = data.items || [];
  hifState.planDrinksCache.set(key, items);
  return items;
}

function findCardMeta(cardId) {
  const cid = Number(cardId || 0);
  if (!cid) return null;
  for (const list of hifState.planCardsCache.values()) {
    const found = list.find((c) => Number(c.id || c.card_id) === cid);
    if (found) return found;
  }
  const deckCard = (hifState.state?.deck_cards || []).find((c) => Number(c.card_id) === cid);
  return deckCard || null;
}

function setStage(stage) {
  hifState.stage = stage;
  const roster = byId("hif-stage-roster");
  const wb = byId("hif-stage-workbench");
  if (roster) roster.hidden = stage !== "roster";
  if (wb) wb.hidden = stage !== "workbench";
}

function setActiveTab(tabName) {
  hifState.activeTab = tabName;
  for (const btn of document.querySelectorAll(".hif-subtab")) {
    const active = btn.dataset.hifTab === tabName;
    btn.classList.toggle("is-active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const panel of ["run", "loadout", "memory", "prefs"]) {
    const el = byId(`hif-tab-${panel}`);
    if (el) el.hidden = panel !== tabName;
  }
  if (tabName === "loadout") renderLoadoutTab();
  if (tabName === "memory") renderMemoryTab();
  if (tabName === "prefs") renderPrefsTab();
}

/* ==================== Stage 1: Profession & Character Roster ==================== */
function renderRosterStage() {
  const cat = hifState.catalog;
  const st = hifState.state;
  if (!cat || !st) return;

  const activePlan = (st.selected_plan || "logic").toLowerCase();
  const counts = { sense: 0, logic: 0, anomaly: 0 };
  for (const p of cat.p_idols || []) {
    const pl = (p.plan || "").toLowerCase();
    if (pl in counts) counts[pl] += 1;
  }
  for (const pl of ["sense", "logic", "anomaly"]) {
    const cntEl = byId(`hif-count-${pl}`);
    if (cntEl) cntEl.textContent = `${counts[pl]} P偶像`;
  }

  for (const card of document.querySelectorAll(".hif-prof-card")) {
    const isSelected = card.dataset.plan === activePlan;
    card.classList.toggle("is-selected", isSelected);
    card.setAttribute("aria-selected", isSelected ? "true" : "false");
  }

  // Sync existing library dropdowns
  for (const selId of ["hif-import-select", "hif-loadout-import-select"]) {
    const sel = byId(selId);
    if (!sel) continue;
    const prevVal = sel.value;
    sel.replaceChildren();
    const defOpt = makeEl("option", "", selId === "hif-import-select" ? "选择本地 produce_library*.json 导入..." : "从已有 produce_library*.json 载入...");
    defOpt.value = "";
    sel.appendChild(defOpt);
    for (const fn of cat.existing_libraries || []) {
      const opt = makeEl("option", "", fn);
      opt.value = fn;
      sel.appendChild(opt);
    }
    if (prevVal) sel.value = prevVal;
  }

  const query = (byId("hif-roster-search")?.value || "").trim().toLowerCase();
  const rarityFilter = byId("hif-roster-rarity")?.value || "all";
  const guideOnly = Boolean(byId("hif-roster-guide-only")?.checked);

  const filtered = (cat.p_idols || [])
    .filter((p) => (p.plan || "").toLowerCase() === activePlan)
    .filter((p) => rarityFilter === "all" || (p.rarity || "").toUpperCase() === rarityFilter)
    .filter((p) => !guideOnly || Boolean(p.has_guide))
    .filter((p) => {
      if (!query) return true;
      const hay = `${p.id} ${p.idol_name} ${p.title} ${p.recommended_effect || ""}`.toLowerCase();
      return hay.includes(query);
    })
    .sort((a, b) => {
      if (Boolean(a.has_guide) !== Boolean(b.has_guide)) return a.has_guide ? -1 : 1;
      const rarOrder = { SSR: 0, SR: 1, R: 2 };
      const ra = rarOrder[(a.rarity || "").toUpperCase()] ?? 9;
      const rb = rarOrder[(b.rarity || "").toUpperCase()] ?? 9;
      if (ra !== rb) return ra - rb;
      return Number(b.id) - Number(a.id);
    });

  const grid = byId("hif-roster-grid");
  if (!grid) return;
  grid.replaceChildren();

  if (filtered.length === 0) {
    grid.appendChild(makeEl("div", "glass-card hif-empty-hint", "当前筛选条件下没有匹配的 P 偶像。"));
    return;
  }

  for (const idol of filtered) {
    const card = makeEl("button", "glass-card hif-idol-card");
    card.type = "button";
    if (Number(st.selected_p_idol_id) === Number(idol.id)) {
      card.classList.add("is-active-idol");
    }

    const media = makeEl("div", "hif-idol-media");
    if (idol.icon_url) {
      const img = document.createElement("img");
      img.src = idol.icon_url;
      img.alt = `${idol.title} ${idol.idol_name}`;
      img.loading = "lazy";
      img.width = 64;
      img.height = 64;
      media.appendChild(img);
    } else {
      media.appendChild(makeEl("span", "hif-avatar-fallback", (idol.idol_name || "P").slice(0, 1)));
    }

    const body = makeEl("div", "hif-idol-body");
    const topRow = makeEl("div", "hif-idol-top");
    topRow.appendChild(makeEl("span", `badge-mini hif-rar-${(idol.rarity || "SSR").toLowerCase()}`, idol.rarity || "SSR"));
    topRow.appendChild(makeEl("span", "muted-note", `#${idol.id}`));
    if (idol.has_guide) {
      topRow.appendChild(makeEl("span", "hif-guide-tag", "★ 专属攻略"));
    }

    const nameEl = makeEl("h3", "hif-idol-name", `[${idol.title}] ${idol.idol_name}`);
    const effEl = makeEl("p", "hif-idol-eff", `推荐轴：${idol.recommended_effect || activePlan}`);

    body.append(topRow, nameEl, effEl);
    if (idol.unique_p_item?.name) {
      body.appendChild(makeEl("p", "hif-idol-item", `固有道具：${idol.unique_p_item.name}`));
    }

    const cta = makeEl("span", "hif-idol-cta", "选择并配置 →");
    card.append(media, body, cta);

    card.addEventListener("click", async () => {
      try {
        await hifState.ctx.apiPost("/api/hif/select-idol", {
          plan: activePlan,
          p_idol_id: idol.id,
          auto_seed_signature: true,
        });
        await refreshHifState();
        setStage("workbench");
        setActiveTab("run");
        hifState.ctx.showToast(`已选择 [${idol.title}] ${idol.idol_name}`);
      } catch (err) {
        hifState.ctx.showToast(`选择角色失败: ${err.message}`);
      }
    });
    grid.appendChild(card);
  }
}

/* ==================== Stage 2: Workbench Header & Tab 1 (Run) ==================== */
function renderWorkbenchHeader() {
  const st = hifState.state;
  if (!st) return;

  const plan = (st.selected_plan || "logic").toLowerCase();
  const idol = st.p_idol;

  const avImg = byId("hif-wb-avatar");
  const avFallback = byId("hif-wb-avatar-fallback");
  if (idol?.icon_url) {
    avImg.src = idol.icon_url;
    avImg.hidden = false;
    avFallback.hidden = true;
  } else {
    avImg.hidden = true;
    avFallback.hidden = false;
    avFallback.textContent = idol?.idol_name ? idol.idol_name.slice(0, 1) : plan.slice(0, 1).toUpperCase();
  }

  byId("hif-wb-idol-name").textContent = idol
    ? `[${idol.title}] ${idol.idol_name} (#${idol.id})`
    : `未指定角色 · ${plan.toUpperCase()} 通用配置`;

  const planPill = byId("hif-wb-plan-pill");
  planPill.textContent = plan.toUpperCase();
  planPill.className = `hif-prof-pill hif-prof-pill--${plan}`;

  const effPill = byId("hif-wb-eff-pill");
  effPill.textContent = idol?.recommended_effect || st.run_config?.scenario_id || "hif_honsen";

  const statusPill = byId("hif-wb-status-pill");
  statusPill.textContent = st.running ? "RUNNING" : "IDLE";
  statusPill.classList.toggle("is-running", Boolean(st.running));

  byId("hif-wb-idol-sub").textContent = idol?.unique_p_item?.name
    ? `固有道具：${idol.unique_p_item.name} · 当前录入 ${st.deck_ids?.length || 0} 张卡 / ${Object.keys(st.card_memory_hooks || {}).length} 条回忆`
    : `当前录入 ${st.deck_ids?.length || 0} 张技能卡 / ${st.p_items?.length || 0} 个 P 道具 / ${Object.keys(st.card_memory_hooks || {}).length} 条回忆`;

  for (const lblId of ["hif-launch-label", "hif-launch-main-label"]) {
    const el = byId(lblId);
    if (el) el.textContent = st.running ? "⏹ 停止 H.I.F 任务" : "▶ 启动 H.I.F 自动化";
  }
  for (const btnId of ["btn-hif-launch", "btn-hif-launch-main"]) {
    const btn = byId(btnId);
    if (btn) btn.classList.toggle("is-running", Boolean(st.running));
  }
}

function renderRunTab() {
  const st = hifState.state;
  if (!st) return;
  const rc = st.run_config || {};

  for (const r of document.querySelectorAll('input[name="hif-loadout-mode"]')) {
    r.checked = r.value === st.loadout_mode;
  }
  const modeChip = byId("hif-mode-chip");
  if (modeChip) modeChip.textContent = (st.loadout_mode || "hybrid").toUpperCase();

  const setVal = (id, val) => {
    const el = byId(id);
    if (el && document.activeElement !== el) el.value = val ?? "";
  };
  setVal("hif-adb-host", rc.adb_host || "127.0.0.1");
  setVal("hif-adb-port", rc.adb_port || 16384);
  setVal("hif-scenario-id", rc.scenario_id || "hif_honsen");
  setVal("hif-max-steps", rc.max_steps || 450);
  setVal("hif-delay", rc.delay ?? 0.5);
  setVal("hif-ppo-episodes", rc.ppo_episodes || 2000);
  setVal("hif-ismcts-sims", rc.ismcts_sims || 64);
  setVal("hif-library-out", rc.library_out || "");
  setVal("hif-csv-out", rc.csv_out || "");
  setVal("hif-debug-dir", rc.debug_dir || "");
  const chkHome = byId("hif-stop-home");
  if (chkHome) chkHome.checked = Boolean(rc.stop_on_home_return ?? true);

  const kpi = st.runtime_kpi || {};
  byId("hif-step-badge").textContent = `STEP ${st.step_count || 0}`;
  byId("hif-kpi-scene").textContent = kpi.scene_type || "idle";
  byId("hif-kpi-turn").textContent = kpi.week ? `W${kpi.week} · T${kpi.turn}/余${kpi.turns_remaining}` : "—";
  byId("hif-kpi-hp").textContent = kpi.max_stamina ? `${kpi.stamina}/${kpi.max_stamina} (${kpi.genki || 0})` : "—";
  byId("hif-kpi-score").textContent = Number(kpi.score || 0).toLocaleString();
  byId("hif-kpi-ppts").textContent = `${kpi.p_points || 0} P`;
  byId("hif-kpi-summary").textContent = `${st.deck_ids?.length || 0}卡 / ${Object.keys(st.card_memory_hooks || {}).length}回忆 / 禁${st.preferences?.banned_drink_ids?.length || 0}饮`;

  if (st.frame_seq !== hifState.lastFrameSeq) {
    hifState.lastFrameSeq = st.frame_seq;
    const img = byId("hif-live-img");
    if (img) img.src = `/api/hif/frame?seq=${st.frame_seq}`;
  }
  const fSrc = byId("hif-frame-source");
  if (fSrc) fSrc.textContent = `${st.frame_source || "idle"} · ${st.frame_meta?.width || 540}×${st.frame_meta?.height || 960}`;

  const logBox = byId("hif-logs-container");
  if (logBox) {
    logBox.replaceChildren();
    for (const entry of st.logs || []) {
      const row = makeEl("div", "log-line");
      row.appendChild(makeEl("span", "log-time", entry.time || ""));
      row.appendChild(makeEl("span", `log-tag log-tag--${entry.level || "info"}`, `[${(entry.level || "info").toUpperCase()}]`));
      row.appendChild(makeEl("span", "log-msg", entry.message || ""));
      logBox.appendChild(row);
    }
    logBox.scrollTop = logBox.scrollHeight;
  }
}

/* ==================== Tab 2: Cards & P-Items Manual/Auto Editor ==================== */
async function renderLoadoutTab() {
  const st = hifState.state;
  const cat = hifState.catalog;
  if (!st || !cat) return;

  const lm = st.loadout_mode || "hybrid";
  byId("hif-loadout-mode-label").textContent = LOADOUT_MODE_SHORT[lm] || lm;
  byId("hif-loadout-mode-desc").textContent = LOADOUT_MODE_DESC[lm] || "";

  const stats = st.idol_stats || {};
  for (const [id, k] of [["hif-stat-vo", "vocal"], ["hif-stat-da", "dance"], ["hif-stat-vi", "visual"]]) {
    const el = byId(id);
    if (el && document.activeElement !== el) el.value = stats[k] ?? 0;
  }

  // Deck list
  const deckList = byId("hif-deck-list");
  const deckCards = st.deck_cards || [];
  byId("hif-deck-count").textContent = String(deckCards.length);
  deckList.replaceChildren();

  if (deckCards.length === 0) {
    deckList.appendChild(makeEl("div", "hif-empty-note", "暂无录入技能卡。可点击右上角「载入角色固有卡」，或在右侧卡池点击加入，或切换到全自动扫描。"));
  } else {
    deckCards.forEach((card, idx) => {
      const row = makeEl("div", "hif-deck-row");

      const left = makeEl("div", "hif-deck-left");
      if (card.icon_url) {
        const img = document.createElement("img");
        img.src = card.icon_url;
        img.alt = card.name;
        img.width = 40;
        img.height = 40;
        img.className = "hif-mini-icon";
        img.addEventListener("click", () => hifState.ctx.openCardModal({ ...card, id: card.card_id, type: card.card_type }));
        left.appendChild(img);
      }

      const info = makeEl("div", "hif-deck-info");
      const title = makeEl("strong", "", `${idx + 1}. ${card.name} (#${card.card_id})`);
      const meta = makeEl(
        "span",
        "muted-note",
        `${(card.card_type || "").toUpperCase()} · ${card.rarity}${card.limit_count === 1 ? " · 限1" : ""}`,
      );
      info.append(title, meta);

      const hookTags = formatHookPills(card.memory_hooks);
      const custTags = formatCustomizationPills(card.customizations);
      if (hookTags.length > 0 || custTags.length > 0) {
        const tagWrap = makeEl("div", "hif-inline-tags");
        custTags.forEach((t) => tagWrap.appendChild(makeEl("span", "badge-mini hif-cust-pill", t)));
        hookTags.forEach((t) => tagWrap.appendChild(makeEl("span", "badge-mini hif-hook-pill", t)));
        info.appendChild(tagWrap);
      }
      left.appendChild(info);

      const actions = makeEl("div", "hif-deck-actions");
      const btnDup = makeEl("button", "btn-secondary btn-compact", "+1");
      btnDup.type = "button";
      btnDup.addEventListener("click", () => mutateDeck((ids) => [...ids, card.card_id]));

      const btnCust = makeEl("button", "btn-secondary btn-compact", "✨强化");
      btnCust.type = "button";
      btnCust.title = "配置开局入手、限次解除、数值增加等卡牌強化 (カスタマイズ)";
      btnCust.addEventListener("click", () => openCardCustomizationModal(card));

      const btnMem = makeEl("button", "btn-secondary btn-compact", "回忆");
      btnMem.type = "button";
      btnMem.addEventListener("click", () => {
        setActiveTab("memory");
        const sel = byId("hif-hook-card-select");
        if (sel) {
          sel.value = String(card.card_id);
          loadHookFormForCard(card.card_id);
        }
      });

      const btnDel = makeEl("button", "btn-secondary btn-compact hif-btn-danger", "移除");
      btnDel.type = "button";
      btnDel.addEventListener("click", () =>
        mutateDeck((ids) => ids.filter((_, i) => i !== idx))
      );

      actions.append(btnDup, btnCust, btnMem, btnDel);
      row.append(left, actions);
      deckList.appendChild(row);
    });
  }

  // P-Item presets & active list
  const presetBox = byId("hif-pitem-presets");
  presetBox.replaceChildren();
  const activePid = st.selected_p_idol_id ? Number(st.selected_p_idol_id) : null;
  const sortedPresets = [...(cat.p_items || [])].sort((a, b) => {
    const aMatch = activePid && Number(a.p_idol_id) === activePid ? 0 : 1;
    const bMatch = activePid && Number(b.p_idol_id) === activePid ? 0 : 1;
    return aMatch - bMatch;
  });

  for (const preset of sortedPresets) {
    const chip = makeEl("button", "hif-preset-chip", `+ ${preset.name}`);
    chip.type = "button";
    chip.title = preset.description || "";
    if (activePid && Number(preset.p_idol_id) === activePid) {
      chip.classList.add("is-highlighted");
    }
    chip.addEventListener("click", async () => {
      const curItems = [...(hifState.state?.p_items || [])];
      if (curItems.some((it) => it.canonical_name === preset.canonical_name || Number(it.item_id) === Number(preset.item_id))) {
        hifState.ctx.showToast(`已存在道具：${preset.name}`);
        return;
      }
      const pairIdx = preset.upgrade_pair_id != null
        ? curItems.findIndex((it) => Number(it.item_id) === Number(preset.upgrade_pair_id))
        : -1;
      if (pairIdx >= 0) {
        curItems[pairIdx] = { ...preset };
      } else {
        curItems.push({ ...preset });
      }
      await saveLoadout({ p_items: curItems });
    });
    presetBox.appendChild(chip);
  }

  const pItemList = byId("hif-pitem-list");
  const pItems = st.p_items || [];
  byId("hif-pitem-count").textContent = String(pItems.length);
  pItemList.replaceChildren();
  const catPItems = cat.p_items || [];
  for (let i = 0; i < pItems.length; i++) {
    const it = pItems[i];
    const row = makeEl("div", "hif-pitem-row");
    const left = makeEl("div", "hif-pitem-info");
    left.appendChild(makeEl("strong", "", `${it.canonical_name || it.name} (${it.rarity || "SSR"} · ${it.mode === "stage" ? "战中" : "培育期"} · ${it.source_type || "support"})`));
    if (it.description) {
      left.appendChild(makeEl("span", "muted-note", it.description));
    }
    const actionsWrap = makeEl("div", "hif-item-actions");
    const catMatch = catPItems.find((p) => Number(p.item_id) === Number(it.item_id) || p.canonical_name === (it.canonical_name || it.name));
    const pairId = it.upgrade_pair_id ?? catMatch?.upgrade_pair_id;
    const pairDef = pairId != null ? catPItems.find((p) => Number(p.item_id) === Number(pairId)) : null;
    if (pairDef) {
      const isUp = Boolean(it.upgraded ?? catMatch?.upgraded ?? String(it.canonical_name || it.name || "").endsWith("+"));
      const toggleBtn = makeEl("button", "btn-secondary btn-compact", isUp ? "切换未强化" : "切换强化+");
      toggleBtn.type = "button";
      toggleBtn.addEventListener("click", async () => {
        const next = pItems.map((item, idx) => (idx === i ? { ...pairDef } : item));
        await saveLoadout({ p_items: next });
      });
      actionsWrap.appendChild(toggleBtn);
    }
    const rmBtn = makeEl("button", "btn-secondary btn-compact hif-btn-danger", "删除");
    rmBtn.type = "button";
    rmBtn.addEventListener("click", async () => {
      const next = pItems.filter((_, idx) => idx !== i);
      await saveLoadout({ p_items: next });
    });
    actionsWrap.appendChild(rmBtn);
    row.append(left, actionsWrap);
    pItemList.appendChild(row);
  }

  // Card Pool Picker
  await renderCardPicker();
}

async function renderCardPicker() {
  const st = hifState.state;
  if (!st) return;
  const plan = (st.selected_plan || "logic").toLowerCase();
  const cards = await ensurePlanCards(plan);

  const q = (byId("hif-card-picker-q")?.value || "").trim().toLowerCase();
  const typeF = byId("hif-card-picker-type")?.value || "all";
  const catF = byId("hif-card-picker-cat")?.value || "usable";
  const rarityF = byId("hif-card-picker-rarity")?.value || "all";
  const upF = byId("hif-card-picker-up")?.value || "all";
  const sortF = byId("hif-card-picker-sort")?.value || "rarity_desc";

  const filtered = cards.filter((c) => {
    if (typeF !== "all" && (c.type || "").toLowerCase() !== typeF) return false;
    if (catF === "usable") {
      if (!isUsableForSelectedIdol(c, st.selected_p_idol_id)) return false;
    } else if (catF !== "all" && (c.category || "") !== catF) {
      return false;
    }
    if (rarityF !== "all" && (c.rarity || "").toUpperCase() !== rarityF.toUpperCase()) return false;
    if (upF === "upgraded" && !c.upgraded) return false;
    if (upF === "base" && c.upgraded) return false;
    if (q) {
      const hay = `${c.id} ${c.name} ${c.description || ""}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const sorted = sortCardPool(filtered, sortF);
  byId("hif-card-picker-count").textContent = `匹配 ${sorted.length} 张（全量展示）`;
  const grid = byId("hif-card-picker-grid");
  grid.replaceChildren();

  for (const c of sorted) {
    const item = makeEl("div", "hif-picker-card");
    const head = makeEl("div", "hif-picker-head");
    if (c.icon_url) {
      const img = document.createElement("img");
      img.src = c.icon_url;
      img.alt = c.name;
      img.width = 40;
      img.height = 40;
      img.loading = "lazy";
      img.addEventListener("click", () => hifState.ctx.openCardModal(c));
      head.appendChild(img);
    }
    const metaCol = makeEl("div", "hif-picker-meta");
    metaCol.appendChild(makeEl("strong", "", `${c.name} (#${c.id})`));
    const custCount = (c.available_customization_ids || []).length;
    const custHint = custCount > 0 ? ` · 可強化×${custCount}` : "";
    metaCol.appendChild(
      makeEl(
        "span",
        "muted-note",
        `${(c.type || "").toUpperCase()} · ${c.rarity}${c.limit_count === 1 ? " · 限1" : ""}${custHint}`,
      ),
    );
    head.appendChild(metaCol);

    const addBtn = makeEl("button", "btn-secondary btn-compact", "＋加入");
    addBtn.type = "button";
    addBtn.addEventListener("click", () => mutateDeck((ids) => [...ids, c.id]));

    const desc = makeEl("p", "hif-picker-desc", c.description || "");
    item.append(head, desc, addBtn);
    grid.appendChild(item);
  }
}

function formatCustomizationPills(custEntry) {
  if (!custEntry || typeof custEntry !== "object") return [];
  const pills = [];
  const custCatalog = hifState.catalog?.customizations || {};
  const lvls = custEntry.levels || {};
  for (const [cid, lvRaw] of Object.entries(lvls)) {
    const lv = Number(lvRaw || 0);
    if (lv <= 0) continue;
    const def = custCatalog[String(cid)];
    const nm = def?.name || `強化#${cid}`;
    pills.push(def && def.max_level > 1 ? `✨${nm} Lv${lv}` : `✨${nm}`);
  }
  const mods = custEntry.modifiers || {};
  if (mods.force_initial_hand) pills.push("✨開始時手札");
  if (mods.remove_limit) pills.push("✨制限削除");
  if (Number(mods.cost_reduction || 0) !== 0) pills.push(`✨コスト-${mods.cost_reduction}`);
  const modLabels = [
    ["score_delta", "パラメータ"],
    ["good_impression_delta", "好印象"],
    ["motivation_delta", "やる気"],
    ["genki_delta", "元気"],
    ["good_condition_delta", "好調"],
    ["concentration_delta", "集中"],
    ["full_power_charge_delta", "全力値"],
  ];
  for (const [mk, lbl] of modLabels) {
    const v = Number(mods[mk] || 0);
    if (v !== 0) pills.push(`✨${lbl}${v > 0 ? `+${v}` : v}`);
  }
  return pills;
}

function openCardCustomizationModal(card) {
  const dialog = byId("hif-card-cust-dialog");
  if (!dialog || !card) return;
  const cid = Number(card.card_id || card.id);
  hifState.editingCustCardId = cid;

  const fullMeta = findCardMeta(cid) || card;
  byId("hif-cust-dialog-title").textContent = `技能卡强化 · ${card.name} (#${cid})`;
  byId("hif-cust-card-meta").textContent = `${(card.card_type || card.type || "").toUpperCase()} · ${card.rarity || ""} ｜ 原始效果：${card.description || fullMeta.description || "—"}`;

  const saved = (hifState.state?.card_customizations || {})[String(cid)] || card.customizations || {};
  const savedLevels = saved.levels || {};
  const savedMods = saved.modifiers || {};

  const availIds = card.available_customization_ids || fullMeta.available_customization_ids || [];
  const custCatalog = hifState.catalog?.customizations || {};
  const officialWrap = byId("hif-cust-official-wrap");
  officialWrap.replaceChildren();

  if (availIds.length === 0) {
    officialWrap.appendChild(
      makeEl(
        "div",
        "muted-note",
        "该卡无官方固定自定义词条（如固有卡或未强化基础卡），可直接使用下方「2. 通用额外强化与数值微调」配置开局入手或数值增加。",
      ),
    );
  } else {
    for (const rawCustId of availIds) {
      const custIdStr = String(rawCustId);
      const def = custCatalog[custIdStr] || { id: rawCustId, name: `強化 #${custIdStr}`, max_level: 1, summary: "" };
      const row = makeEl("div", "hif-cust-official-row");

      const info = makeEl("div", "hif-cust-official-info");
      const priceTxt = Array.isArray(def.prices) && def.prices.length > 0 ? ` (${def.prices.join("/")}P)` : "";
      info.appendChild(makeEl("strong", "", `${def.name}${priceTxt} [#${custIdStr}]`));
      if (def.summary) {
        info.appendChild(makeEl("span", "muted-note", def.summary));
      }

      const sel = document.createElement("select");
      sel.className = "hif-cust-level-select";
      sel.dataset.custId = custIdStr;
      const offOpt = makeEl("option", "", "未启用");
      offOpt.value = "0";
      sel.appendChild(offOpt);
      const maxLv = Math.max(1, Number(def.max_level || 1));
      for (let lv = 1; lv <= maxLv; lv++) {
        const opt = makeEl("option", "", maxLv > 1 ? `启用 Lv${lv}` : "启用");
        opt.value = String(lv);
        sel.appendChild(opt);
      }
      sel.value = String(Number(savedLevels[custIdStr] || 0));

      row.append(info, sel);
      officialWrap.appendChild(row);
    }
  }

  byId("hif-cust-mod-fih").checked = Boolean(savedMods.force_initial_hand);
  byId("hif-cust-mod-nolimit").checked = Boolean(savedMods.remove_limit);
  byId("hif-cust-mod-cost").value = String(Number(savedMods.cost_reduction || 0));
  byId("hif-cust-mod-score").value = String(Number(savedMods.score_delta || 0));
  byId("hif-cust-mod-gi").value = String(Number(savedMods.good_impression_delta || 0));
  byId("hif-cust-mod-mot").value = String(Number(savedMods.motivation_delta || 0));
  byId("hif-cust-mod-genki").value = String(Number(savedMods.genki_delta || 0));
  byId("hif-cust-mod-gc").value = String(Number(savedMods.good_condition_delta || 0));
  byId("hif-cust-mod-conc").value = String(Number(savedMods.concentration_delta || 0));
  byId("hif-cust-mod-fp").value = String(Number(savedMods.full_power_charge_delta || 0));

  dialog.showModal();
}

async function mutateDeck(transformFn) {
  const curIds = [...(hifState.state?.deck_ids || [])];
  const nextIds = transformFn(curIds);
  await saveLoadout({ deck_ids: nextIds });
}

async function saveLoadout(extra = {}) {
  const payload = {
    idol_stats: {
      vocal: Number(byId("hif-stat-vo")?.value || 2400),
      dance: Number(byId("hif-stat-da")?.value || 1400),
      visual: Number(byId("hif-stat-vi")?.value || 1800),
    },
    ...extra,
  };
  await hifState.ctx.apiPost("/api/hif/loadout", payload);
  await refreshHifState();
}

/* ==================== Tab 3: Manual H.I.F Memory Editor ==================== */
function formatHookPills(hooks) {
  if (!hooks || typeof hooks !== "object") return [];
  const tags = [];
  if (hooks.force_initial_hand) tags.push("开局必上手");
  if (hooks.exiles_trouble_on_play) tags.push("除外1眠気");
  if (hooks.free_extra_use_first_play || Number(hooks.extra_use_limit || 0) > 0) {
    const lim = Number(hooks.extra_use_limit || 1);
    const drawS = Number(hooks.draw_on_play || 0) > 0 ? `+抽${hooks.draw_on_play}牌` : "";
    tags.push(`+1使用数${drawS}(限${lim}次)`);
  }
  return tags;
}

function loadHookFormForCard(cardId) {
  const hooks = (hifState.state?.card_memory_hooks || {})[String(cardId)] || {};
  byId("hif-hook-force-init").checked = Boolean(hooks.force_initial_hand);
  byId("hif-hook-exile").checked = Boolean(hooks.exiles_trouble_on_play);
  const extraOn = Boolean(hooks.free_extra_use_first_play || Number(hooks.extra_use_limit || 0) > 0);
  byId("hif-hook-extra-use").checked = extraOn;
  byId("hif-hook-extra-limit").value = String(Number(hooks.extra_use_limit || 1) >= 2 ? 2 : 1);
  byId("hif-hook-draw").checked = Number(hooks.draw_on_play || 0) > 0;
}

async function renderMemoryTab(options = {}) {
  const st = hifState.state;
  const cat = hifState.catalog;
  if (!st || !cat) return;

  const plan = (st.selected_plan || "logic").toLowerCase();
  const planCards = await ensurePlanCards(plan);
  const memCat = cat.memory_catalog || {};

  if (memCat.rule_note) {
    byId("hif-memory-rule-note").textContent = memCat.rule_note;
  }
  byId("hif-wiki-plan-tag").textContent = plan.toUpperCase();

  // Populate card select (Deck cards first, then rest of plan pool)
  const sel = byId("hif-hook-card-select");
  const prevSel = sel.value;
  sel.replaceChildren();

  const seenInSelect = new Set();
  const deckGroup = document.createElement("optgroup");
  deckGroup.label = "当前卡组中的技能卡";
  for (const dc of st.deck_cards || []) {
    if (seenInSelect.has(dc.card_id)) continue;
    seenInSelect.add(dc.card_id);
    const opt = makeEl("option", "", `${dc.name} (#${dc.card_id} · ${dc.rarity})`);
    opt.value = String(dc.card_id);
    deckGroup.appendChild(opt);
  }
  if (deckGroup.childElementCount > 0) sel.appendChild(deckGroup);

  const poolGroup = document.createElement("optgroup");
  poolGroup.label = `${plan.toUpperCase()} 全职业可配置技能卡`;
  for (const pc of planCards) {
    if (seenInSelect.has(pc.id) || (pc.type || "") === "trouble") continue;
    seenInSelect.add(pc.id);
    const opt = makeEl("option", "", `${pc.name} (#${pc.id} · ${pc.rarity})`);
    opt.value = String(pc.id);
    poolGroup.appendChild(opt);
  }
  sel.appendChild(poolGroup);
  if (prevSel && [...sel.options].some((o) => o.value === prevSel)) {
    sel.value = prevSel;
  } else if (sel.options.length > 0) {
    loadHookFormForCard(sel.value);
  }

  // Render active hooks
  const hooksList = byId("hif-active-hooks-list");
  const hooksEntries = Object.entries(st.hooks_enriched || {});
  byId("hif-hook-count").textContent = String(hooksEntries.length);
  hooksList.replaceChildren();

  if (hooksEntries.length === 0) {
    hooksList.appendChild(makeEl("div", "hif-empty-note", "暂未配置指定技能的回忆追加效果。可从上方表单或右侧攻略站速查表点击一键绑定。"));
  } else {
    for (const [cid, item] of hooksEntries) {
      const row = makeEl("div", "hif-hook-row");
      const left = makeEl("div", "hif-deck-left");
      if (item.icon_url) {
        const img = document.createElement("img");
        img.src = item.icon_url;
        img.alt = item.name;
        img.width = 36;
        img.height = 36;
        img.className = "hif-mini-icon";
        left.appendChild(img);
      }
      const info = makeEl("div", "hif-deck-info");
      info.appendChild(makeEl("strong", "", `${item.name} (#${cid})`));
      const tagsWrap = makeEl("div", "hif-inline-tags");
      for (const pill of formatHookPills(item.hooks)) {
        tagsWrap.appendChild(makeEl("span", "badge-mini hif-hook-pill", pill));
      }
      info.appendChild(tagsWrap);
      left.appendChild(info);

      const rm = makeEl("button", "btn-secondary btn-compact hif-btn-danger", "删除");
      rm.type = "button";
      rm.addEventListener("click", async () => {
        const nextHooks = { ...(hifState.state?.card_memory_hooks || {}) };
        delete nextHooks[cid];
        await hifState.ctx.apiPost("/api/hif/memories", { card_memory_hooks: nextHooks });
        await refreshHifState();
      });
      row.append(left, rm);
      hooksList.appendChild(row);
    }
  }

  // Render Wiki Quick-Reference Presets for active plan
  const wikiBox = byId("hif-wiki-presets-container");
  wikiBox.replaceChildren();
  const planGroups = (memCat.plans || {})[plan] || {};
  for (const [gKey, entries] of Object.entries(planGroups)) {
    const sec = makeEl("div", "hif-wiki-group");
    sec.appendChild(makeEl("h3", "hif-wiki-group-title", WIKI_GROUP_TITLES[gKey] || gKey));
    const chipWrap = makeEl("div", "hif-chip-pool");
    for (const entry of entries) {
      const chip = makeEl("button", "hif-preset-chip", `+ ${entry.upgraded_name || entry.name}`);
      chip.type = "button";
      chip.title = `${entry.template_label_ja}\n点击一键绑定到 #${entry.upgraded_card_id || entry.card_id}`;
      const targetCid = String(entry.upgraded_card_id || entry.card_id);
      if ((st.card_memory_hooks || {})[targetCid] || (st.card_memory_hooks || {})[String(entry.card_id)]) {
        chip.classList.add("is-highlighted");
      }
      chip.addEventListener("click", async () => {
        const curHooks = { ...(hifState.state?.card_memory_hooks || {}) };
        const existing = { ...(curHooks[targetCid] || {}) };
        curHooks[targetCid] = { ...existing, ...(entry.hooks || {}) };
        await hifState.ctx.apiPost("/api/hif/memories", { card_memory_hooks: curHooks });
        await refreshHifState();
        hifState.ctx.showToast(`已绑定 H.I.F 回忆：${entry.upgraded_name || entry.name} -> ${entry.template_label_zh}`);
      });
      chipWrap.appendChild(chip);
    }
    sec.appendChild(chipWrap);
    wikiBox.appendChild(sec);
  }

  renderMemorySummary(st);

  // Render 4 Memory Slots
  const slotsGrid = byId("hif-memory-slots-grid");
  const activeEl = document.activeElement;
  const editingSlotField = Boolean(
    activeEl
      && activeEl.closest?.("#hif-memory-slots-grid")
      && ["INPUT", "SELECT", "TEXTAREA"].includes(activeEl.tagName)
  );
  const hasDirtySlotDraft = Boolean(
    slotsGrid?.querySelector?.('.hif-slot-add-bar[data-dirty="1"]')
  );
  if (!options?.forceSlotsRender && (editingSlotField || hasDirtySlotDraft)) {
    return;
  }

  slotsGrid.replaceChildren();
  const passiveTemplates = (memCat.passive_templates || []).filter(
    (pt) => !pt.plans || pt.plans.includes("all") || pt.plans.includes(plan)
  );
  const slots = st.memory_slots || [];

  for (let sIdx = 0; sIdx < 4; sIdx++) {
    const slotData = slots[sIdx] || { slot: sIdx + 1, label: `メモリー #${sIdx + 1}`, owner_name: "", passives: [] };
    const card = makeEl("div", "glass-card hif-slot-card");
    card.dataset.slotIndex = String(sIdx);

    const head = makeEl("div", "hif-slot-head");
    head.appendChild(makeEl("strong", "", `编成回忆 #${sIdx + 1}`));
    const ownerInput = document.createElement("input");
    ownerInput.type = "text";
    ownerInput.className = "hif-slot-owner";
    ownerInput.placeholder = "备注标题（例：美鈴 S+ 回忆）";
    ownerInput.value = slotData.owner_name || "";
    head.appendChild(ownerInput);
    card.appendChild(head);

    const passiveList = makeEl("div", "hif-slot-passives");
    const curPassives = slotData.passives || [];
    for (let pIdx = 0; pIdx < curPassives.length; pIdx++) {
      const mp = curPassives[pIdx];
      const pRow = makeEl("div", "hif-passive-row");
      pRow.appendChild(makeEl("span", "", mp.label || mp.key));
      if (mp.key === "custom_note") {
        pRow.appendChild(makeEl("strong", "", mp.note || ""));
      } else {
        const valEdit = document.createElement("input");
        valEdit.type = "number";
        valEdit.min = "0";
        valEdit.max = "999";
        valEdit.className = "hif-passive-val";
        valEdit.dataset.passiveIndex = String(pIdx);
        valEdit.style.width = "64px";
        valEdit.value = String(mp.value ?? 0);
        valEdit.addEventListener("input", () => {
          valEdit.dataset.dirty = "1";
        });
        valEdit.addEventListener("change", () => {
          valEdit.dataset.dirty = "1";
        });
        pRow.appendChild(valEdit);
      }
      const delP = makeEl("button", "btn-secondary btn-compact", "✕");
      delP.type = "button";
      delP.addEventListener("click", async () => {
        const { nextSlots } = collectMemorySlotsFromDom({ commitDirtyDrafts: false });
        nextSlots[sIdx].passives = (nextSlots[sIdx].passives || []).filter((_, i) => i !== pIdx);
        await hifState.ctx.apiPost("/api/hif/memories", { memory_slots: nextSlots });
        await refreshHifState({ forceSlotsRender: true });
      });
      pRow.appendChild(delP);
      passiveList.appendChild(pRow);
    }
    card.appendChild(passiveList);

    const addBar = makeEl("div", "hif-slot-add-bar");
    const tmplSel = document.createElement("select");
    tmplSel.className = "hif-slot-draft-key";
    for (const pt of passiveTemplates) {
      const opt = makeEl("option", "", pt.label);
      opt.value = pt.key;
      tmplSel.appendChild(opt);
    }
    const valInput = document.createElement("input");
    valInput.type = "number";
    valInput.className = "hif-slot-draft-val";
    valInput.min = "0";
    valInput.max = "999";
    valInput.value = String(passiveTemplates[0]?.default_value ?? 2);
    const noteInput = document.createElement("input");
    noteInput.type = "text";
    noteInput.className = "hif-slot-draft-note";
    noteInput.placeholder = "备注内容";
    noteInput.hidden = true;

    const markDraftDirty = () => {
      addBar.dataset.dirty = "1";
    };
    tmplSel.addEventListener("change", () => {
      const chosen = passiveTemplates.find((pt) => pt.key === tmplSel.value);
      const isCustom = tmplSel.value === "custom_note";
      valInput.hidden = isCustom;
      noteInput.hidden = !isCustom;
      if (chosen && !isCustom) valInput.value = String(chosen.default_value ?? 1);
      markDraftDirty();
    });
    valInput.addEventListener("input", markDraftDirty);
    valInput.addEventListener("change", markDraftDirty);
    noteInput.addEventListener("input", markDraftDirty);
    noteInput.addEventListener("change", markDraftDirty);

    const addBtn = makeEl("button", "btn-secondary btn-compact", "+词条");
    addBtn.type = "button";
    const submitAdd = async () => {
      const chosen = passiveTemplates.find((pt) => pt.key === tmplSel.value);
      const { nextSlots } = collectMemorySlotsFromDom({ commitDirtyDrafts: false });
      nextSlots[sIdx].owner_name = ownerInput.value;
      nextSlots[sIdx].passives.push({
        key: tmplSel.value,
        label: chosen?.label || tmplSel.value,
        value: tmplSel.value === "custom_note" ? 0 : Number(valInput.value || 0),
        note: noteInput.value.trim(),
      });
      await hifState.ctx.apiPost("/api/hif/memories", { memory_slots: nextSlots });
      await refreshHifState({ forceSlotsRender: true });
    };
    addBtn.addEventListener("click", submitAdd);
    valInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        submitAdd();
      }
    });
    noteInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        submitAdd();
      }
    });

    addBar.append(tmplSel, valInput, noteInput, addBtn);
    card.appendChild(addBar);
    slotsGrid.appendChild(card);
  }
}

function renderMemorySummary(st) {
  const sumEl = byId("hif-memory-passives-summary");
  if (!sumEl) return;
  const passives = st?.memory_passives || [];
  if (!passives.length) {
    sumEl.textContent = "尚未配置开局通用词条。可直接在下方槽位选择词条并填写数值，点击「+词条」或直接点击「保存 4 槽位设置」提交。";
    return;
  }
  const agg = new Map();
  for (const mp of passives) {
    if (!mp || !mp.key) continue;
    if (mp.key === "custom_note") continue;
    const val = Number(mp.value || 0);
    if (val <= 0) continue;
    const lbl = mp.label || mp.key;
    agg.set(lbl, (agg.get(lbl) || 0) + val);
  }
  const parts = Array.from(agg.entries()).map(([lbl, total]) => `${lbl.replace("+X", `+${total}`)}`);
  sumEl.textContent = `当前已生效开局词条（共 ${passives.length} 条）：${parts.join(" ｜ ") || `${passives.length} 条备注`}`;
}

function collectMemorySlotsFromDom({ commitDirtyDrafts = true } = {}) {
  const nextSlots = structuredClone(hifState.state?.memory_slots || []);
  while (nextSlots.length < 4) {
    nextSlots.push({
      slot: nextSlots.length + 1,
      label: `メモリー #${nextSlots.length + 1}`,
      owner_name: "",
      passives: [],
    });
  }
  let hasUnsavedChanges = false;
  const memCat = hifState.catalog?.memory_catalog || {};
  const passiveTemplates = memCat.passive_templates || [];

  document.querySelectorAll("#hif-memory-slots-grid .hif-slot-card").forEach((cardEl) => {
    const idx = Number(cardEl.dataset.slotIndex);
    if (!Number.isInteger(idx) || idx < 0 || idx >= nextSlots.length) return;
    const slotObj = nextSlots[idx];
    if (!Array.isArray(slotObj.passives)) slotObj.passives = [];

    const ownerInp = cardEl.querySelector(".hif-slot-owner");
    if (ownerInp && ownerInp.value !== (slotObj.owner_name || "")) {
      slotObj.owner_name = ownerInp.value;
      hasUnsavedChanges = true;
    }

    cardEl.querySelectorAll(".hif-passive-val").forEach((valEl) => {
      const pIdx = Number(valEl.dataset.passiveIndex);
      if (Number.isInteger(pIdx) && slotObj.passives[pIdx]) {
        const nextVal = Math.max(0, Number(valEl.value || 0));
        if (nextVal !== Number(slotObj.passives[pIdx].value || 0) || valEl.dataset.dirty === "1") {
          slotObj.passives[pIdx].value = nextVal;
          hasUnsavedChanges = true;
        }
      }
    });

    if (commitDirtyDrafts) {
      const addBar = cardEl.querySelector(".hif-slot-add-bar");
      if (addBar && addBar.dataset.dirty === "1") {
        const key = addBar.querySelector(".hif-slot-draft-key")?.value || "";
        const val = Math.max(0, Number(addBar.querySelector(".hif-slot-draft-val")?.value || 0));
        const note = (addBar.querySelector(".hif-slot-draft-note")?.value || "").trim();
        const chosen = passiveTemplates.find((pt) => pt.key === key);
        if (key && ((key === "custom_note" && note.length > 0) || (key !== "custom_note" && val > 0))) {
          slotObj.passives.push({
            key,
            label: chosen?.label || key,
            value: key === "custom_note" ? 0 : val,
            note,
          });
          hasUnsavedChanges = true;
        }
      }
    }
  });

  return { nextSlots, hasUnsavedChanges };
}

/* ==================== Tab 4: Produce Card & Drink Preferences ==================== */
let _prefSaveQueue = Promise.resolve();
let _lastPrefFilteredCards = [];

function getFilteredPrefCards(planCards, st) {
  const prefs = st?.preferences || {};
  const priSet = new Set((prefs.priority_card_ids || []).map(Number));
  const keepSet = new Set((prefs.keep_card_ids || []).map(Number));
  const avoidSet = new Set((prefs.avoid_card_ids || []).map(Number));

  const q = (byId("hif-pref-card-q")?.value || "").trim().toLowerCase();
  const typeF = byId("hif-pref-card-type")?.value || "all";
  const catF = byId("hif-pref-card-cat")?.value || "usable";
  const rarityF = byId("hif-pref-card-rarity")?.value || "all";
  const upF = byId("hif-pref-card-up")?.value || "all";
  const effF = byId("hif-pref-card-effect")?.value || "all";
  const statusF = byId("hif-pref-card-status")?.value || "all";
  const sortF = byId("hif-pref-card-sort")?.value || "rarity_desc";

  const rawMatches = (planCards || [])
    .filter((c) => (c.type || "") !== "trouble")
    .filter((c) => {
      if (typeF !== "all" && (c.type || "").toLowerCase() !== typeF) return false;
      if (catF === "usable") {
        if (!q && !isUsableForSelectedIdol(c, st.selected_p_idol_id)) return false;
        if (q && !isUsableForSelectedIdol(c, st.selected_p_idol_id)) return false;
      } else if (catF !== "all" && (c.category || "") !== catF) {
        return false;
      }
      if (rarityF !== "all" && (c.rarity || "").toUpperCase() !== rarityF.toUpperCase()) return false;
      if (upF === "upgraded" && !c.upgraded) return false;
      if (upF === "base" && c.upgraded) return false;
      if (effF !== "all") {
        const effHay = `${c.name || ""} ${c.description || ""} ${c.dsl_actions || ""} ${c.dsl_effects || ""}`;
        if (!effHay.includes(effF)) return false;
      }
      const cidNum = Number(c.id);
      if (statusF === "unassigned" && (priSet.has(cidNum) || keepSet.has(cidNum) || avoidSet.has(cidNum))) return false;
      if (statusF === "priority" && !priSet.has(cidNum)) return false;
      if (statusF === "keep" && !keepSet.has(cidNum)) return false;
      if (statusF === "avoid" && !avoidSet.has(cidNum)) return false;
      if (q) {
        const hay = `${c.id} ${c.name} ${c.description || ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

  return sortCardPool(rawMatches, sortF);
}

async function renderPrefsTab() {
  const st = hifState.state;
  if (!st) return;
  const plan = (st.selected_plan || "logic").toLowerCase();
  const prefs = st.preferences || {};

  byId("hif-pref-spend-all").checked = Boolean(prefs.spend_all_final_shop ?? true);
  byId("hif-pref-reroll-m").checked = Boolean(prefs.reroll_early_non_exhaust_mental ?? true);

  const renderPillBucket = (listId, cntId, items, fieldName) => {
    const box = byId(listId);
    byId(cntId).textContent = String(items.length);
    box.replaceChildren();
    if (items.length === 0) {
      box.appendChild(makeEl("span", "muted-note", "（空）"));
      return;
    }
    for (const item of items) {
      const pill = makeEl("span", "hif-pref-pill");
      pill.appendChild(makeEl("span", "", `${item.name} (#${item.card_id})`));
      const xBtn = makeEl("button", "hif-pill-x", "✕");
      xBtn.type = "button";
      xBtn.addEventListener("click", () => {
        mutatePrefs((curPrefs) => ({
          [fieldName]: (curPrefs[fieldName] || []).filter((id) => Number(id) !== Number(item.card_id)),
        }));
      });
      pill.appendChild(xBtn);
      box.appendChild(pill);
    }
  };

  renderPillBucket("hif-pref-priority-list", "hif-pref-pri-cnt", prefs.priority_cards || [], "priority_card_ids");
  renderPillBucket("hif-pref-keep-list", "hif-pref-keep-cnt", prefs.keep_cards || [], "keep_card_ids");
  renderPillBucket("hif-pref-avoid-list", "hif-pref-avoid-cnt", prefs.avoid_cards || [], "avoid_card_ids");

  // Card score override weights
  const planCards = await ensurePlanCards(plan);
  const wBox = byId("hif-pref-card-weights-list");
  const cardOverrides = prefs.card_score_overrides || {};
  const wEntries = Object.entries(cardOverrides);
  const wCntEl = byId("hif-pref-weight-cnt");
  if (wCntEl) wCntEl.textContent = String(wEntries.length);
  wBox.replaceChildren();
  if (wEntries.length === 0) {
    wBox.appendChild(makeEl("span", "muted-note", "（暂无自定义单卡数值权重，可在下方搜索卡片设定）"));
  } else {
    for (const [cidStr, deltaVal] of wEntries) {
      const meta = findCardMeta(Number(cidStr));
      const row = makeEl("div", "hif-weight-row");
      row.appendChild(makeEl("span", "", `${meta?.name || `Card#${cidStr}`} (#${cidStr})`));
      const inp = document.createElement("input");
      inp.type = "number";
      inp.min = "-8000";
      inp.max = "8000";
      inp.step = "100";
      inp.value = String(deltaVal);
      inp.addEventListener("change", () => {
        mutatePrefs((curPrefs) => {
          const nextMap = { ...(curPrefs.card_score_overrides || {}), [cidStr]: Number(inp.value || 0) };
          if (nextMap[cidStr] === 0) delete nextMap[cidStr];
          return { card_score_overrides: nextMap };
        });
      });
      const delBtn = makeEl("button", "btn-secondary btn-compact", "✕");
      delBtn.type = "button";
      delBtn.addEventListener("click", () => {
        mutatePrefs((curPrefs) => {
          const nextMap = { ...(curPrefs.card_score_overrides || {}) };
          delete nextMap[cidStr];
          return { card_score_overrides: nextMap };
        });
      });
      row.append(inp, delBtn);
      wBox.appendChild(row);
    }
  }

  // Search results for preference card actions (multi-filter & unlimited!)
  const priSet = new Set((prefs.priority_card_ids || []).map(Number));
  const keepSet = new Set((prefs.keep_card_ids || []).map(Number));
  const avoidSet = new Set((prefs.avoid_card_ids || []).map(Number));
  const resBox = byId("hif-pref-card-results");
  resBox.replaceChildren();
  const cardMatches = getFilteredPrefCards(planCards, st);
  _lastPrefFilteredCards = cardMatches;
  const cntEl = byId("hif-pref-card-count");
  if (cntEl) cntEl.textContent = `匹配 ${cardMatches.length} 张`;

  for (const c of cardMatches) {
    const cidNum = Number(c.id);
    const row = makeEl("div", "hif-pref-search-row");
    const left = makeEl("div", "hif-deck-left");
    if (c.icon_url) {
      const img = document.createElement("img");
      img.src = c.icon_url;
      img.alt = c.name;
      img.width = 32;
      img.height = 32;
      img.className = "hif-mini-icon";
      left.appendChild(img);
    }
    const inf = makeEl("div", "hif-deck-info");
    const tags = [];
    if (priSet.has(cidNum)) tags.push("★高优");
    if (keepSet.has(cidNum)) tags.push("🔒锁卡");
    if (avoidSet.has(cidNum)) tags.push("🚫禁选");
    const tagStr = tags.length > 0 ? ` [${tags.join("/")}]` : "";
    inf.appendChild(makeEl("strong", "", `${c.name} (#${c.id})${tagStr}`));
    inf.appendChild(makeEl("span", "muted-note", `${(c.type || "").toUpperCase()} · ${c.rarity} · ${(c.description || "").slice(0, 42)}`));
    left.appendChild(inf);

    const btns = makeEl("div", "hif-deck-actions");
    const mkAdd = (label, listKey, cls = "") => {
      const b = makeEl("button", `btn-secondary btn-compact ${cls}`.trim(), label);
      b.type = "button";
      b.addEventListener("click", () => {
        mutatePrefs((curPrefs) => {
          const cur = new Set((curPrefs[listKey] || []).map(Number));
          cur.add(cidNum);
          const patch = { [listKey]: [...cur] };
          if (listKey === "priority_card_ids") {
            patch.avoid_card_ids = (curPrefs.avoid_card_ids || []).filter((x) => Number(x) !== cidNum);
          } else if (listKey === "avoid_card_ids") {
            patch.priority_card_ids = (curPrefs.priority_card_ids || []).filter((x) => Number(x) !== cidNum);
            patch.keep_card_ids = (curPrefs.keep_card_ids || []).filter((x) => Number(x) !== cidNum);
          }
          return patch;
        });
      });
      return b;
    };
    const bWeight = makeEl("button", "btn-secondary btn-compact", "⚖+500");
    bWeight.type = "button";
    bWeight.addEventListener("click", () => {
      mutatePrefs((curPrefs) => {
        const curMap = curPrefs.card_score_overrides || {};
        return { card_score_overrides: { ...curMap, [String(c.id)]: Number(curMap[String(c.id)] || 0) + 500 } };
      });
    });
    btns.append(
      mkAdd("★高优", "priority_card_ids"),
      mkAdd("🔒锁卡", "keep_card_ids"),
      mkAdd("🚫禁选", "avoid_card_ids", "hif-btn-danger"),
      bWeight,
    );
    row.append(left, btns);
    resBox.appendChild(row);
  }

  // Drinks grid
  const drinks = await ensurePlanDrinks(plan);
  const drinksGrid = byId("hif-pref-drinks-grid");
  drinksGrid.replaceChildren();
  const prefDrinkSet = new Set((prefs.preferred_drink_ids || []).map(Number));
  const banDrinkSet = new Set((prefs.banned_drink_ids || []).map(Number));
  const drinkOverrides = prefs.drink_score_overrides || {};

  for (const d of drinks) {
    const did = Number(d.id);
    const card = makeEl("div", "hif-drink-pref-card");
    const isPref = prefDrinkSet.has(did);
    const isBan = banDrinkSet.has(did);
    if (isPref) card.classList.add("is-preferred");
    if (isBan) card.classList.add("is-banned");

    const top = makeEl("div", "hif-deck-left");
    if (d.icon_url) {
      const img = document.createElement("img");
      img.src = d.icon_url;
      img.alt = d.name;
      img.width = 40;
      img.height = 40;
      img.className = "hif-mini-icon";
      top.appendChild(img);
    }
    const inf = makeEl("div", "hif-deck-info");
    inf.appendChild(makeEl("strong", "", `#${did} ${d.name} (${d.rarity})`));
    inf.appendChild(makeEl("span", "muted-note", (d.description || "").replace(/\n/g, " / ")));
    top.appendChild(inf);

    const seg = makeEl("div", "hif-drink-seg");
    const mkSegBtn = (lbl, modeKey, active) => {
      const b = makeEl("button", `hif-seg-btn${active ? " is-active" : ""} hif-seg-${modeKey}`, lbl);
      b.type = "button";
      b.addEventListener("click", async () => {
        const nextPref = new Set(prefDrinkSet);
        const nextBan = new Set(banDrinkSet);
        nextPref.delete(did);
        nextBan.delete(did);
        if (modeKey === "prefer") nextPref.add(did);
        if (modeKey === "ban") nextBan.add(did);
        await savePrefs({
          preferred_drink_ids: [...nextPref],
          banned_drink_ids: [...nextBan],
        });
      });
      return b;
    };
    seg.append(
      mkSegBtn("⭐ 优先", "prefer", isPref),
      mkSegBtn("⚪ 默认", "neutral", !isPref && !isBan),
      mkSegBtn("🚫 禁用并丢弃", "ban", isBan),
    );

    const wInp = document.createElement("input");
    wInp.type = "number";
    wInp.min = "-8000";
    wInp.max = "8000";
    wInp.step = "100";
    wInp.className = "hif-drink-weight-input";
    wInp.placeholder = "微调±";
    wInp.value = drinkOverrides[String(did)] ? String(drinkOverrides[String(did)]) : "";
    wInp.title = "单瓶饮料额外评分加减值 (-8000 ~ +8000)";
    wInp.addEventListener("change", async () => {
      const nextMap = { ...drinkOverrides };
      const v = Number(wInp.value || 0);
      if (v === 0) delete nextMap[String(did)];
      else nextMap[String(did)] = v;
      await savePrefs({ drink_score_overrides: nextMap });
    });

    const bot = makeEl("div", "hif-drink-bottom");
    bot.append(seg, wInp);
    card.append(top, bot);
    drinksGrid.appendChild(card);
  }
}

function mutatePrefs(updaterFn) {
  _prefSaveQueue = _prefSaveQueue
    .then(async () => {
      const cur = hifState.state?.preferences || {};
      const partial = updaterFn(cur) || {};
      await savePrefs(partial);
    })
    .catch((err) => {
      console.error("mutatePrefs failed:", err);
    });
  return _prefSaveQueue;
}

async function savePrefs(partial = {}) {
  const cur = hifState.state?.preferences || {};
  const payload = {
    priority_card_ids: partial.priority_card_ids ?? cur.priority_card_ids ?? [],
    avoid_card_ids: partial.avoid_card_ids ?? cur.avoid_card_ids ?? [],
    keep_card_ids: partial.keep_card_ids ?? cur.keep_card_ids ?? [],
    card_score_overrides: partial.card_score_overrides ?? cur.card_score_overrides ?? {},
    preferred_drink_ids: partial.preferred_drink_ids ?? cur.preferred_drink_ids ?? [],
    banned_drink_ids: partial.banned_drink_ids ?? cur.banned_drink_ids ?? [19],
    drink_score_overrides: partial.drink_score_overrides ?? cur.drink_score_overrides ?? {},
    spend_all_final_shop: partial.spend_all_final_shop ?? byId("hif-pref-spend-all")?.checked ?? true,
    reroll_early_non_exhaust_mental:
      partial.reroll_early_non_exhaust_mental ?? byId("hif-pref-reroll-m")?.checked ?? true,
  };
  const res = await hifState.ctx.apiPost("/api/hif/preferences", payload);
  if (res?.state) {
    hifState.state = res.state;
    renderWorkbenchHeader();
    if (hifState.activeTab === "prefs") {
      await renderPrefsTab();
    }
  } else {
    await refreshHifState();
  }
}

async function saveRunConfig() {
  const checkedMode = document.querySelector('input[name="hif-loadout-mode"]:checked')?.value || "hybrid";
  const payload = {
    loadout_mode: checkedMode,
    adb_host: byId("hif-adb-host")?.value?.trim() || "127.0.0.1",
    adb_port: Number(byId("hif-adb-port")?.value || 16384),
    scenario_id: byId("hif-scenario-id")?.value || "hif_honsen",
    max_steps: Number(byId("hif-max-steps")?.value || 450),
    delay: Number(byId("hif-delay")?.value ?? 0.5),
    ppo_episodes: Number(byId("hif-ppo-episodes")?.value || 2000),
    ismcts_sims: Number(byId("hif-ismcts-sims")?.value || 64),
    stop_on_home_return: Boolean(byId("hif-stop-home")?.checked ?? true),
    library_out: byId("hif-library-out")?.value?.trim() || "",
    csv_out: byId("hif-csv-out")?.value?.trim() || "",
    debug_dir: byId("hif-debug-dir")?.value?.trim() || "",
  };
  await hifState.ctx.apiPost("/api/hif/config", payload);
  await refreshHifState();
}

/* ==================== Event Bindings & Public Entry ==================== */
function bindHifEvents() {
  // Stage 1: Profession cards
  for (const card of document.querySelectorAll(".hif-prof-card")) {
    card.addEventListener("click", async () => {
      const plan = card.dataset.plan;
      await hifState.ctx.apiPost("/api/hif/select-idol", { plan, p_idol_id: null, auto_seed_signature: false });
      await refreshHifState();
    });
  }

  for (const id of ["hif-roster-search", "hif-roster-rarity", "hif-roster-guide-only"]) {
    const el = byId(id);
    if (el) el.addEventListener(id === "hif-roster-search" ? "input" : "change", renderRosterStage);
  }

  byId("btn-hif-skip-idol")?.addEventListener("click", () => {
    setStage("workbench");
    setActiveTab("run");
  });

  byId("btn-hif-back-roster")?.addEventListener("click", () => {
    setStage("roster");
    renderRosterStage();
  });

  const handleImportByFilename = async (selectId) => {
    const filename = byId(selectId)?.value || "";
    if (!filename) {
      hifState.ctx.showToast("请先选择一个本地 produce_library*.json 文件");
      return;
    }
    try {
      await hifState.ctx.apiPost("/api/hif/import-library", { filename });
      await refreshHifState();
      setStage("workbench");
      hifState.ctx.showToast(`已导入配置 ${filename}`);
    } catch (err) {
      hifState.ctx.showToast(`导入失败: ${err.message}`);
    }
  };
  byId("btn-hif-import-roster")?.addEventListener("click", () => handleImportByFilename("hif-import-select"));
  byId("btn-hif-loadout-import")?.addEventListener("click", () => handleImportByFilename("hif-loadout-import-select"));

  // Helper: read currently checked memory-hook form into a merged card_memory_hooks dict
  const buildMergedHooksWithForm = () => {
    const nextHooks = { ...(hifState.state?.card_memory_hooks || {}) };
    const cid = byId("hif-hook-card-select")?.value;
    if (!cid) return { nextHooks, formChanged: false };
    const extraOn = Boolean(byId("hif-hook-extra-use")?.checked);
    const drawOn = Boolean(byId("hif-hook-draw")?.checked);
    const hookEntry = {
      force_initial_hand: Boolean(byId("hif-hook-force-init")?.checked),
      exiles_trouble_on_play: Boolean(byId("hif-hook-exile")?.checked),
      free_extra_use_first_play: extraOn,
      extra_use_limit: extraOn ? Number(byId("hif-hook-extra-limit")?.value || 1) : 0,
      draw_on_play: extraOn && drawOn ? 1 : 0,
    };
    const anyChecked =
      hookEntry.force_initial_hand ||
      hookEntry.exiles_trouble_on_play ||
      hookEntry.free_extra_use_first_play ||
      hookEntry.draw_on_play > 0;
    const prev = nextHooks[String(cid)] || null;
    if (anyChecked) {
      nextHooks[String(cid)] = hookEntry;
      const changed = JSON.stringify(prev) !== JSON.stringify(hookEntry);
      return { nextHooks, formChanged: changed };
    }
    return { nextHooks, formChanged: false };
  };

  // File upload JSON import & export
  byId("btn-hif-export-json")?.addEventListener("click", async () => {
    try {
      const { nextHooks, formChanged } = buildMergedHooksWithForm();
      const { nextSlots, hasUnsavedChanges } = collectMemorySlotsFromDom({ commitDirtyDrafts: true });
      if (formChanged || hasUnsavedChanges) {
        await hifState.ctx.apiPost("/api/hif/memories", {
          ...(formChanged ? { card_memory_hooks: nextHooks } : {}),
          ...(hasUnsavedChanges ? { memory_slots: nextSlots } : {}),
        });
        await refreshHifState({ forceSlotsRender: true });
      }
      const data = await hifState.ctx.apiGet("/api/hif/export-library");
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `hif_config_${data.plan || "logic"}_${data.p_idol_id || "custom"}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      hifState.ctx.showToast("已导出 H.I.F 配置 JSON");
    } catch (err) {
      hifState.ctx.showToast(`导出失败: ${err.message}`);
    }
  });

  byId("hif-import-file-input")?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      await hifState.ctx.apiPost("/api/hif/import-library", { data });
      await refreshHifState({ forceSlotsRender: true });
      setStage("workbench");
      hifState.ctx.showToast(`已从 ${file.name} 导入配置`);
    } catch (err) {
      hifState.ctx.showToast(`文件解析失败: ${err.message}`);
    } finally {
      e.target.value = "";
    }
  });

  // Sub-tabs
  for (const btn of document.querySelectorAll(".hif-subtab")) {
    btn.addEventListener("click", () => setActiveTab(btn.dataset.hifTab));
  }

  // Launch / Stop buttons (sticky + Tab 1)
  const toggleRun = async () => {
    const running = Boolean(hifState.state?.running);
    try {
      if (running) {
        await hifState.ctx.apiPost("/api/hif/stop");
        hifState.ctx.showToast("已请求停止 H.I.F 自动化");
      } else {
        const { nextHooks, formChanged } = buildMergedHooksWithForm();
        const { nextSlots, hasUnsavedChanges } = collectMemorySlotsFromDom({ commitDirtyDrafts: true });
        if (formChanged || hasUnsavedChanges) {
          await hifState.ctx.apiPost("/api/hif/memories", {
            ...(formChanged ? { card_memory_hooks: nextHooks } : {}),
            ...(hasUnsavedChanges ? { memory_slots: nextSlots } : {}),
          });
        }
        await saveRunConfig();
        await hifState.ctx.apiPost("/api/hif/start");
        hifState.ctx.showToast("H.I.F 自动化任务已启动！");
      }
      await refreshHifState({ forceSlotsRender: true });
    } catch (err) {
      hifState.ctx.showToast(`操作失败: ${err.message}`);
    }
  };
  byId("btn-hif-launch")?.addEventListener("click", toggleRun);
  byId("btn-hif-launch-main")?.addEventListener("click", toggleRun);

  for (const r of document.querySelectorAll('input[name="hif-loadout-mode"]')) {
    r.addEventListener("change", async () => {
      try {
        await saveRunConfig();
        hifState.ctx.showToast(`录入模式已切换为 ${LOADOUT_MODE_SHORT[r.value] || r.value}`);
      } catch (err) {
        hifState.ctx.showToast(`切换失败: ${err.message}`);
      }
    });
  }

  byId("btn-hif-save-config")?.addEventListener("click", async () => {
    try {
      await saveRunConfig();
      hifState.ctx.showToast("运行配置已保存");
    } catch (err) {
      hifState.ctx.showToast(`保存失败: ${err.message}`);
    }
  });

  byId("btn-hif-preview-adb")?.addEventListener("click", async () => {
    const btn = byId("btn-hif-preview-adb");
    if (btn) btn.disabled = true;
    try {
      await saveRunConfig();
      const res = await hifState.ctx.apiPost("/api/hif/preview");
      hifState.ctx.showToast(res.message || "已刷新实机截图");
    } catch (err) {
      hifState.ctx.showToast(`预览抓取失败: ${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
      await refreshHifState();
    }
  });

  byId("btn-hif-clear-logs")?.addEventListener("click", async () => {
    await hifState.ctx.apiPost("/api/hif/logs/clear");
    await refreshHifState();
  });

  // Loadout tab controls
  byId("btn-hif-seed-sig")?.addEventListener("click", async () => {
    await saveLoadout({ seed_signature: true });
    hifState.ctx.showToast("已重新载入角色固有卡与固有道具");
  });
  byId("btn-hif-clear-deck")?.addEventListener("click", async () => {
    await saveLoadout({ deck_ids: [] });
    hifState.ctx.showToast("已清空卡组");
  });
  byId("btn-hif-save-loadout")?.addEventListener("click", async () => {
    await saveLoadout();
    hifState.ctx.showToast("已保存技能卡、三维与道具配置");
  });

  byId("btn-hif-add-custom-pitem")?.addEventListener("click", async () => {
    const nm = byId("hif-new-pitem-name")?.value?.trim();
    if (!nm) {
      hifState.ctx.showToast("请输入 P 道具名称");
      return;
    }
    const newItem = {
      name: nm,
      canonical_name: nm,
      mode: byId("hif-new-pitem-mode")?.value || "stage",
      source_type: byId("hif-new-pitem-source")?.value || "pIdol",
      rarity: "SSR",
      description: byId("hif-new-pitem-desc")?.value?.trim() || "",
    };
    const next = [...(hifState.state?.p_items || []), newItem];
    byId("hif-new-pitem-name").value = "";
    byId("hif-new-pitem-desc").value = "";
    await saveLoadout({ p_items: next });
    hifState.ctx.showToast(`已添加 P 道具：${nm}`);
  });

  for (const id of [
    "hif-card-picker-q",
    "hif-card-picker-type",
    "hif-card-picker-cat",
    "hif-card-picker-rarity",
    "hif-card-picker-up",
    "hif-card-picker-sort",
  ]) {
    const el = byId(id);
    if (el) el.addEventListener(id === "hif-card-picker-q" ? "input" : "change", renderCardPicker);
  }

  byId("btn-hif-cust-save")?.addEventListener("click", async () => {
    const cid = hifState.editingCustCardId;
    if (!cid) return;
    const levels = {};
    document.querySelectorAll("#hif-cust-official-wrap .hif-cust-level-select").forEach((sel) => {
      const lv = Number(sel.value || 0);
      if (lv > 0 && sel.dataset.custId) {
        levels[String(sel.dataset.custId)] = lv;
      }
    });
    const modifiers = {
      force_initial_hand: Boolean(byId("hif-cust-mod-fih")?.checked),
      remove_limit: Boolean(byId("hif-cust-mod-nolimit")?.checked),
      cost_reduction: Number(byId("hif-cust-mod-cost")?.value || 0),
      score_delta: Number(byId("hif-cust-mod-score")?.value || 0),
      good_impression_delta: Number(byId("hif-cust-mod-gi")?.value || 0),
      motivation_delta: Number(byId("hif-cust-mod-mot")?.value || 0),
      genki_delta: Number(byId("hif-cust-mod-genki")?.value || 0),
      good_condition_delta: Number(byId("hif-cust-mod-gc")?.value || 0),
      concentration_delta: Number(byId("hif-cust-mod-conc")?.value || 0),
      full_power_charge_delta: Number(byId("hif-cust-mod-fp")?.value || 0),
    };
    const nextCusts = { ...(hifState.state?.card_customizations || {}) };
    nextCusts[String(cid)] = { levels, modifiers };
    await saveLoadout({ card_customizations: nextCusts });
    byId("hif-card-cust-dialog")?.close();
    hifState.ctx.showToast(`已保存 #${cid} 的卡牌自定义强化`);
  });

  byId("btn-hif-cust-clear")?.addEventListener("click", async () => {
    const cid = hifState.editingCustCardId;
    if (!cid) return;
    const nextCusts = { ...(hifState.state?.card_customizations || {}) };
    delete nextCusts[String(cid)];
    await saveLoadout({ card_customizations: nextCusts });
    byId("hif-card-cust-dialog")?.close();
    hifState.ctx.showToast(`已清除 #${cid} 的卡牌强化配置`);
  });

  // Memory tab controls
  byId("hif-hook-card-select")?.addEventListener("change", (e) => {
    loadHookFormForCard(e.target.value);
  });

  byId("btn-hif-apply-hook")?.addEventListener("click", async () => {
    const cid = byId("hif-hook-card-select")?.value;
    if (!cid) {
      hifState.ctx.showToast("请选择目标技能卡");
      return;
    }
    const extraOn = Boolean(byId("hif-hook-extra-use")?.checked);
    const drawOn = Boolean(byId("hif-hook-draw")?.checked);
    const hookEntry = {
      force_initial_hand: Boolean(byId("hif-hook-force-init")?.checked),
      exiles_trouble_on_play: Boolean(byId("hif-hook-exile")?.checked),
      free_extra_use_first_play: extraOn,
      extra_use_limit: extraOn ? Number(byId("hif-hook-extra-limit")?.value || 1) : 0,
      draw_on_play: extraOn && drawOn ? 1 : 0,
    };
    const nextHooks = { ...(hifState.state?.card_memory_hooks || {}) };
    nextHooks[String(cid)] = hookEntry;
    await hifState.ctx.apiPost("/api/hif/memories", { card_memory_hooks: nextHooks });
    await refreshHifState();
    hifState.ctx.showToast("已添加/更新技能卡回忆效果");
  });

  const saveMemorySlotsFromDom = async ({ includeHooks = false } = {}) => {
    const { nextSlots } = collectMemorySlotsFromDom({ commitDirtyDrafts: true });
    const payload = { memory_slots: nextSlots };
    if (includeHooks) {
      const { nextHooks } = buildMergedHooksWithForm();
      payload.card_memory_hooks = nextHooks;
    }
    await hifState.ctx.apiPost("/api/hif/memories", payload);
    await refreshHifState({ forceSlotsRender: true });
    const passiveCnt = (hifState.state?.memory_passives || []).length;
    const hookCnt = Object.keys(hifState.state?.card_memory_hooks || {}).length;
    return { passiveCnt, hookCnt };
  };

  byId("btn-hif-save-memories")?.addEventListener("click", async () => {
    const { hookCnt, passiveCnt } = await saveMemorySlotsFromDom({ includeHooks: true });
    hifState.ctx.showToast(`已保存回忆配置（${hookCnt} 条技能回忆效果，${passiveCnt} 条开局词条）`);
  });

  byId("btn-hif-save-slots")?.addEventListener("click", async () => {
    const { passiveCnt } = await saveMemorySlotsFromDom({ includeHooks: false });
    hifState.ctx.showToast(`已保存 4 枚回忆槽位（当前生效 ${passiveCnt} 条开局词条）`);
  });

  // Preferences tab controls & multi-dimensional card filters
  byId("hif-pref-card-q")?.addEventListener("input", renderPrefsTab);
  for (const fid of [
    "hif-pref-card-type",
    "hif-pref-card-cat",
    "hif-pref-card-rarity",
    "hif-pref-card-up",
    "hif-pref-card-effect",
    "hif-pref-card-status",
    "hif-pref-card-sort",
  ]) {
    byId(fid)?.addEventListener("change", renderPrefsTab);
  }
  byId("btn-hif-pref-reset-filter")?.addEventListener("click", () => {
    if (byId("hif-pref-card-q")) byId("hif-pref-card-q").value = "";
    if (byId("hif-pref-card-type")) byId("hif-pref-card-type").value = "all";
    if (byId("hif-pref-card-cat")) byId("hif-pref-card-cat").value = "usable";
    if (byId("hif-pref-card-rarity")) byId("hif-pref-card-rarity").value = "all";
    if (byId("hif-pref-card-up")) byId("hif-pref-card-up").value = "all";
    if (byId("hif-pref-card-effect")) byId("hif-pref-card-effect").value = "all";
    if (byId("hif-pref-card-status")) byId("hif-pref-card-status").value = "all";
    if (byId("hif-pref-card-sort")) byId("hif-pref-card-sort").value = "rarity_desc";
    renderPrefsTab();
  });
  byId("btn-hif-pref-bulk-avoid")?.addEventListener("click", async () => {
    const idsToAdd = (_lastPrefFilteredCards || []).map((c) => Number(c.id)).filter((n) => n > 0);
    if (idsToAdd.length === 0) {
      hifState.ctx.showToast("当前筛选列表为空");
      return;
    }
    if (idsToAdd.length > 30 && !window.confirm(`确认将当前筛选出的全部 ${idsToAdd.length} 张卡牌一次性加入「避坑禁选卡」吗？`)) {
      return;
    }
    const addSet = new Set(idsToAdd);
    await mutatePrefs((curPrefs) => {
      const nextAvoid = new Set((curPrefs.avoid_card_ids || []).map(Number));
      for (const id of idsToAdd) nextAvoid.add(id);
      return {
        avoid_card_ids: [...nextAvoid],
        priority_card_ids: (curPrefs.priority_card_ids || []).filter((x) => !addSet.has(Number(x))),
        keep_card_ids: (curPrefs.keep_card_ids || []).filter((x) => !addSet.has(Number(x))),
      };
    });
    hifState.ctx.showToast(`已将 ${idsToAdd.length} 张筛选卡牌加入避坑禁选（当前共 ${(hifState.state?.preferences?.avoid_card_ids || []).length} 张）`);
  });
  byId("btn-hif-clear-priority")?.addEventListener("click", () => mutatePrefs(() => ({ priority_card_ids: [] })));
  byId("btn-hif-clear-keep")?.addEventListener("click", () => mutatePrefs(() => ({ keep_card_ids: [] })));
  byId("btn-hif-clear-avoid")?.addEventListener("click", () => mutatePrefs(() => ({ avoid_card_ids: [] })));
  byId("btn-hif-clear-weights")?.addEventListener("click", () => mutatePrefs(() => ({ card_score_overrides: {} })));
  byId("btn-hif-save-prefs")?.addEventListener("click", async () => {
    await savePrefs();
    hifState.ctx.showToast("已保存全部选卡与饮料偏好");
  });
  for (const id of ["hif-pref-spend-all", "hif-pref-reroll-m"]) {
    byId(id)?.addEventListener("change", async () => {
      await savePrefs();
    });
  }
  byId("btn-hif-reset-prefs")?.addEventListener("click", async () => {
    await hifState.ctx.apiPost("/api/hif/preferences", {
      priority_card_ids: [],
      avoid_card_ids: [],
      keep_card_ids: [],
      card_score_overrides: {},
      preferred_drink_ids: [],
      banned_drink_ids: [19],
      drink_score_overrides: {},
      spend_all_final_shop: true,
      reroll_early_non_exhaust_mental: true,
    });
    await refreshHifState();
    hifState.ctx.showToast("已恢复默认培育偏好（默认禁用并丢弃 #19 青汁）");
  });
}

export async function refreshHifState(options = {}) {
  if (!hifState.ctx) return;
  try {
    const st = await hifState.ctx.apiGet("/api/hif/state");
    hifState.state = st;
    if (st.selected_plan) {
      await Promise.all([ensurePlanCards(st.selected_plan), ensurePlanDrinks(st.selected_plan)]);
    }
    renderRosterStage();
    renderWorkbenchHeader();
    renderRunTab();
    if (hifState.activeTab === "loadout") await renderLoadoutTab();
    if (hifState.activeTab === "memory") await renderMemoryTab(options);
    if (hifState.activeTab === "prefs") await renderPrefsTab();
  } catch (err) {
    hifState.ctx.showToast(`加载 H.I.F 工作台失败: ${err.message}`);
  }
}

export async function initHifWorkbench(ctx) {
  hifState.ctx = ctx;
  bindHifEvents();
  try {
    hifState.catalog = await ctx.apiGet("/api/hif/catalog");
    await refreshHifState();
    if (hifState.state?.selected_p_idol_id || (hifState.state?.deck_ids || []).length > 0) {
      setStage("workbench");
    } else {
      setStage("roster");
    }
  } catch (err) {
    ctx.showToast(`初始化 H.I.F 目录失败: ${err.message}`);
  }
}
