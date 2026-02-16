// Statistics page logic for Agents Among Us
// Agent Summary: per-model aggregates (Honest / Byzantine). Single Game Data: game list + detail.

// =====================================================================
// Global state
// =====================================================================

let allData = [];          // All raw rows from frontend_stats.csv
let currentGameId = null;  // Selected game in Single Game Data tab

// All models in backend (config/model_composition.py). Order: heavyweight then small. Used for Agent Summary tables.
const ALL_MODELS = [
  "Qwen/Qwen3-Next-80B-A3B-Instruct",
  "arcee-ai/Arcee-Nova",
  "Qwen/Qwen2.5-72B-Instruct",
  "meta-llama/Llama-3.3-70B-Instruct",
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
  "zerofata/L3.3-GeneticLemonade-Final-v2-70B",
  "NousResearch/Hermes-4-70B",
  "unsloth/Apertus-70B-Instruct-2509-unsloth-bnb-4bit",
  "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled",
  "Nexusflow/Athene-V2-Chat",
  "MultiverseComputingCAI/HyperNova-60B",
  "meta-llama/Llama-3.2-3B-Instruct",
  "Qwen/Qwen2.5-1.5B-Instruct",
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
];
const DASH = "—";

// =====================================================================
// Utility helpers
// =====================================================================

// Abbreviate game id: "test_005" -> "005"
function abbreviateGameId(gameId) {
  if (!gameId) return "";
  const parts = gameId.split("_");
  return parts[parts.length - 1] || gameId;
}

// Agent_0 -> "0"
function abbreviateAgentName(agentName) {
  if (!agentName) return "";
  return agentName.replace("Agent_", "");
}

// Abbreviate model name for display
function abbreviateModelName(fullName) {
  if (!fullName) return "";
  const short = fullName.split("/").pop() || fullName;
  return short.replace(/-Instruct|-Chat-v1\.0/g, "");
}

// Role label
function roleLabel(alignment) {
  if (alignment === "H") return "Honest";
  if (alignment === "B") return "Byzantine";
  return alignment || "";
}

// Won label
function wonLabel(won) {
  return Number(won) === 1 ? "✓" : "✗";
}

// Parse timestamp string into Date (best-effort)
function parseTimestamp(ts) {
  if (!ts) return null;
  const d = new Date(ts);
  return isNaN(d.getTime()) ? null : d;
}

// Short date like "02/06"
function shortDate(ts) {
  const d = parseTimestamp(ts);
  if (!d) return "";
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${mm}/${dd}`;
}

// =====================================================================
// Data loading
// =====================================================================

async function loadStats() {
  try {
    const res = await fetch("/api/stats/all");
    if (!res.ok) {
      console.error("Failed to load stats:", res.status);
      return;
    }
    const data = await res.json();
    allData = Array.isArray(data) ? data : [];

    updateAgentSummary();
    updateGameList();
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

// =====================================================================
// Agent Summary (per-model aggregates, Honest / Byzantine tabs)
// =====================================================================

function showAgentSubtab(subtabName, evt) {
  document.querySelectorAll(".agent-summary-panel").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".agent-subtab").forEach((btn) => btn.classList.remove("active"));
  const panel = document.getElementById("agent-summary-" + subtabName);
  if (panel) panel.classList.add("active");
  if (evt && evt.currentTarget) evt.currentTarget.classList.add("active");
}

// Aggregate rows by model_name and alignment. Returns { "model_id": { games, wins, correct_votes, ... } } for given alignment.
function getAgentSummaryByModel(alignment) {
  const rows = allData.filter((r) => r.alignment === alignment);
  const byModel = {};

  rows.forEach((row) => {
    const id = (row.model_name || "").trim();
    if (!id) return;
    if (!byModel[id]) {
      byModel[id] = {
        games: 0,
        wins: 0,
        correct_votes: 0,
        incorrect_votes: 0,
        skipped_votes: 0,
        emergency_meetings: 0,
        bodies_reported: 0,
        rounds_survived: 0,
        votes_received: 0,
        eliminations: 0,
        times_eliminated: 0,
        ejections: 0,
      };
    }
    const m = byModel[id];
    m.games += 1;
    if (Number(row.won_game) === 1) m.wins += 1;
    m.correct_votes += Number(row.correct_votes) || 0;
    m.incorrect_votes += Number(row.incorrect_votes) || 0;
    m.skipped_votes += Number(row.skipped_votes) || 0;
    m.emergency_meetings += Number(row.emergency_meetings) || 0;
    m.bodies_reported += Number(row.bodies_reported) || 0;
    m.rounds_survived += Number(row.rounds_survived) || 0;
    m.votes_received += Number(row.votes_received) || 0;
    m.eliminations += Number(row.eliminations) || 0;
    m.times_eliminated += Number(row.times_eliminated) || 0;
    m.ejections += Number(row.ejections) || 0;
  });

  return byModel;
}

function getAgentSummarySearchAndSort() {
  const searchEl = document.getElementById("agent-summary-search");
  const sortEl = document.getElementById("agent-summary-sort");
  return {
    search: (searchEl && searchEl.value) ? searchEl.value.trim().toLowerCase() : "",
    sort: (sortEl && sortEl.value) ? sortEl.value : "a-z",
  };
}

function getFilteredAndSortedModelIds(honestAgg, byzAgg) {
  const { search, sort } = getAgentSummarySearchAndSort();
  let list = ALL_MODELS.map((modelId) => ({
    modelId,
    displayName: abbreviateModelName(modelId),
    hGames: (honestAgg[modelId] && honestAgg[modelId].games) || 0,
    bGames: (byzAgg[modelId] && byzAgg[modelId].games) || 0,
    paramIndex: ALL_MODELS.indexOf(modelId),
  }));

  if (search) {
    list = list.filter((item) => item.displayName.toLowerCase().includes(search));
  }

  const sortKey = sort;
  if (sortKey === "a-z") {
    list.sort((a, b) => a.displayName.localeCompare(b.displayName));
  } else if (sortKey === "z-a") {
    list.sort((a, b) => b.displayName.localeCompare(a.displayName));
  } else if (sortKey === "most-used") {
    list.sort((a, b) => {
      const useA = a.hGames + a.bGames;
      const useB = b.hGames + b.bGames;
      return useB - useA || a.displayName.localeCompare(b.displayName);
    });
  } else if (sortKey === "least-used") {
    list.sort((a, b) => {
      const useA = a.hGames + a.bGames;
      const useB = b.hGames + b.bGames;
      return useA - useB || a.displayName.localeCompare(b.displayName);
    });
  } else if (sortKey === "most-parameters") {
    list.sort((a, b) => a.paramIndex - b.paramIndex);
  } else if (sortKey === "least-parameters") {
    list.sort((a, b) => b.paramIndex - a.paramIndex);
  }

  return list.map((item) => item.modelId);
}

function updateAgentSummary() {
  const honestAgg = getAgentSummaryByModel("H");
  const byzAgg = getAgentSummaryByModel("B");
  const modelIds = getFilteredAndSortedModelIds(honestAgg, byzAgg);

  const honestTbody = document.getElementById("honest-tbody");
  const byzantineTbody = document.getElementById("byzantine-tbody");
  if (!honestTbody || !byzantineTbody) return;

  honestTbody.innerHTML = "";
  byzantineTbody.innerHTML = "";

  modelIds.forEach((modelId) => {
    const displayName = abbreviateModelName(modelId);
    const h = honestAgg[modelId];
    const b = byzAgg[modelId];

    const hr = document.createElement("tr");
    hr.appendChild(cell(displayName));
    hr.appendChild(cell(h ? h.games : DASH));
    hr.appendChild(cell(h ? h.wins : DASH));
    hr.appendChild(cell(h ? h.correct_votes : DASH));
    hr.appendChild(cell(h ? h.incorrect_votes : DASH));
    hr.appendChild(cell(h ? h.skipped_votes : DASH));
    hr.appendChild(cell(h ? h.emergency_meetings : DASH));
    hr.appendChild(cell(h ? h.bodies_reported : DASH));
    hr.appendChild(cell(h ? h.rounds_survived : DASH));
    hr.appendChild(cell(h ? h.votes_received : DASH));
    honestTbody.appendChild(hr);

    const br = document.createElement("tr");
    br.appendChild(cell(displayName));
    br.appendChild(cell(b ? b.games : DASH));
    br.appendChild(cell(b ? b.wins : DASH));
    br.appendChild(cell(b ? b.eliminations : DASH));
    br.appendChild(cell(b ? b.rounds_survived : DASH));
    br.appendChild(cell(b ? b.votes_received : DASH));
    br.appendChild(cell(b ? b.times_eliminated : DASH));
    br.appendChild(cell(b ? b.ejections : DASH));
    byzantineTbody.appendChild(br);
  });
}

function cell(val) {
  const td = document.createElement("td");
  td.textContent = val === undefined || val === null ? DASH : String(val);
  return td;
}

// =====================================================================
// Single Game Data tab (game list + detail)
// =====================================================================

function groupByGame() {
  const map = new Map();

  allData.forEach((row) => {
    const id = row.game_id;
    if (!map.has(id)) {
      map.set(id, []);
    }
    map.get(id).push(row);
  });

  return map;
}

function updateGameList() {
  const tbody = document.getElementById("game-tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const grouped = groupByGame();
  const games = [];

  grouped.forEach((rows, gameId) => {
    const comp = rows[0]?.composition || "";
    const agentsCount = rows.length;
    const ts = rows[0]?.timestamp || "";

    // Determine winner label from rows with won_game === 1
    let winnerLabelText = "";
    const winners = rows.filter((r) => Number(r.won_game) === 1);
    if (winners.length > 0) {
      const align = winners[0].alignment;
      if (align === "H") winnerLabelText = "Honest";
      else if (align === "B") winnerLabelText = "Byz";
    }

    games.push({
      game_id: gameId,
      composition: comp,
      agentsCount,
      winnerLabel: winnerLabelText,
      timestamp: ts,
    });
  });

  // Sort by timestamp desc (newest first)
  games.sort((a, b) => {
    const da = parseTimestamp(a.timestamp);
    const db = parseTimestamp(b.timestamp);
    if (!da && !db) return 0;
    if (!da) return 1;
    if (!db) return -1;
    return db - da;
  });

  games.forEach((g) => {
    const tr = document.createElement("tr");
    tr.dataset.gameId = g.game_id;

    const idCell = document.createElement("td");
    idCell.textContent = g.game_id;
    tr.appendChild(idCell);

    const compCell = document.createElement("td");
    compCell.textContent = g.composition;
    tr.appendChild(compCell);

    const agentsCell = document.createElement("td");
    agentsCell.textContent = g.agentsCount;
    tr.appendChild(agentsCell);

    const winnerCell = document.createElement("td");
    winnerCell.textContent = g.winnerLabel || "";
    tr.appendChild(winnerCell);

    const dateCell = document.createElement("td");
    dateCell.textContent = shortDate(g.timestamp);
    tr.appendChild(dateCell);

    tr.addEventListener("click", () => showGameDetails(g.game_id));

    tbody.appendChild(tr);
  });
}

function showGameDetails(gameId) {
  currentGameId = gameId;
  const details = document.getElementById("game-details");
  if (!details) return;

  const rows = allData.filter((r) => r.game_id === gameId);
  if (rows.length === 0) return;

  const compSpan = document.getElementById("detail-comp");
  const winnerSpan = document.getElementById("detail-winner");
  const dateSpan = document.getElementById("detail-date");
  const idSpan = document.getElementById("detail-game-id");

  idSpan.textContent = gameId;
  compSpan.textContent = rows[0].composition || "";

  const winners = rows.filter((r) => Number(r.won_game) === 1);
  let winnerLabelText = "";
  if (winners.length > 0) {
    const align = winners[0].alignment;
    winnerLabelText =
      align === "H" ? "Honest Agents" : align === "B" ? "Byzantine Agents" : "";
  }
  winnerSpan.textContent = winnerLabelText;
  dateSpan.textContent = rows[0].timestamp || "";

  // Render raw agent data
  const tbody = document.getElementById("detail-tbody");
  tbody.innerHTML = "";

  rows
    .slice()
    .sort((a, b) => {
      const na = Number(abbreviateAgentName(a.agent_name));
      const nb = Number(abbreviateAgentName(b.agent_name));
      return na - nb;
    })
    .forEach((row) => {
      const tr = document.createElement("tr");
      const agentNum = Number(abbreviateAgentName(row.agent_name));
      const displayIndex = Number.isNaN(agentNum) ? row.agent_name : agentNum + 1;

      tr.appendChild(cell(displayIndex));
      tr.appendChild(cell(abbreviateModelName(row.model_name)));
      tr.appendChild(cell(row.alignment || ""));
      tr.appendChild(cell(Number(row.correct_votes) || 0));
      tr.appendChild(cell(Number(row.incorrect_votes) || 0));
      tr.appendChild(cell(Number(row.skipped_votes) || 0));
      tr.appendChild(cell(Number(row.emergency_meetings) || 0));
      tr.appendChild(cell(Number(row.bodies_reported) || 0));
      tr.appendChild(cell(Number(row.rounds_survived) || 0));
      tr.appendChild(cell(Number(row.eliminations) || 0));
      tr.appendChild(cell(Number(row.times_eliminated) || 0));
      tr.appendChild(cell(Number(row.ejections) || 0));
      tr.appendChild(cell(Number(row.num_moves) || 0));
      tr.appendChild(cell(Number(row.votes_received) || 0));

      tbody.appendChild(tr);
    });

  details.style.display = "block";
}

function hideGameDetails() {
  const details = document.getElementById("game-details");
  if (details) details.style.display = "none";
  currentGameId = null;
}

// Trigger download of full CSV
function exportCSV() {
  // Check which tab is active
  const agentSummaryTab = document.querySelector('.stats-tab[data-tab="agent-summary"]');
  const isAgentSummaryActive = agentSummaryTab && agentSummaryTab.classList.contains('active');
  
  if (isAgentSummaryActive) {
    // Export aggregated model data (2 rows per model: Honest + Byzantine)
    exportModelAggregates();
  } else {
    // Export all games and all agents
    window.location.href = "/api/stats/export";
  }
}

// Generate and download model aggregate CSV
function exportModelAggregates() {
  const honestAgg = getAgentSummaryByModel("H");
  const byzAgg = getAgentSummaryByModel("B");
  
  // CSV header
  let csv = "Model,Role,Games,Wins,Correct Votes,Incorrect Votes,Skipped Votes,Emergency Meetings,Bodies Reported,Rounds Survived,Votes Received,Eliminations,Times Eliminated,Ejections\n";
  
  // For each model, add Honest row then Byzantine row
  ALL_MODELS.forEach((modelId) => {
    const displayName = abbreviateModelName(modelId);
    
    // Honest row
    if (honestAgg[modelId]) {
      const h = honestAgg[modelId];
      csv += `"${displayName}",Honest,${h.games},${h.wins},${h.correct_votes},${h.incorrect_votes},${h.skipped_votes},${h.emergency_meetings},${h.bodies_reported},${h.rounds_survived},${h.votes_received},${h.eliminations},${h.times_eliminated},${h.ejections}\n`;
    }
    
    // Byzantine row
    if (byzAgg[modelId]) {
      const b = byzAgg[modelId];
      csv += `"${displayName}",Byzantine,${b.games},${b.wins},${b.correct_votes},${b.incorrect_votes},${b.skipped_votes},${b.emergency_meetings},${b.bodies_reported},${b.rounds_survived},${b.votes_received},${b.eliminations},${b.times_eliminated},${b.ejections}\n`;
    }
  });
  
  // Download as blob
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const date = new Date().toISOString().split('T')[0];
  a.download = `model_aggregates_${date}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}

// Trigger download of a single game's CSV
function downloadGameCSV() {
  if (!currentGameId) return;
  const url = `/api/stats/export_game?game_id=${encodeURIComponent(currentGameId)}`;
  window.location.href = url;
}

// Clear all data (frontend_stats.csv) via backend
async function clearAllData() {
  if (!confirm("Delete ALL statistics? This cannot be undone!")) return;
  await fetch("/api/stats/clear", { method: "POST" });
  alert("All data cleared");
  await loadStats();
}

// Request backend to refresh from logs and append new games
async function refreshData() {
  try {
    const res = await fetch("/api/stats/refresh", { method: "POST" });
    if (!res.ok) {
      alert("Failed to refresh data.");
      return;
    }
    const result = await res.json();
    alert(`Added ${result.new_games || 0} new game(s).`);
    await loadStats();
  } catch (err) {
    console.error("Error refreshing data:", err);
    alert("Error refreshing data.");
  }
}

// =====================================================================
// Tab handling
// =====================================================================

function showTab(tabName, evt) {
  document
    .querySelectorAll(".stats-tab-content")
    .forEach((el) => el.classList.remove("active"));
  const content = document.getElementById(tabName);
  if (content) content.classList.add("active");

  document.querySelectorAll(".stats-tab").forEach((btn) => {
    btn.classList.remove("active");
    if (btn.getAttribute("data-tab") === tabName) btn.classList.add("active");
  });
  if (evt && evt.currentTarget) {
    evt.currentTarget.classList.add("active");
  }
}

// =====================================================================
// Initialization
// =====================================================================

window.addEventListener("DOMContentLoaded", () => {
  const searchEl = document.getElementById("agent-summary-search");
  const sortEl = document.getElementById("agent-summary-sort");
  if (searchEl) {
    searchEl.addEventListener("input", updateAgentSummary);
  }
  if (sortEl) {
    sortEl.addEventListener("change", updateAgentSummary);
  }
  loadStats();
});