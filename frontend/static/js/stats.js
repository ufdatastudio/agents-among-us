// Statistics page logic for Agents Among Us
// Displays RAW rows from frontend_stats.csv via backend API endpoints.

// =====================================================================
// Global state
// =====================================================================

let allData = [];          // All raw rows from frontend_stats.csv
let filteredData = [];     // Current filtered/sorted rows for All Data tab
let currentSortKey = null; // Current sort key for All Data table
let currentSortAsc = true;
let currentGameId = null;  // Selected game in Game List tab

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

    // Initial derived state
    filteredData = [...allData];

    updateSummary();
    populateFilters();
    applyFiltersAndSort();
    updateGameList();
  } catch (err) {
    console.error("Error loading stats:", err);
  }
}

// =====================================================================
// Summary box (simple counts)
// =====================================================================

function updateSummary() {
  const totalGames = new Set(allData.map((r) => r.game_id)).size;
  const totalAgents = allData.length;

  const honestWinsGames = new Set(
    allData
      .filter((r) => Number(r.won_game) === 1 && r.alignment === "H")
      .map((r) => r.game_id),
  ).size;

  const byzWinsGames = new Set(
    allData
      .filter((r) => Number(r.won_game) === 1 && r.alignment === "B")
      .map((r) => r.game_id),
  ).size;

  document.getElementById("total-games").textContent = totalGames;
  document.getElementById("total-agents").textContent = totalAgents;
  document.getElementById("honest-wins").textContent = honestWinsGames;
  document.getElementById("byz-wins").textContent = byzWinsGames;
}

// =====================================================================
// Filters and sorting for All Data tab
// =====================================================================

function populateFilters() {
  const modelFilter = document.getElementById("model-filter");
  if (!modelFilter) return;

  const models = Array.from(
    new Set(allData.map((r) => abbreviateModelName(r.model_name))),
  ).filter((m) => m);

  modelFilter.innerHTML = '<option value="">All Models</option>';
  models.sort().forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelFilter.appendChild(opt);
  });
}

// Apply search + filters + sort, then render table
function applyFiltersAndSort() {
  const searchInput = document.getElementById("search-input");
  const modelFilter = document.getElementById("model-filter");
  const roleFilter = document.getElementById("role-filter");

  const searchTerm = (searchInput?.value || "").toLowerCase();
  const modelValue = modelFilter?.value || "";
  const roleValue = roleFilter?.value || "";

  filteredData = allData.filter((row) => {
    // Model abbreviation
    const modelAbbrev = abbreviateModelName(row.model_name);

    // Role filter
    if (roleValue && row.alignment !== roleValue) return false;

    // Model filter
    if (modelValue && modelAbbrev !== modelValue) return false;

    // Search term across a few fields
    if (searchTerm) {
      const haystack = [
        row.game_id,
        abbreviateGameId(row.game_id),
        row.agent_name,
        abbreviateAgentName(row.agent_name),
        modelAbbrev,
        row.model_name,
        roleLabel(row.alignment),
      ]
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(searchTerm)) return false;
    }

    return true;
  });

  // Sort
  if (currentSortKey) {
    const key = currentSortKey;
    const asc = currentSortAsc;
    filteredData.sort((a, b) => compareAllDataRows(a, b, key, asc));
  }

  renderAllDataTable();
}

// Compare function for sorting All Data table
function compareAllDataRows(a, b, key, asc) {
  const dir = asc ? 1 : -1;

  const getVal = (row) => {
    switch (key) {
      case "game":
        return abbreviateGameId(row.game_id);
      case "agent":
        return Number(abbreviateAgentName(row.agent_name));
      case "model":
        return abbreviateModelName(row.model_name);
      case "role":
        return row.alignment || "";
      case "won":
        return Number(row.won_game);
      case "kills":
        return Number(row.eliminations);
      case "votes":
        return Number(row.votes_received);
      case "correct":
        return Number(row.correct_votes);
      case "rounds":
        return Number(row.rounds_survived);
      default:
        return 0;
    }
  };

  const va = getVal(a);
  const vb = getVal(b);

  if (va < vb) return -1 * dir;
  if (va > vb) return 1 * dir;
  return 0;
}

function renderAllDataTable() {
  const tbody = document.getElementById("all-data-tbody");
  const countSpan = document.getElementById("entry-count");
  if (!tbody || !countSpan) return;

  tbody.innerHTML = "";

  filteredData.forEach((row) => {
    const tr = document.createElement("tr");

    const gameCell = document.createElement("td");
    gameCell.textContent = abbreviateGameId(row.game_id);
    tr.appendChild(gameCell);

    const agentCell = document.createElement("td");
    agentCell.textContent = abbreviateAgentName(row.agent_name);
    tr.appendChild(agentCell);

    const modelCell = document.createElement("td");
    modelCell.textContent = abbreviateModelName(row.model_name);
    tr.appendChild(modelCell);

    const roleCell = document.createElement("td");
    roleCell.textContent = roleLabel(row.alignment);
    tr.appendChild(roleCell);

    const wonCell = document.createElement("td");
    wonCell.textContent = wonLabel(row.won_game);
    tr.appendChild(wonCell);

    const killsCell = document.createElement("td");
    if (row.alignment === "B") {
      killsCell.textContent = Number(row.eliminations) || 0;
    } else {
      killsCell.textContent = "n/a";
    }
    tr.appendChild(killsCell);

    const votesCell = document.createElement("td");
    votesCell.textContent = Number(row.votes_received) || 0;
    tr.appendChild(votesCell);

    const correctCell = document.createElement("td");
    if (row.alignment === "H") {
      correctCell.textContent = Number(row.correct_votes) || 0;
    } else {
      correctCell.textContent = "";
    }
    tr.appendChild(correctCell);

    const roundsCell = document.createElement("td");
    roundsCell.textContent = Number(row.rounds_survived) || 0;
    tr.appendChild(roundsCell);

    tbody.appendChild(tr);
  });

  countSpan.textContent = filteredData.length;
}

// Event handlers for search and filters
function handleFilterChange() {
  applyFiltersAndSort();
}

function sortAllData(key) {
  if (currentSortKey === key) {
    currentSortAsc = !currentSortAsc;
  } else {
    currentSortKey = key;
    currentSortAsc = true;
  }
  applyFiltersAndSort();
}

// =====================================================================
// Game list tab
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

      const idxCell = document.createElement("td");
      idxCell.textContent = abbreviateAgentName(row.agent_name);
      tr.appendChild(idxCell);

      const modelCell = document.createElement("td");
      modelCell.textContent = abbreviateModelName(row.model_name);
      tr.appendChild(modelCell);

      const roleCell = document.createElement("td");
      roleCell.textContent = row.alignment || "";
      tr.appendChild(roleCell);

      const wonCell = document.createElement("td");
      wonCell.textContent = Number(row.won_game) || 0;
      tr.appendChild(wonCell);

      const correctCell = document.createElement("td");
      correctCell.textContent = Number(row.correct_votes) || 0;
      tr.appendChild(correctCell);

      const incorrectCell = document.createElement("td");
      incorrectCell.textContent = Number(row.incorrect_votes) || 0;
      tr.appendChild(incorrectCell);

      const skippedCell = document.createElement("td");
      skippedCell.textContent = Number(row.skipped_votes) || 0;
      tr.appendChild(skippedCell);

      const emergCell = document.createElement("td");
      emergCell.textContent = Number(row.emergency_meetings) || 0;
      tr.appendChild(emergCell);

      const bodiesCell = document.createElement("td");
      bodiesCell.textContent = Number(row.bodies_reported) || 0;
      tr.appendChild(bodiesCell);

      const roundsCell = document.createElement("td");
      roundsCell.textContent = Number(row.rounds_survived) || 0;
      tr.appendChild(roundsCell);

      const killsCell = document.createElement("td");
      killsCell.textContent = Number(row.eliminations) || 0;
      tr.appendChild(killsCell);

      const elimCell = document.createElement("td");
      elimCell.textContent = Number(row.times_eliminated) || 0;
      tr.appendChild(elimCell);

      const ejectCell = document.createElement("td");
      ejectCell.textContent = Number(row.ejections) || 0;
      tr.appendChild(ejectCell);

      const movesCell = document.createElement("td");
      movesCell.textContent = Number(row.num_moves) || 0;
      tr.appendChild(movesCell);

      const votesCell = document.createElement("td");
      votesCell.textContent = Number(row.votes_received) || 0;
      tr.appendChild(votesCell);

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
  window.location.href = "/api/stats/export";
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
  });
  if (evt && evt.currentTarget) {
    evt.currentTarget.classList.add("active");
  }
}

// =====================================================================
// Initialization
// =====================================================================

window.addEventListener("DOMContentLoaded", () => {
  // Wire up filter controls
  const searchInput = document.getElementById("search-input");
  const modelFilter = document.getElementById("model-filter");
  const roleFilter = document.getElementById("role-filter");

  if (searchInput) {
    searchInput.addEventListener("input", handleFilterChange);
  }
  if (modelFilter) {
    modelFilter.addEventListener("change", handleFilterChange);
  }
  if (roleFilter) {
    roleFilter.addEventListener("change", handleFilterChange);
  }

  loadStats();
});

