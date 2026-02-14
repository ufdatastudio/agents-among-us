/**
 * Game Simulation Page - CORRECTED COORDINATES
 * Measured directly from actual map image
 * Press 'D' for debug overlay
 */

// ============================================================================
// ROOM COORDINATES - MANUALLY MEASURED FROM ACTUAL MAP
// ============================================================================

const ROOM_COORDINATES = {
    "Reactor": { x: 23.0, y: 53.6 },       // Clock (bottom-left circular) (FIXED)
    "UpperEngine": { x: 26.4, y: 23.6 },   // Air Cooling (top-left) (FIXED)
    "LowerEngine": { x: 32.2, y: 83.9 },   // Liquid Cooling (bottom-left) (FIXED)
    "Security": { x: 40.0, y: 42.6 },      // Logs (center-left) (FIXED)
    "MedBay": { x: 35.8, y: 11.5 },        // Diagnostics (top-center) (FIXED)
    "Electrical": { x: 39.9, y: 59.8 },    // Bus (center-left) (FIXED)
    "Cafeteria": { x: 49.5, y: 21.6 },     // CPU (center) (FIXED)
    "Admin": { x: 57.4, y: 53.1 },         // BIOS (center-right) (FIXED)
    "Storage": { x: 49.6, y: 82.0 },       // SSD (bottom-center) (FIXED)
    "Weapons": { x: 70.9, y: 16.1 },       // GPU (top-right) (FIXED)
    "O2": { x: 61.6, y: 33.9 },            // VRM (center-right) (FIXED)
    "Navigation": { x: 79.8, y: 43.5 },    // Network (far right) (FIXED) 
    "Shields": { x: 70.9, y: 67.3 },       // Firewall (right) (FIXED)
    "Communications": { x: 70.9, y: 90.2 } // IO (bottom-right) (FIXED)
};

// Map labels to show in debug (what's actually written on the map)
const ROOM_LABELS = {
    "Reactor": "Clock",
    "UpperEngine": "Air Cooling",
    "LowerEngine": "Liquid Cooling",
    "Security": "Logs",
    "MedBay": "Diagnostics",
    "Electrical": "Bus",
    "Cafeteria": "CPU",
    "Admin": "BIOS",
    "Storage": "SSD",
    "Weapons": "GPU",
    "O2": "VRM",
    "Navigation": "Network",
    "Shields": "Firewall",
    "Communications": "IO"
};

const ROOM_HOUSING = {
    "Reactor": { width: 6, height: 12 },
    "UpperEngine": { width: 6, height: 12 },
    "LowerEngine": { width: 6, height: 12 },
    "Security": { width: 5, height: 10 },
    "MedBay": { width: 5, height: 10 },
    "Electrical": { width: 5, height: 10 },
    "Cafeteria": { width: 8, height: 16 },
    "Admin": { width: 5, height: 10 },
    "Storage": { width: 10, height: 10 },
    "Weapons": { width: 10, height: 10 },
    "O2": { width: 5, height: 10 },
    "Navigation": { width: 6, height: 12 },
    "Shields": { width: 10, height: 10 },
    "Communications": { width: 6, height: 12 }
};

const ROOM_CONNECTIONS = [
    ["Reactor", "Security"], ["Reactor", "UpperEngine"], ["Reactor", "LowerEngine"],
    ["Security", "UpperEngine"], ["Security", "LowerEngine"],
    ["UpperEngine", "LowerEngine"], ["UpperEngine", "MedBay"], ["UpperEngine", "Cafeteria"],
    ["LowerEngine", "Electrical"], ["LowerEngine", "Storage"],
    ["MedBay", "Cafeteria"], ["Electrical", "Storage"],
    ["Cafeteria", "Admin"], ["Cafeteria", "Storage"], ["Cafeteria", "Weapons"],
    ["Admin", "Storage"],
    ["Weapons", "O2"], ["Weapons", "Navigation"], ["Weapons", "Shields"],
    ["O2", "Navigation"], ["O2", "Shields"],
    ["Navigation", "Shields"],
    ["Storage", "Shields"], ["Storage", "Communications"],
    ["Shields", "Communications"]
];

// ============================================================================
// DEBUG OVERLAY
// ============================================================================

let debugOverlayVisible = false;
let debugCanvas = null;

function createDebugCanvas() {
    if (debugCanvas) return debugCanvas;
    
    const mapContainer = document.querySelector(".map-container") || document.getElementById("map-container");
    if (!mapContainer) return null;
    
    const canvas = document.createElement("canvas");
    canvas.id = "debug-overlay";
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none";
    canvas.style.zIndex = "1000";
    canvas.style.display = "none";
    
    mapContainer.style.position = "relative";
    mapContainer.appendChild(canvas);
    
    const rect = mapContainer.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    debugCanvas = canvas;
    return canvas;
}

function drawDebugOverlay() {
    const canvas = createDebugCanvas();
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!debugOverlayVisible) return;
    
    // Map boundary
    ctx.strokeStyle = "#FF0000";
    ctx.lineWidth = 4;
    ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
    
    // Calculate room positions
    const roomPos = {};
    Object.keys(ROOM_COORDINATES).forEach(room => {
        const coord = ROOM_COORDINATES[room];
        roomPos[room] = {
            x: (coord.x / 100) * canvas.width,
            y: (coord.y / 100) * canvas.height
        };
    });
    
    // Draw pathways
    ctx.strokeStyle = "#00FFFF";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ROOM_CONNECTIONS.forEach(([r1, r2]) => {
        if (roomPos[r1] && roomPos[r2]) {
            ctx.beginPath();
            ctx.moveTo(roomPos[r1].x, roomPos[r1].y);
            ctx.lineTo(roomPos[r2].x, roomPos[r2].y);
            ctx.stroke();
        }
    });
    ctx.setLineDash([]);
    
    // Draw housing boxes and labels
    Object.keys(ROOM_COORDINATES).forEach(room => {
        const coord = ROOM_COORDINATES[room];
        const housing = ROOM_HOUSING[room];
        const mapLabel = ROOM_LABELS[room] || room;
        const x = (coord.x / 100) * canvas.width;
        const y = (coord.y / 100) * canvas.height;
        const w = (housing.width / 100) * canvas.width;
        const h = (housing.height / 100) * canvas.height;
        
        // Housing box
        ctx.strokeStyle = "#FFFF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x - w/2, y - h/2, w, h);
        
        // Center dot
        ctx.fillStyle = "#FF00FF";
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        
        // Room label (show MAP name, not backend name)
        ctx.font = "12px monospace";
        ctx.fillStyle = "#FFFFFF";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 3;
        ctx.strokeText(mapLabel, x + 10, y - 10);
        ctx.fillText(mapLabel, x + 10, y - 10);
        
        // Coordinates
        ctx.font = "10px monospace";
        const text = `${coord.x.toFixed(1)}%, ${coord.y.toFixed(1)}%`;
        ctx.strokeText(text, x + 10, y + 5);
        ctx.fillText(text, x + 10, y + 5);
    });
    
    // Legend
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(10, 10, 280, 140);
    ctx.font = "14px Arial";
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText("🔍 DEBUG MODE (Press 'D' to toggle)", 20, 30);
    ctx.font = "12px Arial";
    ctx.fillStyle = "#FF0000"; ctx.fillText("■ Red: Hard map boundary", 20, 55);
    ctx.fillStyle = "#00FFFF"; ctx.fillText("■ Cyan: Room pathways", 20, 75);
    ctx.fillStyle = "#FFFF00"; ctx.fillText("■ Yellow: Agent housing boxes", 20, 95);
    ctx.fillStyle = "#FF00FF"; ctx.fillText("■ Magenta: Room centers", 20, 115);
}

function toggleDebug() {
    debugOverlayVisible = !debugOverlayVisible;
    if (debugCanvas) {
        debugCanvas.style.display = debugOverlayVisible ? "block" : "none";
    }
    drawDebugOverlay();
    console.log(debugOverlayVisible ? "🔍 Debug ON" : "🔍 Debug OFF");
}

// ============================================================================
// GAME LOGIC
// ============================================================================

const COLOR_MAP = {
    "red": "#C51111", "orange": "#EF7D0D", "yellow": "#F5F557",
    "lime": "#50EF39", "green": "#117F2D", "cyan": "#38FEDC",
    "blue": "#132ED1", "purple": "#6B2FBB", "brown": "#71491E",
    "pink": "#ED54BA", "white": "#D6E0F0", "black": "#3F474E",
    "🔴": "#C51111", "🟠": "#EF7D0D", "🟡": "#F5F557",
    "🟢": "#117F2D", "🟣": "#6B2FBB", "🟤": "#71491E",
    "🔵": "#132ED1", "🔷": "#38FEDC", "💗": "#ED54BA",
    "🟩": "#50EF39", "⚪": "#D6E0F0", "⚫": "#3F474E"
};

function getAgentColor(colorKey) {
    if (!colorKey || typeof colorKey !== "string") return "#888888";
    var key = colorKey.trim().toLowerCase();
    return COLOR_MAP[key] || COLOR_MAP[colorKey.trim()] || "#888888";
}

function getModelAbbreviation(modelName) {
    if (!modelName) return "Unknown";
    if (modelName.includes("TinyLlama")) return "TinyLlama";
    if (modelName.includes("Qwen")) {
        if (modelName.includes("1.5B")) return "Qwen 1.5B";
        return "Qwen";
    }
    if (modelName.includes("Llama")) return "Llama";
    if (modelName.includes("DeepSeek")) return "DeepSeek";
    return modelName.split("/").pop().substring(0, 10);
}

let lastEventCount = 0;
let agentMarkers = {};
let isPaused = false;
let pollingInterval = null;

function updateGameParams(gameInfo) {
    if (!gameInfo) return;
    const gameIdEl = document.getElementById("gameId");
    const totalAgentsEl = document.getElementById("totalAgents");
    const byzantineEl = document.getElementById("byzantineCount");
    const honestEl = document.getElementById("honestCount");
    const currentRoundEl = document.getElementById("currentRound");
    const totalRoundsEl = document.getElementById("totalRounds");
    const statusEl = document.getElementById("gameStatus");
    
    if (gameIdEl && gameInfo.game_id) gameIdEl.textContent = gameInfo.game_id;
    if (totalAgentsEl && gameInfo.total_agents) totalAgentsEl.textContent = gameInfo.total_agents;
    if (byzantineEl && gameInfo.byzantine_count !== undefined) byzantineEl.textContent = gameInfo.byzantine_count;
    if (honestEl && gameInfo.honest_count !== undefined) honestEl.textContent = gameInfo.honest_count;
    if (currentRoundEl && gameInfo.round !== undefined) currentRoundEl.textContent = gameInfo.round;
    if (totalRoundsEl && gameInfo.total_rounds) totalRoundsEl.textContent = gameInfo.total_rounds;
    if (statusEl && gameInfo.status) {
        statusEl.textContent = gameInfo.status.charAt(0).toUpperCase() + gameInfo.status.slice(1);
    }
}

function updateAgentPositions(agents) {
    if (!agents) return;
    const markersContainer = document.getElementById("agentMarkers");
    if (!markersContainer) return;
    const mapImage = document.getElementById("mapImage");
    if (!mapImage) return;
    const mapRect = mapImage.getBoundingClientRect();
    
    markersContainer.innerHTML = "";
    agentMarkers = {};
    
    const agentsByRoom = {};
    Object.keys(agents).forEach(function(agentKey) {
        const agent = agents[agentKey];
        if (!agent.location) return;
        if (!agentsByRoom[agent.location]) agentsByRoom[agent.location] = [];
        agentsByRoom[agent.location].push({key: agentKey, agent: agent});
    });
    
    Object.keys(agentsByRoom).forEach(function(roomName) {
        const roomAgents = agentsByRoom[roomName];
        const roomCoords = ROOM_COORDINATES[roomName];
        const housing = ROOM_HOUSING[roomName] || { width: 3, height: 3 };
        
        if (!roomCoords) {
            console.warn("Unknown room:", roomName);
            return;
        }
        
        roomAgents.forEach(function(item, index) {
            const agentKey = item.key;
            const agent = item.agent;
            const row = Math.floor(index / 2);
            const col = index % 2;
            const offsetX = (col - 0.5) * (housing.width / 2);
            const offsetY = (row - 0.5) * (housing.height / 2);
            const x_percent = roomCoords.x + offsetX;
            const y_percent = roomCoords.y + offsetY;
            const x = (x_percent / 100) * mapRect.width;
            const y = (y_percent / 100) * mapRect.height;
            
            const agentNumRaw = agentKey.replace("Agent_", "");
            const agentIndex = parseInt(agentNumRaw, 10);
            const displayNum = Number.isNaN(agentIndex) ? agentNumRaw : agentIndex + 1;
            const color = getAgentColor(agent.color);
            
            const marker = document.createElement("div");
            marker.className = "agent-marker";
            marker.id = "marker-" + agentNumRaw;
            marker.style.backgroundColor = color;
            marker.style.left = x + "px";
            marker.style.top = y + "px";
            marker.textContent = displayNum;
            marker.title = agentKey + " - " + roomName;
            
            if (agent.status === "eliminated" || agent.status === "voted_off") {
                marker.style.opacity = "0.5";
            }
            
            markersContainer.appendChild(marker);
            agentMarkers[agentKey] = marker;
        });
    });
}

function updateStatusTable(agents) {
    if (!agents) return;
    const tbody = document.getElementById("statusTableBody");
    if (!tbody) return;
    tbody.innerHTML = "";
    
    const agentKeys = Object.keys(agents).sort(function(a, b) {
        const numA = parseInt(a.replace("Agent_", ""));
        const numB = parseInt(b.replace("Agent_", ""));
        return numA - numB;
    });
    
    agentKeys.forEach(function(agentKey) {
        const agent = agents[agentKey];
        const agentNumRaw = agentKey.replace("Agent_", "");
        const agentIndex = parseInt(agentNumRaw, 10);
        const displayNum = Number.isNaN(agentIndex) ? agentNumRaw : agentIndex + 1;
        const row = document.createElement("tr");
        
        const numCell = document.createElement("td");
        numCell.textContent = displayNum;
        row.appendChild(numCell);
        
        const modelCell = document.createElement("td");
        const modelName = agent.stats && agent.stats.model_name ? agent.stats.model_name : agent.model;
        modelCell.textContent = getModelAbbreviation(modelName);
        row.appendChild(modelCell);
        
        const colorCell = document.createElement("td");
        const colorIndicator = document.createElement("span");
        colorIndicator.className = "agent-color-indicator";
        colorIndicator.style.backgroundColor = getAgentColor(agent.color);
        colorCell.appendChild(colorIndicator);
        row.appendChild(colorCell);
        
        const statusCell = document.createElement("td");
        statusCell.className = "agent-status";
        let statusText = "";
        if (agent.status === "active" || agent.status === "alive") {
            statusText = "Alive ✓";
        } else if (agent.status === "eliminated") {
            statusText = "Dead ✗";
            row.classList.add("agent-dead");
        } else if (agent.status === "voted_off") {
            statusText = "Voted Off ⚖";
            row.classList.add("agent-dead");
        } else {
            statusText = agent.status || "Unknown";
        }
        statusCell.textContent = statusText;
        row.appendChild(statusCell);
        
        const votesCell = document.createElement("td");
        const votesValue = agent.stats && typeof agent.stats.votes_received !== "undefined"
            ? agent.stats.votes_received : agent.votes_received;
        votesCell.textContent = votesValue || 0;
        row.appendChild(votesCell);
        
        const killsCell = document.createElement("td");
        if (agent.role === "byzantine") {
            const elimValue = agent.stats && typeof agent.stats.eliminations !== "undefined"
                ? agent.stats.eliminations : agent.eliminations;
            killsCell.textContent = elimValue || 0;
        } else {
            killsCell.textContent = "n/a";
        }
        row.appendChild(killsCell);
        
        tbody.appendChild(row);
    });
}

function updateLiveFeed(events) {
    const feedContent = document.getElementById("feedContent");
    if (!feedContent) return;
    
    if (!events || !Array.isArray(events)) {
        if (feedContent.children.length === 0) {
            feedContent.innerHTML = "<div class=\"feed-event\"><span class=\"feed-timestamp\">[--]</span><span class=\"feed-text\">No events yet.</span></div>";
        }
        return;
    }
    
    const newEvents = events.slice(lastEventCount);
    
    if (newEvents.length === 0 && feedContent.children.length === 0) {
        feedContent.innerHTML = "<div class=\"feed-event\"><span class=\"feed-timestamp\">[--]</span><span class=\"feed-text\">No events yet.</span></div>";
        return;
    }
    
    if (newEvents.length > 0 && feedContent.children.length === 1) {
        var first = feedContent.querySelector(".feed-text");
        if (first && first.textContent.indexOf("No events yet") !== -1) {
            feedContent.innerHTML = "";
        }
    }
    
    newEvents.forEach(function(event) {
        const eventDiv = document.createElement("div");
        eventDiv.className = "feed-event";
        const timestampSpan = document.createElement("span");
        timestampSpan.className = "feed-timestamp";
        timestampSpan.textContent = "[" + (event.time || event.timestamp || "--:--") + "]";
        const textSpan = document.createElement("span");
        textSpan.className = "feed-text";
        textSpan.textContent = event.msg || event.text || "No message";
        eventDiv.appendChild(timestampSpan);
        eventDiv.appendChild(textSpan);
        feedContent.appendChild(eventDiv);
    });
    
    lastEventCount = events.length;
    feedContent.scrollTop = feedContent.scrollHeight;
}

async function updateGameState() {
    try {
        const response = await fetch("/api/game_state");
        if (!response.ok) return;
        const data = await response.json();
        
        if (data.status === 'waiting') {
            const statusEl = document.getElementById("gameStatus");
            if (statusEl) statusEl.textContent = "Waiting...";
            return;
        }
        
        if (data.status === 'error') {
            const statusEl = document.getElementById("gameStatus");
            if (statusEl) {
                statusEl.textContent = "Error: " + (data.message || "Unknown");
                statusEl.style.color = "#ff4444";
            }
            if (data.process_ended && pollingInterval) {
                clearInterval(pollingInterval);
                pollingInterval = null;
            }
            return;
        }
        
        if (data) {
            let gameInfo = null;
            if (data.game_info) {
                gameInfo = data.game_info;
            } else if (data.agents && Object.keys(data.agents).length > 0) {
                const agents = data.agents || {};
                const agentKeys = Object.keys(agents);
                const totalAgents = agentKeys.length;
                let byzCount = 0, honestCount = 0;
                agentKeys.forEach(function(key) {
                    const a = agents[key];
                    if (!a || !a.role) return;
                    if (a.role === "byzantine") byzCount++;
                    else if (a.role === "honest") honestCount++;
                });
                gameInfo = {
                    game_id: data.game_id,
                    total_agents: totalAgents,
                    byzantine_count: byzCount,
                    honest_count: honestCount,
                    round: data.global && typeof data.global.round !== "undefined" ? data.global.round : undefined,
                    total_rounds: data.global && typeof data.global.total_rounds !== "undefined" ? data.global.total_rounds : undefined,
                    status: data.global && data.global.current_phase ? data.global.current_phase : data.status
                };
            }
            if (gameInfo) updateGameParams(gameInfo);
        }
        
        if (data.agents && Object.keys(data.agents).length > 0) {
            updateAgentPositions(data.agents);
            updateStatusTable(data.agents);
        }
        
        if (data) {
            var events = [];
            if (Array.isArray(data.events)) {
                events = data.events;
            } else if (data.global && Array.isArray(data.global.ui_event_log)) {
                events = data.global.ui_event_log;
            }
            updateLiveFeed(events);
        }
    } catch (error) {
        console.error("Error:", error);
    }
}

function togglePause() {
    isPaused = !isPaused;
    const pauseBtn = document.getElementById("pauseBtn");
    if (isPaused) {
        pauseBtn.textContent = "▶ Resume";
        pauseBtn.classList.add("paused");
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    } else {
        pauseBtn.textContent = "⏸ Pause";
        pauseBtn.classList.remove("paused");
        if (!pollingInterval) {
            pollingInterval = setInterval(updateGameState, 2000);
        }
    }
}

function exitToHome() {
    if (confirm("Exit?")) window.location.href = "/";
}

window.addEventListener("DOMContentLoaded", function() {
    createDebugCanvas();
    document.addEventListener("keydown", (e) => {
        if (e.key === "d" || e.key === "D") toggleDebug();
    });
    
    window.addEventListener("resize", () => {
        if (debugCanvas) {
            const container = debugCanvas.parentElement;
            const rect = container.getBoundingClientRect();
            debugCanvas.width = rect.width;
            debugCanvas.height = rect.height;
            drawDebugOverlay();
        }
    });
    
    updateGameState();
    pollingInterval = setInterval(updateGameState, 2000);
    
    const pauseBtn = document.getElementById("pauseBtn");
    const exitBtn = document.getElementById("exitBtn");
    if (pauseBtn) pauseBtn.addEventListener("click", togglePause);
    if (exitBtn) exitBtn.addEventListener("click", exitToHome);
    
    console.log("🎮 Press 'D' for debug!");
});