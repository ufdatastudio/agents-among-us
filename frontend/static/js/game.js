/**
 * Game Simulation - FINAL WORKING VERSION
 * Your exact coordinates + Housing boxes + Sprites
 * Press 'D' for debug overlay
 */

const ROOM_COORDINATES = {
    "Reactor": { x: 14.2, y: 56.3 },
    "UpperEngine": { x: 18.9, y: 25.2 },
    "LowerEngine": { x: 26.9, y: 88.0 },
    "Security": { x: 37.5, y: 44.3 },
    "MedBay": { x: 31.8, y: 12.3 },
    "Electrical": { x: 37.5, y: 62.8 },
    "Cafeteria": { x: 50.7, y: 23.0 },
    "Admin": { x: 61.7, y: 55.8 },
    "Storage": { x: 50.7, y: 85.5 },
    "Weapons": { x: 80.2, y: 16.8 },
    "O2": { x: 67.4, y: 36.3 },
    "Navigation": { x: 92.6, y: 45.4 },
    "Shields": { x: 80.4, y: 70.5 },
    "Communications": { x: 80.5, y: 94.5 }
};

const ROOM_LABELS = {
    "Reactor": "Clock", "UpperEngine": "Air Cooling", "LowerEngine": "Liquid Cooling",
    "Security": "Logs", "MedBay": "Diagnostics", "Electrical": "Bus",
    "Cafeteria": "CPU", "Admin": "BIOS", "Storage": "SSD",
    "Weapons": "GPU", "O2": "VRM", "Navigation": "Network",
    "Shields": "Firewall", "Communications": "IO"
};

const ROOM_HOUSING = {
    "Reactor": { width: 10, height: 20 },
    "UpperEngine": { width: 10, height: 20 },
    "LowerEngine": { width: 10, height: 20 },
    "Security": { width: 6, height: 12 },
    "MedBay": { width: 6, height: 12 },
    "Electrical": { width: 6, height: 12 },
    "Cafeteria": { width: 10, height: 20 },
    "Admin": { width: 6, height: 12 },
    "Storage": { width: 12, height: 12 },
    "Weapons": { width: 12, height: 12 },
    "O2": { width: 6, height: 12 },
    "Navigation": { width: 10, height: 20 },
    "Shields": { width: 12, height: 12 },
    "Communications": { width: 6, height: 12 }
};

const ROOM_CONNECTIONS = [
    ["Reactor", "Security"], ["Reactor", "UpperEngine"], ["Reactor", "LowerEngine"],
    ["Security", "UpperEngine"], ["Security", "LowerEngine"],
    ["UpperEngine", "LowerEngine"], ["UpperEngine", "MedBay"], ["UpperEngine", "Cafeteria"],
    ["LowerEngine", "Electrical"], ["LowerEngine", "Storage"],
    ["MedBay", "Cafeteria"], ["Electrical", "Storage"],
    ["Cafeteria", "Admin"], ["Cafeteria", "Storage"], ["Cafeteria", "Weapons"],
    ["Admin", "Storage"], ["Weapons", "O2"], ["Weapons", "Navigation"], ["Weapons", "Shields"],
    ["O2", "Navigation"], ["O2", "Shields"], ["Navigation", "Shields"],
    ["Storage", "Shields"], ["Storage", "Communications"], ["Shields", "Communications"]
];

let debugOverlayVisible = false;
let debugCanvas = null;

function createDebugCanvas() {
    if (debugCanvas) return debugCanvas;
    const mapWrapper = document.querySelector(".map-wrapper");
    if (!mapWrapper) return null;
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
    mapWrapper.appendChild(canvas);
    const rect = mapWrapper.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    debugCanvas = canvas;
    return canvas;
}

function getImageDimensions() {
    const mapImage = document.getElementById("mapImage");
    if (!mapImage) return null;
    const rect = mapImage.getBoundingClientRect();
    const containerRect = mapImage.parentElement.getBoundingClientRect();
    return {
        imgWidth: rect.width,
        imgHeight: rect.height,
        imgLeft: rect.left - containerRect.left,
        imgTop: rect.top - containerRect.top
    };
}

function drawDebugOverlay() {
    const canvas = createDebugCanvas();
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!debugOverlayVisible) return;
    
    const dims = getImageDimensions();
    if (!dims) return;
    
    ctx.strokeStyle = "#FF0000";
    ctx.lineWidth = 4;
    ctx.strokeRect(dims.imgLeft + 2, dims.imgTop + 2, dims.imgWidth - 4, dims.imgHeight - 4);
    
    const roomPos = {};
    Object.keys(ROOM_COORDINATES).forEach(room => {
        const coord = ROOM_COORDINATES[room];
        roomPos[room] = {
            x: dims.imgLeft + (coord.x / 100) * dims.imgWidth,
            y: dims.imgTop + (coord.y / 100) * dims.imgHeight
        };
    });
    
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
    
    Object.keys(ROOM_COORDINATES).forEach(room => {
        const coord = ROOM_COORDINATES[room];
        const housing = ROOM_HOUSING[room];
        const mapLabel = ROOM_LABELS[room] || room;
        const x = dims.imgLeft + (coord.x / 100) * dims.imgWidth;
        const y = dims.imgTop + (coord.y / 100) * dims.imgHeight;
        const w = (housing.width / 100) * dims.imgWidth;
        const h = (housing.height / 100) * dims.imgHeight;
        ctx.strokeStyle = "#FFFF00";
        ctx.lineWidth = 2;
        ctx.strokeRect(x - w/2, y - h/2, w, h);
        ctx.fillStyle = "#FF00FF";
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = "12px monospace";
        ctx.fillStyle = "#FFFFFF";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 3;
        ctx.strokeText(mapLabel, x + 10, y - 10);
        ctx.fillText(mapLabel, x + 10, y - 10);
        ctx.font = "10px monospace";
        const text = `${coord.x.toFixed(1)}%, ${coord.y.toFixed(1)}%`;
        ctx.strokeText(text, x + 10, y + 5);
        ctx.fillText(text, x + 10, y + 5);
    });
    
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(10, 10, 280, 140);
    ctx.font = "14px Arial";
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText("🔍 DEBUG (Press 'D')", 20, 30);
    ctx.font = "12px Arial";
    ctx.fillStyle = "#FF0000"; ctx.fillText("■ Red: Map boundary", 20, 55);
    ctx.fillStyle = "#00FFFF"; ctx.fillText("■ Cyan: Pathways", 20, 75);
    ctx.fillStyle = "#FFFF00"; ctx.fillText("■ Yellow: Housing boxes", 20, 95);
    ctx.fillStyle = "#FF00FF"; ctx.fillText("■ Magenta: Room centers", 20, 115);
}

function toggleDebug() {
    debugOverlayVisible = !debugOverlayVisible;
    if (debugCanvas) debugCanvas.style.display = debugOverlayVisible ? "block" : "none";
    drawDebugOverlay();
    console.log(debugOverlayVisible ? "🔍 Debug ON" : "🔍 Debug OFF");
}

const COLOR_MAP = {
    "red": "#C51111", "orange": "#EF7D0D", "yellow": "#F5F557", "lime": "#50EF39",
    "green": "#117F2D", "cyan": "#38FEDC", "blue": "#132ED1", "purple": "#6B2FBB",
    "brown": "#71491E", "pink": "#ED54BA", "white": "#D6E0F0", "black": "#3F474E",
    "🔴": "#C51111", "🟠": "#EF7D0D", "🟡": "#F5F557", "🟢": "#117F2D",
    "🟣": "#6B2FBB", "🟤": "#71491E", "🔵": "#132ED1", "🔷": "#38FEDC",
    "💗": "#ED54BA", "🟩": "#50EF39", "⚪": "#D6E0F0", "⚫": "#3F474E"
};

function getAgentColor(colorKey) {
    if (!colorKey || typeof colorKey !== "string") return "#888888";
    var key = colorKey.trim().toLowerCase();
    return COLOR_MAP[key] || COLOR_MAP[colorKey.trim()] || "#888888";
}

// Color slug from backend (may send "red" or "🔴"); used for sprite lookup
function getColorSlug(colorKey) {
    if (!colorKey || typeof colorKey !== "string") return "red";
    var key = colorKey.trim().toLowerCase();
    if (LIVING_SPRITES[key]) return key;
    var emojiToSlug = { "🔴": "red", "🟠": "orange", "🟡": "yellow", "🟩": "lime", "🟢": "green", "🔷": "cyan", "🔵": "blue", "🟣": "purple", "🟤": "brown", "💗": "pink", "⚪": "white", "⚫": "black" };
    return emojiToSlug[colorKey.trim()] || "red";
}

// Among Us sprites: living (by color slug) and dead (by color slug)
var LIVING_SPRITES = {
    red: "https://preview.redd.it/an871k4o1sn51.png?width=440&format=png&auto=webp&s=85dcd6cb73b8760802e254ee14dfa3c7ab444591",
    orange: "https://preview.redd.it/iio3xm4o1sn51.png?width=440&format=png&auto=webp&s=2b9fb1b29396502998feda5c6ed2ed75919c6ad8",
    yellow: "https://preview.redd.it/xprpkp063sn51.png?width=440&format=png&auto=webp&s=5d51eb262af4a50e8f935218feb52682540aa525",
    lime: "https://preview.redd.it/76glbq4o1sn51.png?width=440&format=png&auto=webp&s=a22610bfbd735d024448389fd80009b255c33524",
    green: "https://preview.redd.it/vf3ojm4o1sn51.png?width=440&format=png&auto=webp&s=7cfa65a910d76e324fcc4c23468a9b801c3b74d5",
    cyan: "https://preview.redd.it/0j244l4o1sn51.png?width=440&format=png&auto=webp&s=c74e2de99bdb7da7471469d8274a4eaae244207e",
    blue: "https://preview.redd.it/ph2jho4o1sn51.png?width=440&format=png&auto=webp&s=7e080e5447d69d1425a8b8a20f1115de18aa69fd",
    purple: "https://preview.redd.it/9kvk25sh2sn51.png?width=440&format=png&auto=webp&s=c469d1dc3fda76a0d2271cecb8d422f1aff925ab",
    brown: "https://preview.redd.it/f7f4fmpi2sn51.png?width=440&format=png&auto=webp&s=79d8eaf10daa28753816cfc8ec5cd26cfa517d29",
    pink: "https://preview.redd.it/ppawzo4o1sn51.png?width=440&format=png&auto=webp&s=d09c261013546996e8325d507ff230a7e9513793",
    white: "https://preview.redd.it/xyqo6hx42sn51.png?width=440&format=png&auto=webp&s=3bf357e64a68883aee1618a1abdadc16d9ceee73",
    black: "https://preview.redd.it/4eof2l4o1sn51.png?width=440&format=png&auto=webp&s=02f3a9c7fdb96a50204c5dc272a7e72dfff7cbac"
};
var DEAD_SPRITES = {
    red: "https://preview.redd.it/rnj1si3kzwn51.png?width=720&format=png&auto=webp&s=6e7243bb5c2d8f27921313b0f8ef27617523d604",
    orange: "https://preview.redd.it/h506lc3kzwn51.png?width=720&format=png&auto=webp&s=de8d4c645916b08bec416f5d9d3a1486f25aa8a3",
    yellow: "https://preview.redd.it/jogjcd3kzwn51.png?width=720&format=png&auto=webp&s=9da38080a842e0cd3be2a4b5bc30de5023813eba",
    lime: "https://preview.redd.it/yok6ie3kzwn51.png?width=720&format=png&auto=webp&s=45a18604dd35acf60755e2116619500d803f2e97",
    green: "https://preview.redd.it/vxq41e3kzwn51.png?width=720&format=png&auto=webp&s=e39aff4b156e52f4b379883418c2afeb89087043",
    cyan: "https://preview.redd.it/jlu5ah3kzwn51.png?width=720&format=png&auto=webp&s=39a7d7b8998ef25b69b8a4d9ef4935a4063b8499",
    blue: "https://preview.redd.it/b26i9g3kzwn51.png?width=720&format=png&auto=webp&s=cd78f7f49e933ac7d68dc2effec086d981501313",
    purple: "https://preview.redd.it/73q3te3kzwn51.png?width=720&format=png&auto=webp&s=922aa6edccc727cf71a26c0dc516eee90d58b403",
    pink: "https://preview.redd.it/lpny4e3kzwn51.png?width=720&format=png&auto=webp&s=d29c06d1fbc294866b320bc4c5e7086ea349b749",
    brown: "https://preview.redd.it/o5tubc3kzwn51.png?width=720&format=png&auto=webp&s=9dda98ebfeed0e63e6275a0f520fec64adea4678",
    white: "https://preview.redd.it/yk5tjb3kzwn51.png?width=720&format=png&auto=webp&s=dc62d0b1b9eea0c16cbda9d08756d4e7b3a97dc0",
    black: "https://preview.redd.it/6vegnf3kzwn51.png?width=720&format=png&auto=webp&s=4ea01f3bd3597b3e10674acf20cd7af468dfd583"
};

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
    const fields = {
        "gameId": gameInfo.game_id,
        "totalAgents": gameInfo.total_agents,
        "byzantineCount": gameInfo.byzantine_count,
        "honestCount": gameInfo.honest_count,
        "currentRound": gameInfo.round,
        "totalRounds": gameInfo.total_rounds
    };
    Object.keys(fields).forEach(id => {
        const el = document.getElementById(id);
        if (el && fields[id] !== undefined) el.textContent = fields[id];
    });
    const statusEl = document.getElementById("gameStatus");
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
    const dims = getImageDimensions();
    if (!dims) return;
    
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
        const housing = ROOM_HOUSING[roomName] || { width: 10, height: 20 };
        if (!roomCoords) return;
        
        roomAgents.forEach(function(item, index) {
            const agentKey = item.key;
            const agent = item.agent;
            const agentsInRoom = roomAgents.length;
            let offsetX = 0, offsetY = 0;
            
            if (agentsInRoom === 1) {
                offsetX = 0;
                offsetY = 0;
            } else if (agentsInRoom === 2) {
                offsetX = (index === 0 ? -0.3 : 0.3) * (housing.width / 2);
                offsetY = 0;
            } else if (agentsInRoom === 3) {
                const positions = [{x: -0.3, y: -0.2}, {x: 0.3, y: -0.2}, {x: 0, y: 0.2}];
                offsetX = positions[index].x * (housing.width / 2);
                offsetY = positions[index].y * (housing.height / 2);
            } else if (agentsInRoom === 4) {
                const row = Math.floor(index / 2);
                const col = index % 2;
                offsetX = (col - 0.5) * (housing.width / 2);
                offsetY = (row - 0.5) * (housing.height / 2);
            } else {
                const cols = Math.ceil(Math.sqrt(agentsInRoom));
                const row = Math.floor(index / cols);
                const col = index % cols;
                offsetX = (col / Math.max(cols - 1, 1) - 0.5) * housing.width * 0.8;
                offsetY = (row / Math.max(Math.ceil(agentsInRoom / cols) - 1, 1) - 0.5) * housing.height * 0.8;
            }
            
            const x_percent = roomCoords.x + offsetX;
            const y_percent = roomCoords.y + offsetY;
            const x = dims.imgLeft + (x_percent / 100) * dims.imgWidth;
            const y = dims.imgTop + (y_percent / 100) * dims.imgHeight;
            
            const agentNumRaw = agentKey.replace("Agent_", "");
            const agentIndex = parseInt(agentNumRaw, 10);
            const displayNum = Number.isNaN(agentIndex) ? agentNumRaw : agentIndex;
            
            // Hardcoded color order matching config page: Red, Orange, Yellow, Lime, Green, Cyan, Blue, Purple, Brown, Pink, White, Black
            const hardcodedColors = ["red", "orange", "yellow", "lime", "green", "cyan", "blue", "purple", "brown", "pink", "white", "black"];
            const colorSlug = hardcodedColors[agentIndex] || "red";
            
            const isAlive = agent.status !== "eliminated" && agent.status !== "voted_off";
            const spriteUrl = isAlive
                ? (LIVING_SPRITES[colorSlug] || LIVING_SPRITES.red)
                : (DEAD_SPRITES[colorSlug] || DEAD_SPRITES.red);
            
            const marker = document.createElement("div");
            marker.className = "agent-marker";
            marker.id = "marker-" + agentNumRaw;
            marker.style.left = x + "px";
            marker.style.top = y + "px";
            marker.title = "Agent " + displayNum + " - " + roomName + (isAlive ? " (Alive)" : " (Dead)");
            if (!isAlive) {
                marker.classList.add("agent-marker-dead");
            }
            const img = document.createElement("img");
            img.className = "agent-marker-sprite";
            img.src = spriteUrl;
            img.alt = "Agent " + displayNum;
            marker.appendChild(img);
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
        return parseInt(a.replace("Agent_", "")) - parseInt(b.replace("Agent_", ""));
    });
    
    agentKeys.forEach(function(agentKey) {
        const agent = agents[agentKey];
        const agentNumRaw = agentKey.replace("Agent_", "");
        const agentIndex = parseInt(agentNumRaw, 10);
        const displayNum = Number.isNaN(agentIndex) ? agentNumRaw : agentIndex;
        const row = document.createElement("tr");
        
        const numCell = document.createElement("td");
        numCell.textContent = displayNum;
        row.appendChild(numCell);
        
        const modelCell = document.createElement("td");
        const modelName = agent.stats && agent.stats.model_name ? agent.stats.model_name : agent.model;
        modelCell.textContent = getModelAbbreviation(modelName);
        row.appendChild(modelCell);
        
        const roleCell = document.createElement("td");
        const role = (agent.role || "").toLowerCase();
        roleCell.textContent = role === "byzantine" ? "Byz." : "Hon.";
        roleCell.className = "agent-role-cell";
        if (role === "byzantine") row.classList.add("agent-byzantine");
        row.appendChild(roleCell);
        
        const statusCell = document.createElement("td");
        statusCell.className = "agent-status-cell";
        var isAlive = agent.status === "active" || agent.status === "alive";
        if (agent.status === "eliminated" || agent.status === "voted_off") {
            row.classList.add("agent-dead");
        }
        
        // Hardcoded color order matching config page: Red, Orange, Yellow, Lime, Green, Cyan, Blue, Purple, Brown, Pink, White, Black
        const hardcodedColors = ["red", "orange", "yellow", "lime", "green", "cyan", "blue", "purple", "brown", "pink", "white", "black"];
        var colorSlug = hardcodedColors[agentIndex] || "red";
        
        var spriteUrl = isAlive
            ? (LIVING_SPRITES[colorSlug] || LIVING_SPRITES.red)
            : (DEAD_SPRITES[colorSlug] || DEAD_SPRITES.red);
        var img = document.createElement("img");
        img.className = "agent-status-sprite";
        img.src = spriteUrl;
        img.alt = isAlive ? "Alive" : "Dead";
        img.title = isAlive ? "Alive" : (agent.status === "voted_off" ? "Voted off" : "Dead");
        statusCell.appendChild(img);
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
        var eventType = (event.type || "").toLowerCase();
        if (eventType === "kill" || eventType === "eject") {
            eventDiv.classList.add("feed-event--danger");
        } else if (eventType === "meeting") {
            eventDiv.classList.add("feed-event--warning");
        }
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
    // Only auto-scroll when we added new events AND user was already near the bottom
    if (newEvents.length > 0) {
        var el = feedContent;
        var atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        if (atBottom) {
            el.scrollTop = el.scrollHeight;
        }
    }
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
                let byzCount = 0, honestCount = 0;
                agentKeys.forEach(function(key) {
                    const a = agents[key];
                    if (!a || !a.role) return;
                    if (a.role === "byzantine") byzCount++;
                    else if (a.role === "honest") honestCount++;
                });
                gameInfo = {
                    game_id: data.game_id,
                    total_agents: agentKeys.length,
                    byzantine_count: byzCount,
                    honest_count: honestCount,
                    round: data.global && data.global.round,
                    total_rounds: data.global && data.global.total_rounds,
                    status: data.global && data.global.current_phase || data.status
                };
            }
            if (gameInfo) updateGameParams(gameInfo);
            
            // Stop polling if game is over
            const currentPhase = (data.global && data.global.current_phase) || "";
            if (currentPhase === "GAME OVER" && pollingInterval) {
                console.log("🎮 Game ended, stopping polling");
                clearInterval(pollingInterval);
                pollingInterval = null;
            }
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
    
    let resizeTimeout;
    window.addEventListener("resize", () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (debugCanvas) {
                const container = debugCanvas.parentElement;
                const rect = container.getBoundingClientRect();
                debugCanvas.width = rect.width;
                debugCanvas.height = rect.height;
                drawDebugOverlay();
            }
            updateGameState();
        }, 250);
    });
    
    updateGameState();
    pollingInterval = setInterval(updateGameState, 2000);
    
    const pauseBtn = document.getElementById("pauseBtn");
    const exitBtn = document.getElementById("exitBtn");
    if (pauseBtn) pauseBtn.addEventListener("click", togglePause);
    if (exitBtn) exitBtn.addEventListener("click", exitToHome);
    
    console.log("🎮 Press 'D' for debug!");
});