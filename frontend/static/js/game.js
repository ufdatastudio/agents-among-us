/**
 * Game Simulation 
 * Press D for debug overlay
 */

const ROOM_COORDINATES = {
    "Clock": { x: 12.7, y: 54.3 },
    "Air Cooling": { x: 17.4, y: 22.3 },
    "Liquid Cooling": { x: 25.3, y: 86.4 },
    "Logs": { x: 35.7, y: 43.9 },
    "Diagnostics": { x: 30.3, y: 10.4},
    "Bus": { x: 36.2, y: 60.1 },
    "CPU": { x: 49.0, y: 21.2 },
    "BIOS": { x: 60.2, y: 54.2 },
    "SSD": { x: 49.3, y: 84.2 },
    "GPU": { x: 78.9, y: 15.2 },
    "VRM": { x: 66.3, y: 33.3 },
    "Network": { x: 91.6, y: 43.5 },
    "Firewall": { x: 78.7, y: 68.8 },
    "IO": { x: 79.0, y: 92.7 }
};

const ROOM_HOUSING = {
    "Clock": { width: 10, height: 20 },
    "Air Cooling": { width: 10, height: 20 },
    "Liquid Cooling": { width: 10, height: 20 },
    "Logs": { width: 6, height: 12 },
    "Diagnostics": { width: 6, height: 12 },
    "Bus": { width: 6, height: 12 },
    "CPU": { width: 10, height: 20 },
    "BIOS": { width: 6, height: 12 },
    "SSD": { width: 12, height: 12 },
    "GPU": { width: 12, height: 12 },
    "VRM": { width: 6, height: 12 },
    "Network": { width: 10, height: 20 },
    "Firewall": { width: 12, height: 12 },
    "IO": { width: 6, height: 12 }
};

const ROOM_CONNECTIONS = [
    ["Clock", "Logs"], ["Clock", "Air Cooling"], ["Clock", "Liquid Cooling"],
    ["Logs", "Air Cooling"], ["Logs", "Liquid Cooling"],
    ["Air Cooling", "Liquid Cooling"], ["Air Cooling", "Diagnostics"], ["Air Cooling", "CPU"],
    ["Liquid Cooling", "Bus"], ["Liquid Cooling", "SSD"],
    ["Diagnostics", "CPU"], ["Bus", "SSD"],
    ["CPU", "BIOS"], ["CPU", "SSD"], ["CPU", "GPU"],
    ["BIOS", "SSD"], ["GPU", "VRM"], ["GPU", "Network"], ["GPU", "Firewall"],
    ["VRM", "Network"], ["VRM", "Firewall"], ["Network", "Firewall"],
    ["SSD", "Firewall"], ["SSD", "IO"], ["Firewall", "IO"]
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
        ctx.strokeText(room, x + 10, y - 10);
        ctx.fillText(room, x + 10, y - 10);
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

function getColorSlug(colorKey) {
    if (!colorKey || typeof colorKey !== "string") return "red";
    var key = colorKey.trim().toLowerCase();
    if (LIVING_SPRITES[key]) return key;
    var emojiToSlug = { "🔴": "red", "🟠": "orange", "🟡": "yellow", "🟩": "lime", "🟢": "green", "🔷": "cyan", "🔵": "blue", "🟣": "purple", "🟤": "brown", "💗": "pink", "⚪": "white", "⚫": "black" };
    return emojiToSlug[colorKey.trim()] || "red";
}

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
let pollingInterval = null;
let lastPhase = "";
let lastRound = 0;
let clearedAgents = new Set(); // Track agents removed by meetings/ejections

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
        
        // Remove ejected agents permanently
        if (agent.status === "ejected") {
            clearedAgents.add(agentKey);
            return;
        }
        
        // Remove agents that have been cleared by a meeting
        if (clearedAgents.has(agentKey)) {
            return;
        }
        
        // Show all other agents (alive or newly eliminated)
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
                offsetX = (index === 0 ? -1.5 : 1.5);
                offsetY = 0;
            } else if (agentsInRoom === 3) {
                const positions = [{x: -1.5, y: -1}, {x: 1.5, y: -1}, {x: 0, y: 1}];
                offsetX = positions[index].x;
                offsetY = positions[index].y;
            } else if (agentsInRoom === 4) {
                const row = Math.floor(index / 2);
                const col = index % 2;
                offsetX = (col - 0.5) * 3;
                offsetY = (row - 0.5) * 2;
            } else {
                const cols = Math.ceil(Math.sqrt(agentsInRoom));
                const row = Math.floor(index / cols);
                const col = index % cols;
                const maxWidth = housing.width * 0.7;
                const maxHeight = housing.height * 0.7;
                offsetX = (col / Math.max(cols - 1, 1) - 0.5) * maxWidth;
                offsetY = (row / Math.max(Math.ceil(agentsInRoom / cols) - 1, 1) - 0.5) * maxHeight;
            }
            
            const x_percent = roomCoords.x + offsetX;
            const y_percent = roomCoords.y + offsetY;
            const x = dims.imgLeft + (x_percent / 100) * dims.imgWidth;
            const y = dims.imgTop + (y_percent / 100) * dims.imgHeight;
            
            const agentNumRaw = agentKey.replace("Agent_", "");
            const agentIndex = parseInt(agentNumRaw, 10);
            const displayNum = Number.isNaN(agentIndex) ? agentNumRaw : agentIndex;
            
            const hardcodedColors = ["red", "orange", "yellow", "lime", "green", "cyan", "blue", "purple", "brown", "pink", "white", "black"];
            const colorSlug = hardcodedColors[agentIndex] || "red";
            
            const isAlive = agent.status === "active" || agent.status === "alive";
            const spriteUrl = isAlive
                ? (LIVING_SPRITES[colorSlug] || LIVING_SPRITES.red)
                : (DEAD_SPRITES[colorSlug] || DEAD_SPRITES.red);
            
            const marker = document.createElement("div");
            marker.className = "agent-marker";
            marker.id = "marker-" + agentNumRaw;
            marker.style.left = x + "px";
            marker.style.top = y + "px";
            marker.title = "Agent " + displayNum + " - " + roomName + (isAlive ? " (Alive)" : " (Dead Body)");
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
        if (agent.status === "eliminated" || agent.status === "ejected") {
            row.classList.add("agent-dead");
        }
        
        const hardcodedColors = ["red", "orange", "yellow", "lime", "green", "cyan", "blue", "purple", "brown", "pink", "white", "black"];
        var colorSlug = hardcodedColors[agentIndex] || "red";
        
        var spriteUrl = isAlive
            ? (LIVING_SPRITES[colorSlug] || LIVING_SPRITES.red)
            : (DEAD_SPRITES[colorSlug] || DEAD_SPRITES.red);
        var img = document.createElement("img");
        img.className = "agent-status-sprite";
        img.src = spriteUrl;
        img.alt = isAlive ? "Alive" : "Dead";
        img.title = isAlive ? "Alive" : (agent.status === "ejected" ? "Voted off" : "Dead");
        statusCell.appendChild(img);
        row.appendChild(statusCell);
        
        const locationCell = document.createElement("td");
        if (agent.status === "ejected") {
            locationCell.textContent = "Ejected";
        } else {
            locationCell.textContent = agent.location || "Unknown";
        }
        locationCell.style.fontSize = "0.75rem";
        row.appendChild(locationCell);
        
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
        const msg = event.msg || event.text || "";
        const eventType = (event.type || "").toLowerCase();

        if (eventType === "meeting") {
            const title = msg.includes("Body") ? "Body Reported" : "Emergency Meeting";
            showDiscussionChat(title);
        }

        if (eventType === "chat") {
            addDiscussionMessage(event);
            return;
        }

        if (eventType === "vote") {
            addDiscussionMessage(event);
        }

        const eventDiv = document.createElement("div");
        eventDiv.className = "feed-event";
        if (eventType === "kill" || eventType === "eject") {
            eventDiv.classList.add("feed-event--danger");
        } else if (eventType === "meeting" || eventType === "vote") {
            eventDiv.classList.add("feed-event--warning");
        }
        const timestampSpan = document.createElement("span");
        timestampSpan.className = "feed-timestamp";
        timestampSpan.textContent = "[" + (event.time || event.timestamp || "--:--") + "]";
        const textSpan = document.createElement("span");
        textSpan.className = "feed-text";
        textSpan.textContent = msg;
        eventDiv.appendChild(timestampSpan);
        eventDiv.appendChild(textSpan);
        feedContent.appendChild(eventDiv);
    });
    lastEventCount = events.length;
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
            
            const currentPhase = (data.global && data.global.current_phase) || "";
            if (currentPhase === "GAME OVER" && pollingInterval) {
                console.log("🎮 Game ended, stopping polling");
                clearInterval(pollingInterval);
                pollingInterval = null;
                
                const events = (data.global && data.global.ui_event_log) || [];
                const lastEvent = events[events.length - 1];
                const winMessage = lastEvent ? (lastEvent.msg || lastEvent.text || "") : "";
                
                if (winMessage.includes("HONEST") || winMessage.includes("Honest")) {
                    showWinScreen("honest");
                } else if (winMessage.includes("BYZANTINE") || winMessage.includes("Byzantine")) {
                    showWinScreen("byzantine");
                }
            }
        }
        
        if (data.agents && Object.keys(data.agents).length > 0) {
            updateAgentPositions(data.agents);
            updateStatusTable(data.agents);
        }
        
        if (data) {
            const currentPhase = (data.global && data.global.current_phase) || "";
            const currentRound = (data.global && data.global.round) || 0;

            var tickEvents = [];

            if (currentRound && currentRound !== lastRound) {
                if (lastRound !== 0) {
                    tickEvents.push({ msg: "---", type: "tick" });
                }
                tickEvents.push({ msg: "Round " + currentRound, type: "tick" });
                lastRound = currentRound;
            }

            if (currentPhase && currentPhase !== lastPhase) {
                if (lastPhase !== "") {
                    tickEvents.push({ msg: "Phase: " + currentPhase, type: "tick" });
                }
                
                // When entering DISCUSSION, mark all eliminated agents as cleared
                if (currentPhase === "DISCUSSION") {
                    Object.keys(data.agents).forEach(function(agentKey) {
                        const agent = data.agents[agentKey];
                        if (agent.status === "eliminated") {
                            clearedAgents.add(agentKey);
                        }
                    });
                }
                
                if (currentPhase === "MOVEMENT" && lastPhase !== "") {
                    closeDiscussionChat();
                }
                lastPhase = currentPhase;
            }

            var events = [];
            if (Array.isArray(data.events)) {
                events = data.events;
            } else if (data.global && Array.isArray(data.global.ui_event_log)) {
                events = data.global.ui_event_log;
            }

            tickEvents.forEach(function(te) {
                addFeedTick(te.msg);
            });

            updateLiveFeed(events);
        }
    } catch (error) {
        console.error("Error:", error);
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
    
    const exitBtn = document.getElementById("exitBtn");
    if (exitBtn) exitBtn.addEventListener("click", exitToHome);
    
    console.log("🎮 Press 'D' for debug!");
});

function showWinScreen(winner) {
    const winScreen = document.getElementById("winScreen");
    const winImage = document.getElementById("winImage");
    
    if (!winScreen || !winImage) return;
    
    if (winner === "honest") {
        winImage.src = "/static/images/HonestWin.png";
    } else if (winner === "byzantine") {
        winImage.src = "/static/images/ByzantineWin.png";
    }
    
    winScreen.style.display = "flex";
}

function closeWinScreen() {
    const winScreen = document.getElementById("winScreen");
    if (winScreen) {
        winScreen.style.display = "none";
    }
}

function addFeedTick(text) {
    const feedContent = document.getElementById("feedContent");
    if (!feedContent) return;
    if (feedContent.children.length === 1) {
        const first = feedContent.querySelector(".feed-text");
        if (first && first.textContent.includes("No events yet")) {
            feedContent.innerHTML = "";
        }
    }
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, "0");
    const mm = String(now.getMinutes()).padStart(2, "0");
    const ss = String(now.getSeconds()).padStart(2, "0");
    const time = hh + ":" + mm + ":" + ss;
    const div = document.createElement("div");
    div.className = "feed-event feed-event--tick";
    div.innerHTML = '<span class="feed-timestamp">[' + time + ']</span><span class="feed-text">' + text + '</span>';
    feedContent.appendChild(div);
    const atBottom = feedContent.scrollHeight - feedContent.scrollTop - feedContent.clientHeight < 80;
    if (atBottom) feedContent.scrollTop = feedContent.scrollHeight;
}

var hardcodedChatColors = ["red","orange","yellow","lime","green","cyan","blue","purple","brown","pink","white","black"];

function showDiscussionChat(title) {
    const chat = document.getElementById("discussionChat");
    const titleEl = document.getElementById("discussionChatTitle");
    const msgs = document.getElementById("discussionChatMessages");
    if (!chat) return;
    if (titleEl && title) titleEl.textContent = title;
    if (msgs) msgs.innerHTML = "";
    chat.style.display = "flex";
}

function openDiscussionChat() {
    const chat = document.getElementById("discussionChat");
    if (chat) chat.style.display = "flex";
}

function closeDiscussionChat() {
    const chat = document.getElementById("discussionChat");
    if (chat) chat.style.display = "none";
}

function addDiscussionMessage(event) {
    const msgs = document.getElementById("discussionChatMessages");
    if (!msgs) return;

    const msg = event.msg || event.text || "";
    const time = event.time || event.timestamp || "--:--";
    const isVote = /Agent_\d+\s+voted/.test(msg);

    let agentName = "", messageText = msg;
    const colonMatch = msg.match(/^(Agent_\d+)\s*:\s*(.*)/s);
    const voteMatch  = msg.match(/^(Agent_\d+)\s+voted/);
    if (colonMatch) {
        agentName = colonMatch[1];
        messageText = colonMatch[2].trim();
    } else if (voteMatch) {
        agentName = voteMatch[1];
    }

    const agentIndex = agentName ? parseInt(agentName.replace("Agent_", "")) : 0;
    const colorSlug = hardcodedChatColors[agentIndex] || "red";
    const spriteUrl = LIVING_SPRITES[colorSlug] || LIVING_SPRITES.red;

    const div = document.createElement("div");
    div.className = isVote ? "chat-message vote" : "chat-message";
    
    if (isVote) {
        const boldMsg = msg.replace(/(Agent_\d+)/g, '<strong>$1</strong>');
        div.innerHTML =
            '<div class="chat-message-content">' +
                '<div class="chat-message-text">' + boldMsg + '</div>' +
            '</div>';
    } else {
        div.innerHTML =
            '<div class="chat-message-sprite"><img src="' + spriteUrl + '" alt="' + agentName + '"></div>' +
            '<div class="chat-message-content">' +
                '<div class="chat-message-header">' +
                    '<span class="chat-message-name">' + agentName + '</span>' +
                    '<span class="chat-message-time">[' + time + ']</span>' +
                '</div>' +
                '<div class="chat-message-text">' + messageText + '</div>' +
            '</div>';
    }
    
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}