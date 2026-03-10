/**
 * Game Simulation 
 * Press D for debug overlay. Grey = old room-to-room lines (obsolete). Orange = road nodes and edges (agents move on these).
 */

// origional room coords
const ROOM_COORDINATES = {
    "Reactor": { x: 12.4, y: 54.0 },
    "UpperEngine": { x: 17.4, y: 22.3 },
    "LowerEngine": { x: 25.3, y: 85.3 },
    "Security": { x: 36.3, y: 42.2 },
    "MedBay": { x: 30.3, y: 9.8 },
    "Electrical": { x: 36.2, y: 60.1 },
    "Cafeteria": { x: 49.3, y: 20.3 },
    "Admin": { x: 60.2, y: 53.0 },
    "Storage": { x: 49.3, y: 84.2 },
    "Weapons": { x: 78.9, y: 14.8 },
    "O2": { x: 66.0, y: 33.3 },
    "Navigation": { x: 91.2, y: 43.1 },
    "Shields": { x: 78.9, y: 68.2 },
    "Communications": { x: 79.0, y: 91.9 }
};

// box where sprites will be held in each room
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
    "Navigation": { width: 7, height: 14 },
    "Shields": { width: 7, height: 14 },
    "Communications": { width: 6, height: 12 }
};

// connections of rooms
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

// map nodes for roads (25 total nodes)
const ROAD_NODES = {
    "1":  { x: 30.2, y: 22.1 },
    "2":  { x: ROOM_COORDINATES["MedBay"].x,         y: ROOM_COORDINATES["MedBay"].y },
    "3":  { x: 27.2, y: 22.1 },
    "4":  { x: ROOM_COORDINATES["UpperEngine"].x,    y: ROOM_COORDINATES["UpperEngine"].y },
    "5":  { x: 27.2, y: 41.0 },
    "6":  { x: ROOM_COORDINATES["Security"].x,       y: ROOM_COORDINATES["Security"].y },
    "7":  { x: 27.2, y: 54.0 },
    "8":  { x: ROOM_COORDINATES["Reactor"].x,        y: ROOM_COORDINATES["Reactor"].y },
    "9":  { x: 27.2, y: 69.4 },  // corridor between 7 and 10 (same x as 7, a little below)
    "10": { x: ROOM_COORDINATES["LowerEngine"].x,    y: ROOM_COORDINATES["LowerEngine"].y },
    "11": { x: 36.2, y: 82.5 },
    "12": { x: ROOM_COORDINATES["Electrical"].x,     y: ROOM_COORDINATES["Electrical"].y },
    "13": { x: ROOM_COORDINATES["Storage"].x,        y: ROOM_COORDINATES["Storage"].y },
    "14": { x: 78.9, y: 82.4 },
    "15": { x: ROOM_COORDINATES["Communications"].x, y: ROOM_COORDINATES["Communications"].y },
    "16": { x: ROOM_COORDINATES["Shields"].x,        y: ROOM_COORDINATES["Shields"].y },
    "17": { x: 78.9, y: 43.1 },
    "18": { x: ROOM_COORDINATES["Navigation"].x,     y: ROOM_COORDINATES["Navigation"].y },
    "19": { x: 78.9, y: 33.0 },
    "20": { x: ROOM_COORDINATES["O2"].x,             y: ROOM_COORDINATES["O2"].y },
    "21": { x: ROOM_COORDINATES["Weapons"].x,        y: ROOM_COORDINATES["Weapons"].y },
    "22": { x: 60.9, y: 14.8 },
    "23": { x: ROOM_COORDINATES["Cafeteria"].x,      y: ROOM_COORDINATES["Cafeteria"].y },
    "24": { x: 49.3, y: 53 },
    "25": { x: ROOM_COORDINATES["Admin"].x,          y: ROOM_COORDINATES["Admin"].y }
};

// connections of nodes
const ROAD_EDGES = [
    ["1","2"], ["1","3"], ["1","23"],
    ["2","1"],
    ["3","1"], ["3","4"], ["3","5"],
    ["4","3"],
    ["5","3"], ["5","6"], ["5","7"],
    ["6","5"],
    ["7","5"], ["7","8"], ["7","9"],
    ["8","7"],
    ["9","7"], ["9","10"],
    ["10","9"], ["10","11"], ["10","13"],
    ["11","10"], ["11","12"], ["11","13"],
    ["12","11"],
    ["13","11"], ["13","14"], ["13","24"],
    ["14","13"], ["14","15"], ["14","16"],
    ["15","14"],
    ["16","14"], ["16","17"],
    ["17","16"], ["17","18"], ["17","19"],
    ["18","17"],
    ["19","17"], ["19","20"], ["19","21"],
    ["20","19"],
    ["21","19"], ["21","22"],
    ["22","21"], ["22","23"],
    ["23","1"], ["23","22"], ["23","24"],  
    ["24","13"], ["24","23"], ["24","25"],
    ["25","24"]
];

// room nodes to room names
const ROOM_TO_NODE = {
    "MedBay": "2", "UpperEngine": "4", "Security": "6", "Reactor": "8", "LowerEngine": "10",
    "Electrical": "12", "Storage": "13", "Communications": "15", "Shields": "16",
    "Navigation": "18", "O2": "20", "Weapons": "21", "Cafeteria": "23", "Admin": "25"
};

const NODE_LABELS = {};
for (let i = 1; i <= 25; i++) NODE_LABELS[String(i)] = i;

let debugOverlayVisible = false;
let debugCanvas = null;

// debug overlay (press D during simulation)
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

// Convert a node (percent coords) to pixels using current map dimensions
function nodeToPixels(nodePercent) {
    if (!nodePercent) return null;
    const dims = getImageDimensions();
    if (!dims) return null;
    return {
        x: dims.imgLeft + (nodePercent.x / 100) * dims.imgWidth,
        y: dims.imgTop + (nodePercent.y / 100) * dims.imgHeight
    };
}

// Build adjacency list from ROAD_EDGES with distances (for Dijkstra)
function buildRoadGraph() {
    const adj = {};
    function addEdge(a, b, dist) {
        if (!adj[a]) adj[a] = [];
        adj[a].push({ id: b, dist: dist });
    }
    ROAD_EDGES.forEach(([a, b]) => {
        const na = ROAD_NODES[a];
        const nb = ROAD_NODES[b];
        if (!na || !nb) return;
        const dx = na.x - nb.x;
        const dy = na.y - nb.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        addEdge(a, b, dist);
        addEdge(b, a, dist);
    });
    return adj;
}

function dijkstraShortestPath(adj, startId, endId) {
    const dist = {};
    const prev = {};
    const q = new Set(Object.keys(adj));
    Object.keys(adj).forEach(id => { dist[id] = id === startId ? 0 : Infinity; prev[id] = null; });
    while (q.size > 0) {
        let u = null;
        let best = Infinity;
        q.forEach(id => {
            if (dist[id] < best) { best = dist[id]; u = id; }
        });
        if (u === null || u === endId) break;
        q.delete(u);
        (adj[u] || []).forEach(({ id: v, dist: w }) => {
            const alt = dist[u] + w;
            if (alt < dist[v]) { dist[v] = alt; prev[v] = u; }
        });
    }
    const path = [];
    for (let cur = endId; cur != null; cur = prev[cur]) path.unshift(cur);
    return path.length > 0 && path[0] === startId ? path : null;
}

// Returns array of { x, y } in pixels for the road path from roomA to roomB, or null
function getRoadPathForRooms(roomA, roomB) {
    const startId = ROOM_TO_NODE[roomA];
    const endId = ROOM_TO_NODE[roomB];
    if (!startId || !endId || startId === endId) return null;
    const adj = buildRoadGraph();
    const pathIds = dijkstraShortestPath(adj, startId, endId);
    if (!pathIds || pathIds.length < 2) return null;
    const dims = getImageDimensions();
    if (!dims) return null;
    return pathIds.map(id => {
        const n = ROAD_NODES[id];
        return n ? nodeToPixels(n) : null;
    }).filter(Boolean);
}

const MOVE_DURATION_MS = 1500;
const STAGGER_MS = 220;  // delay between agents on same path so they move in a line instead of overlapping
let agentPreviousRoom = {};
let animatingAgents = new Set();
let agentPathKey = {};   // also for stagger

function animateAlongPath(agentKey, marker, waypointsPx, durationMs, onDone) {
    if (!marker || !waypointsPx || waypointsPx.length < 2) {
        if (onDone) onDone();
        return;
    }
    animatingAgents.add(agentKey);
    const segmentCount = waypointsPx.length - 1;
    const durationPerSegment = durationMs / segmentCount;
    let i = 0;
    function runSegment() {
        if (i >= segmentCount) {
            animatingAgents.delete(agentKey);
            const last = waypointsPx[waypointsPx.length - 1];
            if (last && agentPositions) agentPositions[agentKey] = { x: last.x, y: last.y };
            if (onDone) onDone();
            return;
        }
        const from = waypointsPx[i];
        const to = waypointsPx[i + 1];
        i++;
        marker.style.transition = "none";
        marker.style.left = from.x + "px";
        marker.style.top = from.y + "px";
        marker.offsetHeight;
        marker.style.transition = "left " + durationPerSegment + "ms linear, top " + durationPerSegment + "ms linear, transform 0.2s ease";
        marker.style.left = to.x + "px";
        marker.style.top = to.y + "px";
        setTimeout(runSegment, durationPerSegment);
    }
    runSegment();
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
    
    // orignal direct room-to-room lines (greyed out) 
    ctx.strokeStyle = "#888888";
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

    // room boxes and magenta centers for each room (room nodes overlay the magenta dots now)
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

    // Road graph: edges (orange) – agents move only on these
    ctx.strokeStyle = "#FF8C00";
    ctx.lineWidth = 2;
    ROAD_EDGES.forEach(([id1, id2]) => {
        const n1 = ROAD_NODES[id1];
        const n2 = ROAD_NODES[id2];
        if (!n1 || !n2) return;
        const p1 = { x: dims.imgLeft + (n1.x / 100) * dims.imgWidth, y: dims.imgTop + (n1.y / 100) * dims.imgHeight };
        const p2 = { x: dims.imgLeft + (n2.x / 100) * dims.imgWidth, y: dims.imgTop + (n2.y / 100) * dims.imgHeight };
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
    });

    // Road nodes: circles and numbers (orange) – drawn last so room nodes overlay the purple dots
    Object.keys(ROAD_NODES).forEach(nodeId => {
        const n = ROAD_NODES[nodeId];
        const px = dims.imgLeft + (n.x / 100) * dims.imgWidth;
        const py = dims.imgTop + (n.y / 100) * dims.imgHeight;
        const label = (typeof NODE_LABELS !== "undefined" && NODE_LABELS[nodeId]) ? String(NODE_LABELS[nodeId]) : "?";
        ctx.fillStyle = "#FF8C00";
        ctx.strokeStyle = "#CC6600";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.font = "bold 11px monospace";
        ctx.fillStyle = "#000000";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, px, py);
    });
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    
    ctx.fillStyle = "rgba(0, 0, 0, 0.8)";
    ctx.fillRect(10, 10, 280, 160);
    ctx.font = "14px Arial";
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText("DEBUG (Press 'D')", 20, 30);
    ctx.font = "12px Arial";
    ctx.fillStyle = "#FF0000"; ctx.fillText("Red: Map boundary", 20, 55);
    ctx.fillStyle = "#888888"; ctx.fillText("Grey: Obsolete room lines", 20, 75);
    ctx.fillStyle = "#FF8C00"; ctx.fillText("Orange: Road nodes & edges (agent paths)", 20, 95);
    ctx.fillStyle = "#FFFF00"; ctx.fillText("Yellow: Housing boxes", 20, 115);
    ctx.fillStyle = "#FF00FF"; ctx.fillText("Magenta: Room centers", 20, 135);
}

function toggleDebug() {
    debugOverlayVisible = !debugOverlayVisible;
    if (debugCanvas) debugCanvas.style.display = debugOverlayVisible ? "block" : "none";
    drawDebugOverlay();
    console.log(debugOverlayVisible ? "Debug ON" : "Debug OFF");
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
    if (modelName.indexOf(":") !== -1) {
        var parts = modelName.split(":");
        var provider = parts[0];
        var modelId = parts[1];
        var tag = provider.charAt(0).toUpperCase() + provider.slice(1);
        return tag + "/" + modelId.substring(0, 15);
    }
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
let agentPositions = {};
let pollingInterval = null;
let lastPhase = "";
let lastRound = 0;
let clearedAgents = new Set();
let currentTokenUsage = {};

// SUSPICION SCORES 
let enabledClassifiers = {
    sgd: false,
    svm: false,
    lr: false
};
let suspicionInitialized = false;

function initSuspicionTracking(data) {
    if (suspicionInitialized) return;
    
    // Check if suspicion data exists in game state
    if (data && data.suspicion && data.suspicion.enabled_classifiers) {
        enabledClassifiers = data.suspicion.enabled_classifiers;
        setupClassifierColumns();
        suspicionInitialized = true;
    }
}

function setupClassifierColumns() {
    // Show/hide column headers
    const sgdHeader = document.getElementById('sgdHeader');
    const svmHeader = document.getElementById('svmHeader');
    const lrHeader = document.getElementById('lrHeader');
    
    if (sgdHeader) sgdHeader.style.display = enabledClassifiers.sgd ? '' : 'none';
    if (svmHeader) svmHeader.style.display = enabledClassifiers.svm ? '' : 'none';
    if (lrHeader) lrHeader.style.display = enabledClassifiers.lr ? '' : 'none';
}

function updateSuspicionScores(suspicionData) {
    if (!suspicionData || !suspicionData.scores) return;
    
    const scores = suspicionData.scores;
    
    // Update only the score cells in existing rows
    Object.keys(scores).forEach(agentKey => {
        const agentScores = scores[agentKey];
        const agentNum = agentKey.replace('Agent_', '');
        
        // Update SGD
        if (enabledClassifiers.sgd) {
            const sgdCell = document.getElementById(`suspicion-${agentNum}-sgd`);
            if (sgdCell && agentScores.SGD !== undefined && agentScores.SGD !== null) {
                sgdCell.textContent = agentScores.SGD.toFixed(2);
                sgdCell.className = getSuspicionClass(agentScores.SGD);
            }
        }
        
        // Update SVM
        if (enabledClassifiers.svm) {
            const svmCell = document.getElementById(`suspicion-${agentNum}-svm`);
            if (svmCell && agentScores.SVM !== undefined && agentScores.SVM !== null) {
                svmCell.textContent = agentScores.SVM.toFixed(2);
                svmCell.className = getSuspicionClass(agentScores.SVM);
            }
        }
        
        // Update LR
        if (enabledClassifiers.lr) {
            const lrCell = document.getElementById(`suspicion-${agentNum}-lr`);
            if (lrCell && agentScores.LogisticRegression !== undefined && agentScores.LogisticRegression !== null) {
                lrCell.textContent = agentScores.LogisticRegression.toFixed(2);
                lrCell.className = getSuspicionClass(agentScores.LogisticRegression);
            }
        }
    });
}

function getSuspicionClass(score) {
    if (score === undefined || score === null) return '';
    if (score >= 0.7) return 'suspicion-high';
    if (score >= 0.4) return 'suspicion-medium';
    return 'suspicion-low';
}

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

    // Track which agents are still present this tick
    const seenAgents = new Set();

    const agentsByRoom = {};
    Object.keys(agents).forEach(function(agentKey) {
        const agent = agents[agentKey];
        if (!agent.location) return;
        
        if (agent.status === "ejected") {
            clearedAgents.add(agentKey);
            return;
        }
        
        if (clearedAgents.has(agentKey)) {
            return;
        }
        
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

            let marker = agentMarkers[agentKey];

            // Create marker if it doesn't exist yet
            if (!marker) {
                marker = document.createElement("div");
                marker.className = "agent-marker";
                marker.id = "marker-" + agentNumRaw;
                const img = document.createElement("img");
                img.className = "agent-marker-sprite";
                img.alt = "Agent " + displayNum;
                marker.appendChild(img);
                markersContainer.appendChild(marker);
                agentMarkers[agentKey] = marker;
            }

            // Update sprite and title / alive-dead styling
            const imgEl = marker.querySelector(".agent-marker-sprite");
            if (imgEl && imgEl.src !== spriteUrl) {
                imgEl.src = spriteUrl;
            }
            marker.title = "Agent " + displayNum + " - " + roomName + (isAlive ? " (Alive)" : " (Dead Body)");
            if (!isAlive) {
                marker.classList.add("agent-marker-dead");
            } else {
                marker.classList.remove("agent-marker-dead");
            }

            const prevRoom = agentPreviousRoom[agentKey];
            const roomChanged = prevRoom != null && prevRoom !== roomName;

            if (animatingAgents.has(agentKey)) {
                seenAgents.add(agentKey);
                return;
            }

            if (roomChanged && isAlive && typeof getRoadPathForRooms === "function") {
                const waypointsPx = getRoadPathForRooms(prevRoom, roomName);
                if (waypointsPx && waypointsPx.length >= 2) {
                    const pathKey = prevRoom + "|" + roomName;
                    let samePathCount = 0;
                    Object.keys(agentPathKey).forEach(function(k) {
                        if (agentPathKey[k] === pathKey) samePathCount++;
                    });
                    const delayMs = samePathCount * STAGGER_MS;
                    agentPreviousRoom[agentKey] = roomName;
                    animatingAgents.add(agentKey);
                    function startMove() {
                        agentPathKey[agentKey] = pathKey;
                        animateAlongPath(agentKey, marker, waypointsPx, MOVE_DURATION_MS, function() {
                            delete agentPathKey[agentKey];
                        });
                    }
                    if (delayMs > 0) {
                        setTimeout(startMove, delayMs);
                    } else {
                        startMove();
                    }
                    seenAgents.add(agentKey);
                    return;
                }
            }

            agentPreviousRoom[agentKey] = roomName;

            const prevPos = agentPositions[agentKey];
            marker.style.left = (prevPos ? prevPos.x : x) + "px";
            marker.style.top = (prevPos ? prevPos.y : y) + "px";

            marker.offsetHeight;

            marker.style.left = x + "px";
            marker.style.top = y + "px";

            agentPositions[agentKey] = { x: x, y: y };
            seenAgents.add(agentKey);
        });
    });

    // Remove markers for agents that disappeared this tick
    Object.keys(agentMarkers).forEach(function(key) {
        if (!seenAgents.has(key)) {
            const marker = agentMarkers[key];
            if (marker && marker.parentElement === markersContainer) {
                markersContainer.removeChild(marker);
            }
            delete agentMarkers[key];
            delete agentPositions[key];
            delete agentPreviousRoom[key];
            delete agentPathKey[key];
            animatingAgents.delete(key);
        }
    });
}

function updateStatusTable(agents) {
    if (!agents) return;
    const tbody = document.getElementById("statusTableBody");
    if (!tbody) return;
    
    const cachedScores = {};
    tbody.querySelectorAll('tr').forEach(row => {
        const cells = row.querySelectorAll('td');
        if (cells.length > 0) {
            const agentNum = cells[0].textContent;
            cachedScores[agentNum] = {};
            
            if (enabledClassifiers.sgd) {
                const sgdCell = row.querySelector(`[id$="-sgd"]`);
                if (sgdCell && sgdCell.textContent !== "-") {
                    cachedScores[agentNum].sgd = {
                        value: sgdCell.textContent,
                        className: sgdCell.className
                    };
                }
            }
            
            if (enabledClassifiers.svm) {
                const svmCell = row.querySelector(`[id$="-svm"]`);
                if (svmCell && svmCell.textContent !== "-") {
                    cachedScores[agentNum].svm = {
                        value: svmCell.textContent,
                        className: svmCell.className
                    };
                }
            }
            
            if (enabledClassifiers.lr) {
                const lrCell = row.querySelector(`[id$="-lr"]`);
                if (lrCell && lrCell.textContent !== "-") {
                    cachedScores[agentNum].lr = {
                        value: lrCell.textContent,
                        className: lrCell.className
                    };
                }
            }
        }
    });
    
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
        var modelLabel = getModelAbbreviation(modelName);
        if (modelName && modelName.indexOf(":") !== -1 && currentTokenUsage[modelName]) {
            var tokens = currentTokenUsage[modelName];
            modelLabel += " (" + (tokens.input_tokens + tokens.output_tokens) + "t)";
        }
        modelCell.textContent = modelLabel;
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
        
        if (enabledClassifiers.sgd) {
            const sgdCell = document.createElement("td");
            sgdCell.id = `suspicion-${agentNumRaw}-sgd`;
            
            // Restore cached value if exists
            if (cachedScores[displayNum] && cachedScores[displayNum].sgd) {
                sgdCell.textContent = cachedScores[displayNum].sgd.value;
                sgdCell.className = cachedScores[displayNum].sgd.className;
            } else {
                sgdCell.textContent = "-";
            }
            row.appendChild(sgdCell);
        }
        
        if (enabledClassifiers.svm) {
            const svmCell = document.createElement("td");
            svmCell.id = `suspicion-${agentNumRaw}-svm`;
            
            // Restore cached value if exists
            if (cachedScores[displayNum] && cachedScores[displayNum].svm) {
                svmCell.textContent = cachedScores[displayNum].svm.value;
                svmCell.className = cachedScores[displayNum].svm.className;
            } else {
                svmCell.textContent = "-";
            }
            row.appendChild(svmCell);
        }
        
        if (enabledClassifiers.lr) {
            const lrCell = document.createElement("td");
            lrCell.id = `suspicion-${agentNumRaw}-lr`;
            
            // Restore cached value if exists
            if (cachedScores[displayNum] && cachedScores[displayNum].lr) {
                lrCell.textContent = cachedScores[displayNum].lr.value;
                lrCell.className = cachedScores[displayNum].lr.className;
            } else {
                lrCell.textContent = "-";
            }
            row.appendChild(lrCell);
        }
        
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
        if (eventType === "vote") {
            eventDiv.classList.add("feed-event--vote");
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
        
        if (!suspicionInitialized) {
            initSuspicionTracking(data);
        }
        
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
                console.log("Game ended, stopping polling");
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
        
        if (data.token_usage) {
            currentTokenUsage = data.token_usage;
        }

        if (data.agents && Object.keys(data.agents).length > 0) {
            updateAgentPositions(data.agents);
            updateStatusTable(data.agents);

            if (data.suspicion && data.suspicion.scores) {
                updateSuspicionScores(data.suspicion);
            }
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
    
    initializeDraggableChat();
    
    console.log("Press 'D' for debug!");
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
function initializeDraggableChat() {
    const chat = document.getElementById("discussionChat");
    const header = document.getElementById("discussionChatHeader");
    const resizeHandle = chat.querySelector(".discussion-chat-resize-handle");
    
    if (!chat || !header) return;
    
    // Draggable functionality
    let isDragging = false;
    let currentX, currentY, initialX, initialY;
    
    header.addEventListener("mousedown", function(e) {
        // Don't drag if clicking close button
        if (e.target.classList.contains("discussion-chat-close")) return;
        
        isDragging = true;
        
        // Get current transform or use default center position
        const rect = chat.getBoundingClientRect();
        initialX = rect.left;
        initialY = rect.top;
        currentX = e.clientX - initialX;
        currentY = e.clientY - initialY;
        
        // Remove transform centering when starting to drag
        chat.style.transform = "none";
        chat.style.left = initialX + "px";
        chat.style.top = initialY + "px";
        
        e.preventDefault();
    });
    
    document.addEventListener("mousemove", function(e) {
        if (!isDragging) return;
        
        e.preventDefault();
        
        const newX = e.clientX - currentX;
        const newY = e.clientY - currentY;
        
        chat.style.left = newX + "px";
        chat.style.top = newY + "px";
    });
    
    document.addEventListener("mouseup", function() {
        isDragging = false;
    });
    
    // Resizable functionality
    if (resizeHandle) {
        let isResizing = false;
        let startX, startY, startWidth, startHeight;
        
        resizeHandle.addEventListener("mousedown", function(e) {
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            
            const rect = chat.getBoundingClientRect();
            startWidth = rect.width;
            startHeight = rect.height;
            
            e.preventDefault();
            e.stopPropagation();
        });
        
        document.addEventListener("mousemove", function(e) {
            if (!isResizing) return;
            
            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;
            
            const newWidth = Math.max(300, startWidth + deltaX);
            const newHeight = Math.max(200, startHeight + deltaY);
            
            chat.style.width = newWidth + "px";
            chat.style.height = newHeight + "px";
        });
        
        document.addEventListener("mouseup", function() {
            isResizing = false;
        });
    }
}