/**
 * Game Simulation Page - Real-time updates and display
 * Polls /api/game_state every 2 seconds and updates the UI
 */

// ============================================================================
// ROOM COORDINATES
// Map room names to X/Y pixel positions on the map image
// These will need to be adjusted based on the actual map image dimensions
// ============================================================================

const ROOM_COORDINATES = {
    "Cafeteria": { x: 400, y: 300 },
    "Weapons": { x: 100, y: 150 },
    "O2": { x: 200, y: 100 },
    "Navigation": { x: 300, y: 120 },
    "Shields": { x: 350, y: 200 },
    "Communications": { x: 450, y: 400 },
    "Storage": { x: 500, y: 350 },
    "Admin": { x: 420, y: 280 },
    "Electrical": { x: 550, y: 450 },
    "LowerEngine": { x: 600, y: 500 },
    "UpperEngine": { x: 150, y: 400 },
    "Security": { x: 180, y: 480 },
    "Reactor": { x: 100, y: 550 },
    "MedBay": { x: 250, y: 350 }
};

// Color mapping for agent markers
const COLOR_MAP = {
    "red": "#C51111",
    "orange": "#EF7D0D",
    "yellow": "#F5F557",
    "lime": "#50EF39",
    "green": "#117F2D",
    "cyan": "#38FEDC",
    "blue": "#132ED1",
    "purple": "#6B2FBB",
    "brown": "#71491E",
    "pink": "#ED54BA",
    "white": "#D6E0F0",
    "black": "#3F474E"
};

// Model name abbreviations
function getModelAbbreviation(modelName) {
    if (!modelName) return "Unknown";
    if (modelName.includes("TinyLlama")) return "TinyLlama";
    if (modelName.includes("Qwen")) {
        if (modelName.includes("1.5B")) return "Qwen 1.5B";
        if (modelName.includes("72B")) return "Qwen 72B";
        if (modelName.includes("80B")) return "Qwen 80B";
        return "Qwen";
    }
    if (modelName.includes("Llama-3.2")) return "Llama 3.2";
    if (modelName.includes("Llama-3.3")) return "Llama 3.3";
    if (modelName.includes("DeepSeek")) return "DeepSeek";
    if (modelName.includes("GeneticLemonade")) return "Genetic";
    if (modelName.includes("Hermes")) return "Hermes";
    if (modelName.includes("Arcee")) return "Arcee";
    return modelName.split("/").pop().substring(0, 10);
}

// ============================================================================
// STATE MANAGEMENT
// ============================================================================

let lastEventCount = 0;
let agentMarkers = {};
let isPaused = false;
let pollingInterval = null;

// ============================================================================
// UPDATE FUNCTIONS
// ============================================================================

/**
 * Updates game parameters in the header
 */
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

/**
 * Updates agent positions on the map
 */
function updateAgentPositions(agents) {
    if (!agents) return;
    
    const markersContainer = document.getElementById("agentMarkers");
    if (!markersContainer) return;
    
    const mapImage = document.getElementById("mapImage");
    if (!mapImage) return;
    
    // Get map dimensions
    const mapRect = mapImage.getBoundingClientRect();
    const mapWidth = mapImage.naturalWidth || mapRect.width;
    const mapHeight = mapImage.naturalHeight || mapRect.height;
    
    // Clear existing markers
    markersContainer.innerHTML = "";
    agentMarkers = {};
    
    // Create markers for each agent
    Object.keys(agents).forEach(function(agentKey) {
        const agent = agents[agentKey];
        if (!agent.location) return;
        
        const roomCoords = ROOM_COORDINATES[agent.location];
        if (!roomCoords) {
            console.warn("Unknown room:", agent.location);
            return;
        }
        
        // Calculate position relative to displayed map size
        const scaleX = mapRect.width / mapWidth;
        const scaleY = mapRect.height / mapHeight;
        const x = roomCoords.x * scaleX;
        const y = roomCoords.y * scaleY;
        
        // Extract agent number from key (e.g., "Agent_0" -> 0)
        const agentNum = agentKey.replace("Agent_", "");
        const color = COLOR_MAP[agent.color] || "#000000";
        
        // Create marker element
        const marker = document.createElement("div");
        marker.className = "agent-marker";
        marker.id = "marker-" + agentNum;
        marker.style.backgroundColor = color;
        marker.style.left = x + "px";
        marker.style.top = y + "px";
        marker.textContent = agentNum;
        marker.title = agentKey + " - " + agent.location;
        
        // Add status class if dead
        if (agent.status === "eliminated" || agent.status === "voted_off") {
            marker.style.opacity = "0.5";
        }
        
        markersContainer.appendChild(marker);
        agentMarkers[agentKey] = marker;
    });
}

/**
 * Updates the agent status table
 */
function updateStatusTable(agents) {
    if (!agents) return;
    
    const tbody = document.getElementById("statusTableBody");
    if (!tbody) return;
    
    tbody.innerHTML = "";
    
    // Sort agents by number
    const agentKeys = Object.keys(agents).sort(function(a, b) {
        const numA = parseInt(a.replace("Agent_", ""));
        const numB = parseInt(b.replace("Agent_", ""));
        return numA - numB;
    });
    
    agentKeys.forEach(function(agentKey) {
        const agent = agents[agentKey];
        const agentNum = agentKey.replace("Agent_", "");
        
        const row = document.createElement("tr");
        
        // Agent number
        const numCell = document.createElement("td");
        numCell.textContent = agentNum;
        row.appendChild(numCell);
        
        // Model (abbreviated)
        const modelCell = document.createElement("td");
        modelCell.textContent = getModelAbbreviation(agent.model);
        row.appendChild(modelCell);
        
        // Color indicator
        const colorCell = document.createElement("td");
        const colorIndicator = document.createElement("span");
        colorIndicator.className = "agent-color-indicator";
        colorIndicator.style.backgroundColor = COLOR_MAP[agent.color] || "#000000";
        colorCell.appendChild(colorIndicator);
        row.appendChild(colorCell);
        
        // Status
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
        
        // Votes received
        const votesCell = document.createElement("td");
        votesCell.textContent = agent.votes_received || 0;
        row.appendChild(votesCell);
        
        // Kills (only for Byzantine)
        const killsCell = document.createElement("td");
        if (agent.role === "byzantine") {
            killsCell.textContent = agent.eliminations || 0;
        } else {
            killsCell.textContent = "n/a";
        }
        row.appendChild(killsCell);
        
        tbody.appendChild(row);
    });
}

/**
 * Updates the live feed with new events
 */
function updateLiveFeed(events) {
    if (!events || !Array.isArray(events)) return;
    
    const feedContent = document.getElementById("feedContent");
    if (!feedContent) return;
    
    // Only add new events (after lastEventCount)
    const newEvents = events.slice(lastEventCount);
    
    newEvents.forEach(function(event) {
        const eventDiv = document.createElement("div");
        eventDiv.className = "feed-event";
        
        const timestampSpan = document.createElement("span");
        timestampSpan.className = "feed-timestamp";
        timestampSpan.textContent = "[" + (event.timestamp || "Unknown") + "]";
        
        const textSpan = document.createElement("span");
        textSpan.className = "feed-text";
        textSpan.textContent = event.text || "";
        
        eventDiv.appendChild(timestampSpan);
        eventDiv.appendChild(textSpan);
        feedContent.appendChild(eventDiv);
    });
    
    // Update last event count
    lastEventCount = events.length;
    
    // Auto-scroll to bottom
    feedContent.scrollTop = feedContent.scrollHeight;
}

// ============================================================================
// MAIN UPDATE FUNCTION
// ============================================================================

/**
 * Fetches game state from API and updates all UI components
 */
async function updateGameState() {
    try {
        const response = await fetch("/api/game_state");
        if (!response.ok) {
            console.error("Failed to fetch game state:", response.status);
            return;
        }
        
        const data = await response.json();
        
        // Update game parameters
        if (data.game_info) {
            updateGameParams(data.game_info);
        }
        
        // Update agent positions on map
        if (data.agents) {
            updateAgentPositions(data.agents);
        }
        
        // Update status table
        if (data.agents) {
            updateStatusTable(data.agents);
        }
        
        // Update live feed
        if (data.events) {
            updateLiveFeed(data.events);
        }
        
    } catch (error) {
        console.error("Error updating game state:", error);
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

// ============================================================================
// CONTROL BUTTON HANDLERS
// ============================================================================

/**
 * Toggles pause/resume for the simulation
 */
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
        // Resume polling
        if (!pollingInterval) {
            pollingInterval = setInterval(updateGameState, 2000);
        }
    }
}

/**
 * Reverses to the last round (calls backend API)
 */
async function reverseToLastRound() {
    if (confirm("Are you sure you want to go back to the last round?")) {
        try {
            const response = await fetch("/api/reverse_round", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                }
            });
            
            if (response.ok) {
                // Refresh game state
                await updateGameState();
                alert("Reversed to last round.");
            } else {
                alert("Failed to reverse round. Please try again.");
            }
        } catch (error) {
            console.error("Error reversing round:", error);
            alert("Error reversing round. Please try again.");
        }
    }
}

/**
 * Exits to home page (hard exit)
 */
function exitToHome() {
    if (confirm("Are you sure you want to exit? This will stop the simulation.")) {
        window.location.href = "/";
    }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

// Start polling when page loads
window.addEventListener("DOMContentLoaded", function() {
    // Initial update
    updateGameState();
    
    // Poll every 2 seconds
    pollingInterval = setInterval(updateGameState, 2000);
    
    // Set up control buttons
    const pauseBtn = document.getElementById("pauseBtn");
    const reverseBtn = document.getElementById("reverseBtn");
    const exitBtn = document.getElementById("exitBtn");
    
    if (pauseBtn) {
        pauseBtn.addEventListener("click", togglePause);
    }
    
    if (reverseBtn) {
        reverseBtn.addEventListener("click", reverseToLastRound);
    }
    
    if (exitBtn) {
        exitBtn.addEventListener("click", exitToHome);
    }
    
    console.log("Game simulation page loaded. Polling /api/game_state every 2 seconds.");
});
