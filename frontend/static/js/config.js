// Available models from backend
const MODELS = [
    { value: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", name: "TinyLlama 1.1B" },
    { value: "Qwen/Qwen2.5-1.5B-Instruct", name: "Qwen 1.5B" },
    { value: "meta-llama/Llama-3.2-3B-Instruct", name: "Llama 3.2 3B" },
    { value: "meta-llama/Llama-3.3-70B-Instruct", name: "Llama 3.3 70B" },
    { value: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", name: "DeepSeek R1 70B" },
    { value: "zerofata/L3.3-GeneticLemonade-Final-v2-70B", name: "GeneticLemonade 70B" },
    { value: "NousResearch/Hermes-4-70B", name: "Hermes 4 70B" },
    { value: "Qwen/Qwen2.5-72B-Instruct", name: "Qwen 2.5 72B" },
    { value: "Qwen/Qwen3-Next-80B-A3B-Instruct", name: "Qwen 3 80B" },
    { value: "arcee-ai/Arcee-Nova", name: "Arcee Nova 73B" }
];

// Available colors (Among Us crew colors)
const COLORS = [
    { value: "red", name: "🔴 Red", hex: "#C51111" },
    { value: "blue", name: "🔵 Blue", hex: "#132ED1" },
    { value: "green", name: "🟢 Green", hex: "#117F2D" },
    { value: "pink", name: "💗 Pink", hex: "#ED54BA" },
    { value: "orange", name: "🟠 Orange", hex: "#EF7D0D" },
    { value: "yellow", name: "🟡 Yellow", hex: "#F5F557" },
    { value: "black", name: "⚫ Black", hex: "#3F474E" },
    { value: "white", name: "⚪ White", hex: "#D6E0F0" },
    { value: "purple", name: "🟣 Purple", hex: "#6B2FBB" },
    { value: "brown", name: "🟤 Brown", hex: "#71491E" },
    { value: "cyan", name: "🔷 Cyan", hex: "#38FEDC" },
    { value: "lime", name: "🟩 Lime", hex: "#50EF39" }
];

function generateAgentTable() {
    const numAgents = parseInt(document.getElementById('num_agents').value);
    
    // Validate input
    if (numAgents < 4 || numAgents > 12) {
        alert('Number of agents must be between 4 and 12!');
        document.getElementById('num_agents').value = 4;
        return;
    }
    
    const tableBody = document.getElementById('agentTableBody');
    tableBody.innerHTML = ''; // Clear existing rows
    
    // Generate rows
    for (let i = 0; i < numAgents; i++) {
        const row = createAgentRow(i);
        tableBody.appendChild(row);
    }
    
    // Show table and settings sections
    document.getElementById('agentTableSection').style.display = 'block';
    document.getElementById('gameSettingsSection').style.display = 'block';
    document.getElementById('startBtn').disabled = false;
}

function createAgentRow(index) {
    const row = document.createElement('tr');
    
    // Agent number
    const numCell = document.createElement('td');
    numCell.textContent = index + 1;
    numCell.className = 'agent-num';
    row.appendChild(numCell);
    
    // Model selector
    const modelCell = document.createElement('td');
    const modelSelect = document.createElement('select');
    modelSelect.name = `agent_${index}_model`;
    modelSelect.className = 'table-select';
    modelSelect.required = true;
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select Model --';
    modelSelect.appendChild(defaultOption);
    
    // Add model options
    MODELS.forEach(model => {
        const option = document.createElement('option');
        option.value = model.value;
        option.textContent = model.name;
        modelSelect.appendChild(option);
    });
    
    // Set default to TinyLlama for easy testing
    if (index < 3) {
        modelSelect.value = MODELS[0].value;
    }
    
    modelCell.appendChild(modelSelect);
    row.appendChild(modelCell);
    
    // Role selector
    const roleCell = document.createElement('td');
    const roleSelect = document.createElement('select');
    roleSelect.name = `agent_${index}_role`;
    roleSelect.className = 'table-select role-select';
    roleSelect.required = true;
    
    const honestOption = document.createElement('option');
    honestOption.value = 'honest';
    honestOption.textContent = '✅ Honest';
    roleSelect.appendChild(honestOption);
    
    const byzOption = document.createElement('option');
    byzOption.value = 'byzantine';
    byzOption.textContent = '❌ Byzantine';
    roleSelect.appendChild(byzOption);
    
    // Set first 2 agents as byzantine by default
    if (index < 2) {
        roleSelect.value = 'byzantine';
    }
    
    roleCell.appendChild(roleSelect);
    row.appendChild(roleCell);
    
    // Color selector
    const colorCell = document.createElement('td');
    const colorSelect = document.createElement('select');
    colorSelect.name = `agent_${index}_color`;
    colorSelect.className = 'table-select color-select';
    colorSelect.required = true;
    
    COLORS.forEach((color, idx) => {
        if (idx < numAgents) { // Only show as many colors as agents
            const option = document.createElement('option');
            option.value = color.value;
            option.textContent = color.name;
            option.style.color = color.hex;
            colorSelect.appendChild(option);
        }
    });
    
    // Set default color (assign sequentially)
    if (index < COLORS.length) {
        colorSelect.value = COLORS[index].value;
    }
    
    colorCell.appendChild(colorSelect);
    row.appendChild(colorCell);
    
    return row;
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    // Generate default 4 agents automatically
    generateAgentTable();
});