/**
 * Configuration page logic for Among Us LLM simulation.
 * Generates agent table, validates inputs, and handles form behavior.
 */

// ---------------------------------------------------------------------------
// Model options (value = backend model id, name = display label)
// Ordered from most parameters to least parameters.
// ---------------------------------------------------------------------------
const MODELS = [
    { value: "Qwen/Qwen3-Next-80B-A3B-Instruct", name: "Qwen 3 80B (Qwen/Qwen3-Next-80B-A3B-Instruct)" },
    { value: "arcee-ai/Arcee-Nova", name: "Arcee Nova 73B (arcee-ai/Arcee-Nova)" },
    { value: "Qwen/Qwen2.5-72B-Instruct", name: "Qwen 2.5 72B (Qwen/Qwen2.5-72B-Instruct)" },
    { value: "meta-llama/Llama-3.3-70B-Instruct", name: "Llama 3.3 70B (meta-llama/Llama-3.3-70B-Instruct)" },
    { value: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", name: "DeepSeek R1 70B (deepseek-ai/DeepSeek-R1-Distill-Llama-70B)" },
    { value: "zerofata/L3.3-GeneticLemonade-Final-v2-70B", name: "GeneticLemonade 70B (zerofata/L3.3-GeneticLemonade-Final-v2-70B)" },
    { value: "NousResearch/Hermes-4-70B", name: "Hermes 4 70B (NousResearch/Hermes-4-70B)" },
    { value: "meta-llama/Llama-3.2-3B-Instruct", name: "Llama 3.2 3B (meta-llama/Llama-3.2-3B-Instruct)" },
    { value: "Qwen/Qwen2.5-1.5B-Instruct", name: "Qwen 1.5B (Qwen/Qwen2.5-1.5B-Instruct)" },
    { value: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", name: "TinyLlama 1.1B (TinyLlama/TinyLlama-1.1B-Chat-v1.0)" }
];

// ---------------------------------------------------------------------------
// Among Us colors with sprite URLs (order: Red, Orange, Yellow, Lime, Green, Cyan, Blue, Purple, Brown, Pink, White, Black)
// Sprites also saved in static/data/among-us-sprites.json for future use.
// ---------------------------------------------------------------------------
const COLORS = [
    { value: "red", name: "Red", spriteUrl: "https://preview.redd.it/an871k4o1sn51.png?width=440&format=png&auto=webp&s=85dcd6cb73b8760802e254ee14dfa3c7ab444591" },
    { value: "orange", name: "Orange", spriteUrl: "https://preview.redd.it/iio3xm4o1sn51.png?width=440&format=png&auto=webp&s=2b9fb1b29396502998feda5c6ed2ed75919c6ad8" },
    { value: "yellow", name: "Yellow", spriteUrl: "https://preview.redd.it/xprpkp063sn51.png?width=440&format=png&auto=webp&s=5d51eb262af4a50e8f935218feb52682540aa525" },
    { value: "lime", name: "Lime", spriteUrl: "https://preview.redd.it/76glbq4o1sn51.png?width=440&format=png&auto=webp&s=a22610bfbd735d024448389fd80009b255c33524" },
    { value: "green", name: "Green", spriteUrl: "https://preview.redd.it/vf3ojm4o1sn51.png?width=440&format=png&auto=webp&s=7cfa65a910d76e324fcc4c23468a9b801c3b74d5" },
    { value: "cyan", name: "Cyan", spriteUrl: "https://preview.redd.it/0j244l4o1sn51.png?width=440&format=png&auto=webp&s=c74e2de99bdb7da7471469d8274a4eaae244207e" },
    { value: "blue", name: "Blue", spriteUrl: "https://preview.redd.it/ph2jho4o1sn51.png?width=440&format=png&auto=webp&s=7e080e5447d69d1425a8b8a20f1115de18aa69fd" },
    { value: "purple", name: "Purple", spriteUrl: "https://preview.redd.it/9kvk25sh2sn51.png?width=440&format=png&auto=webp&s=c469d1dc3fda76a0d2271cecb8d422f1aff925ab" },
    { value: "brown", name: "Brown", spriteUrl: "https://preview.redd.it/f7f4fmpi2sn51.png?width=440&format=png&auto=webp&s=79d8eaf10daa28753816cfc8ec5cd26cfa517d29" },
    { value: "pink", name: "Pink", spriteUrl: "https://preview.redd.it/ppawzo4o1sn51.png?width=440&format=png&auto=webp&s=d09c261013546996e8325d507ff230a7e9513793" },
    { value: "white", name: "White", spriteUrl: "https://preview.redd.it/xyqo6hx42sn51.png?width=440&format=png&auto=webp&s=3bf357e64a68883aee1618a1abdadc16d9ceee73" },
    { value: "black", name: "Black", spriteUrl: "https://preview.redd.it/4eof2l4o1sn51.png?width=440&format=png&auto=webp&s=02f3a9c7fdb96a50204c5dc272a7e72dfff7cbac" }
];

// Default model for new agents (TinyLlama = smallest, now last in dropdown)
const TINY_LLAMA_VALUE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

// Default config for first 4 agents: Model, Role, Color = Red, Orange, Yellow, Lime (then Green, Cyan, ... as you add agents)
const DEFAULT_AGENTS_4 = [
    { model: TINY_LLAMA_VALUE, role: "byzantine", color: "red" },
    { model: TINY_LLAMA_VALUE, role: "byzantine", color: "orange" },
    { model: TINY_LLAMA_VALUE, role: "honest", color: "yellow" },
    { model: TINY_LLAMA_VALUE, role: "honest", color: "lime" }
];

/**
 * Validates agent count: must be integer between 4 and 12.
 * @returns {number|null} Valid count or null if invalid
 */
function validateAgentCount() {
    const input = document.getElementById("num_agents");
    const raw = input.value.trim();
    const num = parseInt(raw, 10);
    if (Number.isNaN(num) || num < 4 || num > 12) {
        alert("Number of agents must be between 4 and 12.");
        input.value = "4";
        return null;
    }
    return num;
}

/**
 * Builds default model/role/color for an agent by index.
 * First 4 use DEFAULT_AGENTS_4 (red, orange, yellow, lime); rest use TinyLlama, Honest, and next color in COLORS order.
 */
function getDefaultForAgent(index) {
    if (index < DEFAULT_AGENTS_4.length) {
        return DEFAULT_AGENTS_4[index];
    }
    return {
        model: TINY_LLAMA_VALUE,
        role: "honest",
        color: COLORS[index].value
    };
}

/**
 * Creates one table row for an agent (number, model dropdown, role dropdown, color dropdown).
 * @param {number} index - 0-based agent index
 * @returns {HTMLTableRowElement}
 */
function createAgentRow(index) {
    const defaults = getDefaultForAgent(index);
    const row = document.createElement("tr");

    // Column 1: Agent number
    const numCell = document.createElement("td");
    numCell.className = "agent-num";
    numCell.textContent = index + 1;
    row.appendChild(numCell);

    // Column 2: Model dropdown
    const modelCell = document.createElement("td");
    const modelSelect = document.createElement("select");
    modelSelect.name = `agent_${index}_model`;
    modelSelect.className = "table-select model-select";
    modelSelect.required = true;
    MODELS.forEach(function (m) {
        const opt = document.createElement("option");
        opt.value = m.value;
        opt.textContent = m.name;
        modelSelect.appendChild(opt);
    });
    modelSelect.value = defaults.model;
    modelCell.appendChild(modelSelect);
    row.appendChild(modelCell);

    // Column 3: Role dropdown (Honest / Byzantine)
    const roleCell = document.createElement("td");
    const roleSelect = document.createElement("select");
    roleSelect.name = `agent_${index}_role`;
    roleSelect.className = "table-select role-select";
    roleSelect.required = true;
    ["honest", "byzantine"].forEach(function (r) {
        const opt = document.createElement("option");
        opt.value = r;
        opt.textContent = r === "honest" ? "Honest" : "Byzantine";
        roleSelect.appendChild(opt);
    });
    roleSelect.value = defaults.role;
    roleCell.appendChild(roleSelect);
    row.appendChild(roleCell);

    // Column 4: Color picker with sprite images (custom dropdown)
    const colorCell = createColorPickerCell(index, defaults.color);
    row.appendChild(colorCell);

    return row;
}

/**
 * Creates a table cell with a custom color dropdown showing Among Us sprites.
 * Uses a hidden input for form submission (name="agent_{agentIndex}_color").
 * @param {number} agentIndex - 0-based agent index
 * @param {string} selectedValue - initial color value (e.g. "red")
 * @returns {HTMLTableCellElement}
 */
function createColorPickerCell(agentIndex, selectedValue) {
    const selected = COLORS.find(function (c) { return c.value === selectedValue; }) || COLORS[0];
    const cell = document.createElement("td");
    cell.className = "color-picker-cell";

    const hiddenInput = document.createElement("input");
    hiddenInput.type = "hidden";
    hiddenInput.name = "agent_" + agentIndex + "_color";
    hiddenInput.value = selected.value;
    hiddenInput.setAttribute("data-color-picker-value", "1");

    const wrapper = document.createElement("div");
    wrapper.className = "color-picker";

    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "color-picker-trigger table-select";
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    const triggerImg = document.createElement("img");
    triggerImg.src = selected.spriteUrl;
    triggerImg.alt = selected.name;
    triggerImg.className = "color-picker-sprite";
    const triggerLabel = document.createElement("span");
    triggerLabel.className = "color-picker-label";
    triggerLabel.textContent = selected.name;
    trigger.appendChild(triggerImg);
    trigger.appendChild(triggerLabel);

    const dropdown = document.createElement("div");
    dropdown.className = "color-picker-dropdown";
    dropdown.setAttribute("role", "listbox");
    dropdown.hidden = true;

    COLORS.forEach(function (c) {
        const option = document.createElement("div");
        option.className = "color-picker-option";
        option.setAttribute("role", "option");
        option.setAttribute("data-value", c.value);
        const optImg = document.createElement("img");
        optImg.src = c.spriteUrl;
        optImg.alt = c.name;
        optImg.className = "color-picker-sprite";
        const optLabel = document.createElement("span");
        optLabel.textContent = c.name;
        option.appendChild(optImg);
        option.appendChild(optLabel);
        option.addEventListener("click", function () {
            if (this.classList.contains("color-picker-option-disabled")) return;
            hiddenInput.value = c.value;
            triggerImg.src = c.spriteUrl;
            triggerImg.alt = c.name;
            triggerLabel.textContent = c.name;
            closeColorPickerDropdown(dropdown);
            trigger.setAttribute("aria-expanded", "false");
        });
        dropdown.appendChild(option);
    });

    trigger.addEventListener("click", function (e) {
        e.stopPropagation();
        var isOpen = !dropdown.hidden;
        closeAllColorPickers();
        if (!isOpen) {
            dropdown.colorPickerWrapper = wrapper;
            dropdown.colorPickerAgentIndex = agentIndex;
            document.body.appendChild(dropdown);
            updateColorPickerDisabledOptions(dropdown, agentIndex);
            var rect = trigger.getBoundingClientRect();
            dropdown.style.position = "fixed";
            dropdown.style.top = (rect.bottom + 4) + "px";
            dropdown.style.left = rect.left + "px";
            dropdown.style.minWidth = rect.width + "px";
            dropdown.hidden = false;
            trigger.setAttribute("aria-expanded", "true");
        }
    });

    wrapper.appendChild(hiddenInput);
    wrapper.appendChild(trigger);
    wrapper.appendChild(dropdown);
    cell.appendChild(wrapper);
    return cell;
}

/**
 * Returns the set of color values currently selected by other agents (excluding the given agent index).
 * Used to disable those options in this agent's color dropdown so each agent has a unique color.
 */
function getUsedColorsExcept(agentIndex) {
    var used = new Set();
    var inputs = document.querySelectorAll('input[name^="agent_"][name$="_color"]');
    inputs.forEach(function (input) {
        var match = input.name.match(/^agent_(\d+)_color$/);
        if (match && parseInt(match[1], 10) !== agentIndex && input.value) {
            used.add(input.value);
        }
    });
    return used;
}

/**
 * Disables color options in this dropdown that are already selected by other agents.
 * @param {HTMLElement} dropdown - The .color-picker-dropdown element
 * @param {number} agentIndex - 0-based index of the agent this picker belongs to
 */
function updateColorPickerDisabledOptions(dropdown, agentIndex) {
    var used = getUsedColorsExcept(agentIndex);
    dropdown.querySelectorAll(".color-picker-option").forEach(function (option) {
        var value = option.getAttribute("data-value");
        if (used.has(value)) {
            option.classList.add("color-picker-option-disabled");
            option.setAttribute("aria-disabled", "true");
        } else {
            option.classList.remove("color-picker-option-disabled");
            option.removeAttribute("aria-disabled");
        }
    });
}

/**
 * Closes one color picker dropdown and returns it to its wrapper (if it was moved to body).
 * @param {HTMLElement} dropdown - The .color-picker-dropdown element
 */
function closeColorPickerDropdown(dropdown) {
    dropdown.hidden = true;
    dropdown.style.position = "";
    dropdown.style.top = "";
    dropdown.style.left = "";
    dropdown.style.minWidth = "";
    if (dropdown.colorPickerWrapper) {
        dropdown.colorPickerWrapper.appendChild(dropdown);
        dropdown.colorPickerWrapper = null;
    }
}

/**
 * Closes all color picker dropdowns on the page (returns them to their wrappers and hides).
 */
function closeAllColorPickers() {
    document.querySelectorAll(".color-picker-dropdown").forEach(function (el) {
        closeColorPickerDropdown(el);
    });
    document.querySelectorAll(".color-picker-trigger[aria-expanded=\"true\"]").forEach(function (el) {
        el.setAttribute("aria-expanded", "false");
    });
}

/**
 * Clears the agent table body and fills it with one row per agent (4–12).
 * Validates agent count; if invalid, shows alert and does nothing.
 * Called on page load and when CONFIRM is clicked.
 */
function generateAgentTable() {
    const count = validateAgentCount();
    if (count === null) return;

    const tbody = document.getElementById("agentTableBody");
    tbody.innerHTML = "";

    for (let i = 0; i < count; i++) {
        tbody.appendChild(createAgentRow(i));
    }
}

/**
 * Validates number of rounds (1–20). Called on form submit.
 * @returns {boolean} true if valid
 */
function validateRounds() {
    const input = document.getElementById("num_rounds");
    const num = parseInt(input.value, 10);
    if (Number.isNaN(num) || num < 1 || num > 20) {
        alert("Number of rounds must be between 1 and 20.");
        input.focus();
        return false;
    }
    return true;
}

/**
 * Validates that every agent has a unique color. Called on form submit.
 * @returns {boolean} true if all colors are unique
 */
function validateUniqueColors() {
    var values = [];
    document.querySelectorAll('input[name^="agent_"][name$="_color"]').forEach(function (input) {
        if (input.value) values.push(input.value);
    });
    var seen = new Set();
    for (var i = 0; i < values.length; i++) {
        if (seen.has(values[i])) {
            alert("Each agent must have a unique color. Two or more agents share the same color.");
            return false;
        }
        seen.add(values[i]);
    }
    return true;
}

/**
 * Maximum number of Byzantine agents allowed for a given number of agents.
 * Rule: Byzantines must be fewer than half (byz < size/2).
 * For 6 agents or fewer, also cap at 3.
 * @param {number} numAgents - total number of agents (4–12)
 * @returns {number}
 */
function getMaxByzantines(numAgents) {
    var maxByHalf = Math.floor((numAgents - 1) / 2);
    if (numAgents <= 6) {
        return Math.min(3, maxByHalf);
    }
    return maxByHalf;
}

/**
 * Validates Byzantine count: at least 1, and at most getMaxByzantines(numAgents).
 * Called on form submit.
 * @returns {boolean} true if valid
 */
function validateByzantineCount() {
    var numAgents = 0;
    var byzantineCount = 0;
    document.querySelectorAll('select[name^="agent_"][name$="_role"]').forEach(function (select) {
        numAgents++;
        if (select.value === "byzantine") byzantineCount++;
    });
    if (byzantineCount < 1) {
        alert("At least one agent must be Byzantine.");
        return false;
    }
    var maxByz = getMaxByzantines(numAgents);
    if (byzantineCount > maxByz) {
        var msg = "Too many Byzantine agents. For " + numAgents + " agents, at most " + maxByz + " can be Byzantine (must be fewer than half of the group";
        if (numAgents <= 6) {
            msg += ", and for 6 or fewer agents the maximum is 3";
        }
        msg += ").";
        alert(msg);
        return false;
    }
    return true;
}

/**
 * Form submit handler: validate rounds, unique colors, and Byzantine count before allowing submit.
 */
function onConfigSubmit(e) {
    if (!validateRounds()) {
        e.preventDefault();
        return false;
    }
    if (!validateUniqueColors()) {
        e.preventDefault();
        return false;
    }
    if (!validateByzantineCount()) {
        e.preventDefault();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Initialize on DOM ready: build default 4-agent table and bind CONFIRM + form
// ---------------------------------------------------------------------------
window.addEventListener("DOMContentLoaded", function () {
    generateAgentTable();

    var confirmBtn = document.getElementById("confirmAgentsBtn");
    if (confirmBtn) {
        confirmBtn.addEventListener("click", generateAgentTable);
    }

    var form = document.getElementById("configForm");
    if (form) {
        form.addEventListener("submit", onConfigSubmit);
    }

    // Close color picker dropdowns when clicking outside
    document.addEventListener("click", function () {
        closeAllColorPickers();
    });
});
