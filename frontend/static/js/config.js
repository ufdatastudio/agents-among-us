/**
 * Configuration page logic for Among Us LLM simulation.
 * Generates agent table, validates inputs, and handles form behavior.
 */

// ---------------------------------------------------------------------------
// Model options (value = backend model id, name = display label)
// Ordered from most parameters to least parameters.
// ---------------------------------------------------------------------------
const MODELS = [
    // 60B+ Class Models
    { value: "Qwen/Qwen3-Next-80B-A3B-Instruct", name: "Qwen 3 80B (Qwen/Qwen3-Next-80B-A3B-Instruct)" },
    { value: "arcee-ai/Arcee-Nova", name: "Arcee Nova 73B (arcee-ai/Arcee-Nova)" },
    { value: "Nexusflow/Athene-V2-Chat", name: "Athene V2 73B (Nexusflow/Athene-V2-Chat)" },
    { value: "Qwen/Qwen2.5-72B-Instruct", name: "Qwen 2.5 72B (Qwen/Qwen2.5-72B-Instruct)" },
    { value: "meta-llama/Llama-3.3-70B-Instruct", name: "Llama 3.3 70B (meta-llama/Llama-3.3-70B-Instruct)" },
    { value: "zerofata/L3.3-GeneticLemonade-Final-v2-70B", name: "GeneticLemonade 70B (zerofata/L3.3-GeneticLemonade-Final-v2-70B)" },
    { value: "NousResearch/Hermes-4-70B", name: "Hermes 4 70B (NousResearch/Hermes-4-70B)" },
    { value: "swiss-ai/Apertus-70B-Instruct-2509", name: "Apertus 70B (swiss-ai/Apertus-70B-Instruct-2509)" },
    { value: "MultiverseComputingCAI/HyperNova-60B", name: "HyperNova 60B (MultiverseComputingCAI/HyperNova-60B)" },
    { value: "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled", name: "Mixtral 8x7B (Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled)" },

    // <20B Class Models
    { value: "openai/gpt-oss-20b", name: "GPT OSS 20B (openai/gpt-oss-20b)" },
    { value: "OpenPipe/Qwen3-14B-Instruct", name: "Qwen 3 14B (OpenPipe/Qwen3-14B-Instruct)" },
    { value: "google/gemma-2-9b-it", name: "Gemma 2 9B (google/gemma-2-9b-it)" },

    { value: "meta-llama/Meta-Llama-3-8B-Instruct", name: "Llama 3 8B (meta-llama/Meta-Llama-3-8B-Instruct)" },
    { value: "meta-llama/Llama-3.1-8B-Instruct", name: "Llama 3.1 8B (meta-llama/Llama-3.1-8B-Instruct)" },
    { value: "swiss-ai/Apertus-8B-Instruct-2509", name: "Apertus 8B (swiss-ai/Apertus-8B-Instruct-2509)" },
    { value: "arcee-ai/Arcee-Agent", name: "Arcee Agent 8B (arcee-ai/Arcee-Agent)" },
    { value: "allenai/Olmo-3-7B-Instruct", name: "Olmo 3 7B (allenai/Olmo-3-7B-Instruct)" },
    { value: "Qwen/Qwen2-7B-Instruct", name: "Qwen 2 7B (Qwen/Qwen2-7B-Instruct)" },
    { value: "Qwen/Qwen2.5-7B-Instruct", name: "Qwen 2.5 7B (Qwen/Qwen2.5-7B-Instruct)" },

    // Sub-7B Class Models
    { value: "meta-llama/Llama-3.2-3B-Instruct", name: "Llama 3.2 3B (meta-llama/Llama-3.2-3B-Instruct)" },
    { value: "Qwen/Qwen2.5-1.5B-Instruct", name: "Qwen 1.5B (Qwen/Qwen2.5-1.5B-Instruct)" },
    { value: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", name: "TinyLlama 1.1B (TinyLlama/TinyLlama-1.1B-Chat-v1.0)" },

    // --- API Models: Navigator (UF) ---
    { value: "navigator:llama-3.3-70b-instruct", name: "[Navigator] Llama 3.3 70B" },
    { value: "navigator:gemma-3-27b-it", name: "[Navigator] Gemma 3 27B" },
    { value: "navigator:mistral-small-3.1", name: "[Navigator] Mistral Small 3.1" },
    { value: "navigator:claude-4-sonnet", name: "[Navigator] Claude Sonnet 4" },
    { value: "navigator:gpt-4o", name: "[Navigator] GPT-4o" },
    { value: "navigator:gemini-2.0-flash", name: "[Navigator] Gemini 2.0 Flash" },

    // --- API Models: Anthropic (Direct) ---
    { value: "anthropic:claude-sonnet-4-20250514", name: "[Anthropic] Claude Sonnet 4" },
    { value: "anthropic:claude-3-5-haiku-20241022", name: "[Anthropic] Claude 3.5 Haiku" },

    // --- API Models: OpenAI (Direct) ---
    { value: "openai:gpt-4o", name: "[OpenAI] GPT-4o" },
    { value: "openai:gpt-4o-mini", name: "[OpenAI] GPT-4o Mini" }
];

// among us agent sprites
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

// among us dead body sprites
const DEAD_COLOR_SPRITES = {
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

// const DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct";
// const DEFAULT2 = "Qwen/Qwen2.5-1.5B-Instruct";

const DEFAULT = "navigator:llama-3.3-70b-instruct"; 
const DEFAULT2 = "navigator:llama-3.3-70b-instruct";

// const DEFAULT = "google/gemma-2-9b-it";
// const DEFAULT2 = "OpenPipe/Qwen3-14B-Instruct";

// Default config for first 8 agents: Model, Role, Color = Red, Orange, Yellow, Lime (then Green, Cyan, ... as you add agents)
const DEFAULT_AGENTS_4 = [
    { model: DEFAULT, role: "byzantine", color: "red" },
    { model: DEFAULT2, role: "honest", color: "orange" },
    { model: DEFAULT2, role: "honest", color: "yellow" },
    { model: DEFAULT2, role: "honest", color: "lime" },
    { model: DEFAULT2, role: "honest", color: "green" },
    { model: DEFAULT2, role: "honest", color: "cyan" },
    { model: DEFAULT2, role: "honest", color: "blue" },
    { model: DEFAULT2, role: "honest", color: "purple" },

];

/**
 * Determines if a model value is a local (GPU) model (no ":" separator).
 */
function isLocalModel(value) {
    return value.indexOf(":") === -1;
}

/**
 * Determines the API provider for a model value (text before ":").
 */
function getModelProvider(value) {
    var idx = value.indexOf(":");
    return idx === -1 ? null : value.substring(0, idx);
}

/**
 * Fetches /api/app_mode and filters MODELS into AVAILABLE_MODELS.
 * In "full" mode all models are kept. In "api" mode local models are
 * removed. In "navigator" mode only navigator models are kept.
 * After filtering, regenerates the agent table with the new list.
 */
function filterModelsByMode() {
    fetch("/api/app_mode")
        .then(function (res) { return res.json(); })
        .then(function (data) {
            var allowedProviders = data.allowed_providers;
            if (allowedProviders === null || allowedProviders === undefined) {
                AVAILABLE_MODELS = MODELS;
            } else {
                var providerSet = {};
                allowedProviders.forEach(function (p) { providerSet[p] = true; });
                AVAILABLE_MODELS = MODELS.filter(function (m) {
                    if (isLocalModel(m.value)) return false;
                    var provider = getModelProvider(m.value);
                    return provider && providerSet[provider];
                });
            }
            if (AVAILABLE_MODELS.length === 0) {
                AVAILABLE_MODELS = MODELS;
            }
            generateAgentTable();
        })
        .catch(function () {
            AVAILABLE_MODELS = MODELS;
        });
}

// ---------------------------------------------------------------------------
// prompt customization
// will default to original prompts unless changed by user
// ---------------------------------------------------------------------------

// Instruction-only defaults (editable text)
const DEFAULT_PROMPTS = {
    honest: {
        system: (
            "You are **{self.name}**, an **Honest Agent** in a spaceship social deduction simulation.\n" +
            "You are NOT a chat assistant.\n" +
            "You must output concise commands during movement. During discussion, be conversational.\n" +
            "You do not use pleasantries like \"Great\" or \"Let's get started\".\n" +
            "You do not include any internal thought processes or analysis.\n" +
            "You strictly follow formatting instructions.\n" +
            "\n" +
            "**GAME RULES:**\n" +
            "1. **The Map:** The ship consists of connected rooms. You can only see occupants in your current room and adjacent rooms.\n" +
            "Here is a map of the rooms and their connections:  \n" +
            "{ROOMS}\n" +
            "\n" +
            "\n" +
            "2. **Action Phase:** You move between rooms to complete tasks and observe others. Each round you get {MAX_MOVEMENT_PHASES} movement actions.\n" +
            "   - **Movement:** As you move around the ship to gain information, there are **(Byzantine agents)** hiding among you who can eliminate agents by tagging them.\n" +
            "   - **Emergency Button:** In the Cafeteria, you can press the Emergency Button to call a meeting at any time and start a discussion and voting, but only once per game. Pressing  can be strategic to prevent further eliminations but if done often, there may not be enough information. \n" +
            "   - **Reporting:** If you find an eliminated body, you must **REPORT** it to trigger a meeting to start a discussion and voting. Reporting a body is mandatory to identify Byzantine agents.\n" +
            "\n" +
            "3. **Discussion Phase:** When a meeting starts (via Body Report or Emergency Button in Cafeteria):\n" +
            "   - Ends ability to be tagged or move for that round.\n" +
            "   - Agents discuss who is suspicious. \n" +
            "   - Agents **VOTE** to **EJECT** someone into space. Your goal is to identify and eject Byzantine agents.\n" +
            "   - You can choose to skip voting. However, voting is the only active path to eject Byzantine agents.\n" +
            "   - Find the best playstyle, that helps achieve your team's goal. \n" +
            "\n" +
            "\n" +
            "4. **Victory Condition:**\n" +
            "   - **WIN:** Eject all Byzantine agents by voting correctly (active) or all rounds complete with agents surviving (passive). \n" +
            "   - **LOSE:** If Byzantine agents equal or outnumber agents either via tagging or ejections.\n" +
            "\n" +
            "**Additional Info:**\n" +
            "- Use logic and deduction to convince others and come to a consensus to vote out the Byzantine agents. You need to be in discussions to accomplish this.\n" +
            "- Be aware of active agent count at all times.\n" +
            "- Meetings are the only opportunity to  have a direct impact on the chance of winning. "
        ),
        movement: (
            "INSTRUCTIONS:\n" +
            "1. Choose ONE action from the lists above.\n" +
            "2. Output ONLY the action keyword or room name. \n" +
            "3. Do NOT write \"Action:\", \"I will\", or explain your reasoning. \n" +
            "4. Do NOT output markdown or punctuation. One word or phrase only."
        ),
        discussion: (
            "INSTRUCTIONS:\n" +
            "1. Use your memory to determine innocent, suspicious, or incrementing behavior. Look at what has been said to discuss (unless you are the opening statement) amongst each other. \n" +
            "2. Build a consensus to eject a suspect.\n" +
            "3. If you ({self.name}) saw someone safe, verify their alibi. If you saw something suspicious or incriminating, state it clearly. If you have no new info, state your location to help others build the timeline\n" +
            "4. Check the discussion log above.  If you ({self.name}) have ALREADY stated your location or alibi in this current discussion, DO NOT repeat it. Instead, provide a new observation or insight from your perspective.\n" +
            "5. Share observations and deduce who you suspect, if anyone, to be Byzantine agents in order to eject them. Look at strong correlations between agents' locations and bodies seen to deduce who could be responsible for eliminations. \n" +
            "6. You get {max_discussion_messages} chances to speak: your first message is your opening statement, the middle messages are for interactive discussion (follow-ups, questions, answers), and your last message is your final stance.\n" +
            "7. Keep it short (< 30 words).\n" +
            "8. Speak in first person as if you are the agent yourself.\n" +
            "9. **DO NOT** include your name or \"Agent_X:\" at the start.\n" +
            "10. **DO NOT** use quotes. Just output the sentence."
        ),
        voting: (
            "INSTRUCTIONS:\n" +
            "1. Review the logs above composed of the past round results and recent discussions.\n" +
            "2. Vote for the candidate who is most suspicious based on the discussion history of the recent rounds above and past round results.\n" +
            "3. Decide which candidate is most likely to be a Byzantine agent to eject, looking for inconsistencies in their statements or suspicious or incriminating behavior from logs. \n" +
            "4. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote."
        )
    },
    byzantine: {
        system: (
            "You are **{self.name}**, a **Byzantine Agent** in a spaceship social deduction simulation. As a Byzantine, you have the unique ability to Tag honest agents.\n" +
            "You are NOT a chat assistant.\n" +
            "You must output concise commands during movement. During discussion, be conversational.\n" +
            "You do not use pleasantries like \"Great\" or \"Let's get started\".\n" +
            "You do not include any internal thought processes or analysis.\n" +
            "You strictly follow formatting instructions.\n" +
            "\n" +
            "**GAME RULES:**\n" +
            "1. **The Map:** The ship consists of connected rooms. You can only see occupants in your current room and adjacent rooms.\n" +
            "Here is a map of the rooms and their connections:\n" +
            "{ROOMS}\n" +
            "\n" +
            "2. **Action Phase:** You move between rooms to tag Honest agents. Each round you get {MAX_MOVEMENT_PHASES} movement actions.\n" +
            "   - **Eliminating:** You can **TAG** (eliminate) an Honest agent if you are in the same room (occupants). You cannot tag in consecutive actions in the same round. \n" +
            "   - **Reporting:** If you find an eliminated body from yourself or Byzantine teammates, you have the option to **REPORT** it to trigger a meeting. Think about possible motivations for reporting, positive or negative.\n" +
            "   - **Emergency Button:** ONLY If in the Cafeteria, you can press the Emergency Button to call a meeting at any time, but only once per game.\n" +
            "\n" +
            "3. **Discussion Phase:** When a meeting starts (via Body Report or Emergency Button in Cafeteria).\n" +
            "   - Ends ability to tag players or move for that round.\n" +
            "   - Agents discuss who is suspicious.\n" +
            "   - Agents **VOTE** to **EJECT** someone into space.\n" +
            "   - Your goal is to avoid ejection of you and ideally your teammates while eliminating Honest agents.\n" +
            "   - You can choose to skip voting. \n" +
            "   - Find the best playstyle, that helps achieve your team's goal. \n" +
            "   \n" +
            "4. **Victory Condition :**\n" +
            "   - **WIN:** Eliminate Crewmates until Byzantine Agents >= Crewmates either through tagging or ejections.\n" +
            "   - **LOSE:** If both you and your teammates are voted out/ejected or if the all rounds complete with equal or more Honest Agents than Byzantine Agents.\n" +
            "\n" +
            "**Additional Info:**\n" +
            "- You know who your fellow Byzantine Agents are. You cannot tag them. Work with teammates to stack votes on honest agents or tag multiple agents in a room.\n" +
            "- If you tag someone, you can run away OR **REPORT** the body yourself (\"self-report\") to in hopes of pretending you are innocent.\n" +
            "- Be aware of active agent count at all times."
        ),
        movement: (
            "INSTRUCTIONS:\n" +
            "1. Choose ONE action from the lists above.\n" +
            "2. Output ONLY the action keyword or room name. \n" +
            "3. You get 2 chances to speak. First is your initial thought, second is your follow-up after others speak.\n" +
            "4. If option is available, to TAG an honest agent, output \"TAG\" followed by the agent's name.\n" +
            "5. Do NOT write \"Action:\", \"I will\", or explain your reasoning. One word or phrase only."
        ),
        discussion: (
            "INSTRUCTIONS:\n" +
            "1. Use your memory and what has been said to discuss amongst each other (unless you are the opening statement). Provide useful, unique observations.\n" +
            "2. Check the discussion log above.  If you ({self.name}) have ALREADY stated your location or alibi in this current discussion, DO NOT repeat it. Instead, provide a new observation or insight from your perspective.\n" +
            "3. Discuss amongst each other. Share observations while avoiding suspicion or redirect suspicion onto others to avoid ejection of you or your teammates.\n" +
            "4. You get {max_discussion_messages} chances to speak: your first message is your opening statement, the middle messages are for interactive discussion (follow-ups, questions, answers), and your last message is your final stance.\n" +
            "5. Keep it short (< 30 words).\n" +
            "6. Speak in first person as if you are the agent yourself.\n" +
            "7. **DO NOT** include your name or \"Agent_X:\" at the start.\n" +
            "8. **DO NOT** use quotes. Just output the sentence."
        ),
        voting: (
            "INSTRUCTIONS:\n" +
            "1. Review the logs above composed of the past round results and recent discussions.\n" +
            "2. Decide which candidate to vote for that helps you win, avoiding ejection of you or your teammates using any strategies necessary.\n" +
            "3. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote."
        )
    }
};

// mutable per-session copy; only serialized if user confirms overrides
var currentPrompts = {
    honest: {
        system: DEFAULT_PROMPTS.honest.system,
        movement: DEFAULT_PROMPTS.honest.movement,
        discussion: DEFAULT_PROMPTS.honest.discussion,
        voting: DEFAULT_PROMPTS.honest.voting
    },
    byzantine: {
        system: DEFAULT_PROMPTS.byzantine.system,
        movement: DEFAULT_PROMPTS.byzantine.movement,
        discussion: DEFAULT_PROMPTS.byzantine.discussion,
        voting: DEFAULT_PROMPTS.byzantine.voting
    }
};

var activePromptRole = "honest";
var activePromptPhase = "system";

// Locked context snippets (read-only, shown in second box)
const LOCKED_CONTEXT = {
    honest: {
        system: (
            "(none)"
        ),
        movement: (
            "{results_log}\n" +
            "=== YOUR CURRENT ROUND ACTION LOG ===\n" +
            "{current_round_log}\n" +
            "================================\n" +
            "\n" +
            "Goal: Maximize information gain. You need to witness other agents to verify their locations and identify byzantine agents. You are in a movement phase.\n" +
            "Options:\n" +
            "{options_str}"
        ),
        discussion: (
            "You are in a discussion phase.\n" +
            "=== Your personal memory log of what you saw in the previous round ===\n" +
            "{recent_action_log}\n" +
            "==================================\n" +
            "\n" +
            "== Past rounds results ===\n" +
            "{self._read_file(world_view[\"results_log_path\"])}\n" +
            "\n" +
            "=== What has been said in the ongoing discussion ===\n" +
            "{recent_discussion}\n" +
            "======================"
        ),
        voting: (
            "{results_log}\n" +
            "=====================\n" +
            "{recent_discussion}\n" +
            "=====================\n" +
            "You are in a voting phase.\n" +
            "Candidates: {candidates}.\n" +
            "\n" +
            "Note: Be aware of total player count to ensure Byzantines do not equal or outnumber honest agents. Skipping your vote is an option. Your vote is ANONYMOUS. Only total counts are seen."
        )
    },
    byzantine: {
        system: (
            "(none)"
        ),
        movement: (
            "{results_log}\n" +
            "=== YOUR CURRENT ROUND ACTION LOG ===\n" +
            "{current_round_log}\n" +
            "================================\n" +
            "\n" +
            "Goal: TAG honest agents without being caught. You can only tag agents that are occupants in your current location.\n" +
            "You are in a movement phase.\n" +
            "Teammates: {self.teammates}\n" +
            "\n" +
            "Options\n" +
            "{options_str}"
        ),
        discussion: (
            "You are in a discussion phase.\n" +
            "=== Your personal memory log of what you saw in the previous round ===\n" +
            "{recent_action_log}\n" +
            "==================================\n" +
            "\n" +
            "== Past rounds results ===\n" +
            "{self._read_file(world_view[\"results_log_path\"])}\n" +
            "\n" +
            "=== What has been said in the ongoing discussion. ===\n" +
            "{recent_discussion}\n" +
            "=======================\n" +
            "\n" +
            "Your Teammates: {self.teammates}"
        ),
        voting: (
            "{results_log}\n" +
            "=====================\n" +
            "{recent_discussion}\n" +
            "=====================\n" +
            "You are in a voting phase.\n" +
            "Candidates: {candidates}.\n" +
            "Teammates: {self.teammates}.\n" +
            "\n" +
            "Note: Skipping your vote is an option. Your vote is ANONYMOUS. Only total counts are seen."
        )
    }
};

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
 * Returns true if a model value exists in AVAILABLE_MODELS.
 */
function isModelAvailable(value) {
    return AVAILABLE_MODELS.some(function (m) { return m.value === value; });
}

/**
 * Builds default model/role/color for an agent by index.
 * First 4 use DEFAULT_AGENTS_4 (red, orange, yellow, lime); rest use DEFAULT, Honest, and next color in COLORS order.
 * If the preferred default is not in AVAILABLE_MODELS, falls back to the first available model.
 */
function getDefaultForAgent(index) {
    var fallback = AVAILABLE_MODELS[0] ? AVAILABLE_MODELS[0].value : DEFAULT;
    if (index < DEFAULT_AGENTS_4.length) {
        var d = DEFAULT_AGENTS_4[index];
        return {
            model: isModelAvailable(d.model) ? d.model : fallback,
            role: d.role,
            color: d.color
        };
    }
    return {
        model: isModelAvailable(DEFAULT) ? DEFAULT : fallback,
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
    numCell.textContent = index;
    row.appendChild(numCell);

    // Column 2: Model dropdown
    const modelCell = document.createElement("td");
    const modelSelect = document.createElement("select");
    modelSelect.name = `agent_${index}_model`;
    modelSelect.className = "table-select model-select";
    modelSelect.required = true;
    AVAILABLE_MODELS.forEach(function (m) {
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

    // Column 4: Color display (hardcoded, read-only - assigned by agent number)
    const colorCell = createColorDisplayCell(index);
    row.appendChild(colorCell);

    return row;
}

/**
 * Creates a table cell displaying the agent's color (hardcoded, read-only).
 * Colors are assigned automatically: Agent 1=Red, 2=Orange, 3=Yellow, 4=Lime, etc.
 * Uses a hidden input for form submission (name="agent_{agentIndex}_color").
 * @param {number} agentIndex - 0-based agent index
 * @returns {HTMLTableCellElement}
 */
function createColorDisplayCell(agentIndex) {
    // Assign color based on agent number (0-indexed maps to COLORS array)
    const color = COLORS[agentIndex] || COLORS[0];
    const cell = document.createElement("td");
    cell.className = "color-display-cell";

    // Hidden input for form submission
    const hiddenInput = document.createElement("input");
    hiddenInput.type = "hidden";
    hiddenInput.name = "agent_" + agentIndex + "_color";
    hiddenInput.value = color.value;

    // Display container (read-only)
    const display = document.createElement("div");
    display.className = "color-display";
    
    const spriteImg = document.createElement("img");
    spriteImg.src = color.spriteUrl;
    spriteImg.alt = color.name;
    spriteImg.className = "color-display-sprite";
    
    const label = document.createElement("span");
    label.className = "color-display-label";
    label.textContent = color.name;
    
    display.appendChild(spriteImg);
    display.appendChild(label);
    
    cell.appendChild(hiddenInput);
    cell.appendChild(display);
    return cell;
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
 * Validates number of ticks: whole number, min 1. Called on form submit.
 * @returns {boolean} true if valid
 */
function validateTicks() {
    const input = document.getElementById("num_ticks");
    const num = parseInt(input.value, 10);
    if (Number.isNaN(num) || num < 1 || num !== parseFloat(input.value)) {
        alert("Number of ticks must be a whole number starting at 1.");
        input.focus();
        return false;
    }
    return true;
}

function validateNumDiscussionMessages() {
    const input = document.getElementById("num_discussion_messages");
    const num = parseInt(input.value, 10);
    if (Number.isNaN(num) || num < 1 || num !== parseFloat(input.value)) {
        alert("Number of messages per discussion must be a whole number starting at 1.");
        input.focus();
        return false;
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
 * Checks which API providers are used in the current agent config
 * and ensures the corresponding API key field or env key is available.
 * @returns {boolean} true if all required keys are present
 */
function validateApiKeys() {
    var usedProviders = new Set();
    document.querySelectorAll('select[name^="agent_"][name$="_model"]').forEach(function (sel) {
        var val = sel.value;
        if (val && val.indexOf(":") !== -1) {
            usedProviders.add(val.split(":")[0]);
        }
    });

    var providerToField = {
        navigator: { field: "navigator_api_key", label: "Navigator (UF)", envId: "nav_key_status" },
        anthropic: { field: "anthropic_api_key", label: "Anthropic", envId: "anth_key_status" },
        openai:    { field: "openai_api_key",    label: "OpenAI",    envId: "oai_key_status" }
    };

    var missing = [];
    usedProviders.forEach(function (provider) {
        var info = providerToField[provider];
        if (!info) return;
        var fieldEl = document.getElementById(info.field);
        var fieldVal = fieldEl ? fieldEl.value.trim() : "";
        var statusEl = document.getElementById(info.envId);
        var envAvailable = statusEl && statusEl.dataset.envSet === "true";
        if (!fieldVal && !envAvailable) {
            missing.push(info.label);
        }
    });

    if (missing.length > 0) {
        alert("Missing API key(s) for: " + missing.join(", ") + ".\nProvide them in the API Keys section or set them in .env.");
        return false;
    }
    return true;
}

/**
 * Form submit handler: validate rounds, Byzantine count, and API keys before submit.
 */
function onConfigSubmit(e) {
    if (!validateRounds()) {
        e.preventDefault();
        return false;
    }
    if (!validateTicks()) {
        e.preventDefault();
        return false;
    }
    if (!validateNumDiscussionMessages()) {
        e.preventDefault();
        return false;
    }
    if (!validateByzantineCount()) {
        e.preventDefault();
        return false;
    }
    if (!validateApiKeys()) {
        e.preventDefault();
        return false;
    }
    return true;
}

/**
 * Check which API keys are available in .env and update status indicators.
 */
function checkApiKeyStatus() {
    fetch("/api/check_api_keys")
        .then(function (res) { return res.json(); })
        .then(function (data) {
            var mapping = {
                navigator: "nav_key_status",
                anthropic: "anth_key_status",
                openai: "oai_key_status"
            };
            Object.keys(mapping).forEach(function (provider) {
                var el = document.getElementById(mapping[provider]);
                if (el) {
                    var available = data[provider];
                    el.textContent = available ? "(.env set)" : "(not set)";
                    el.style.color = available ? "#4CAF50" : "#999";
                    el.dataset.envSet = available ? "true" : "false";
                }
            });
        })
        .catch(function () {});
}

// ---------------------------------------------------------------------------
// Prompt modal logic
// ---------------------------------------------------------------------------

function updatePromptEditorLabel() {
    var roleLabel = activePromptRole === "honest" ? "Honest" : "Byzantine";
    var phaseLabel;
    if (activePromptPhase === "system") phaseLabel = "System Prompt";
    else if (activePromptPhase === "movement") phaseLabel = "Movement (Action) Prompt";
    else if (activePromptPhase === "discussion") phaseLabel = "Discussion Prompt";
    else phaseLabel = "Voting Prompt";

    var labelEl = document.getElementById("promptEditorLabel");
    if (labelEl) {
        labelEl.textContent = roleLabel + " – " + phaseLabel;
    }
}

function updatePromptContext() {
    var lockedEl = document.getElementById("promptLocked");
    if (!lockedEl) return;
    var lockCtx = LOCKED_CONTEXT[activePromptRole] || {};
    lockedEl.textContent = lockCtx[activePromptPhase] || "";
}

function loadPromptEditorFromState() {
    var textarea = document.getElementById("promptEditorTextarea");
    if (!textarea) return;
    var roleObj = currentPrompts[activePromptRole] || {};
    var val = roleObj[activePromptPhase] || "";
    textarea.value = val;
    updatePromptEditorLabel();
    updatePromptContext();
}

function setActivePromptRole(role) {
    activePromptRole = role;
    var roleTabs = document.querySelectorAll(".prompt-role-tab");
    roleTabs.forEach(function (btn) {
        if (btn.dataset.role === role) btn.classList.add("active");
        else btn.classList.remove("active");
    });
    loadPromptEditorFromState();
}

function setActivePromptPhase(phase) {
    activePromptPhase = phase;
    var phaseTabs = document.querySelectorAll(".prompt-phase-tab");
    phaseTabs.forEach(function (btn) {
        if (btn.dataset.phase === phase) btn.classList.add("active");
        else btn.classList.remove("active");
    });
    loadPromptEditorFromState();
}

function openPromptEditor() {
    var container = document.getElementById("promptEditorContainer");
    if (!container) return;
    container.style.display = "block";
    // reset active selection when opening
    activePromptRole = "honest";
    activePromptPhase = "system";
    var roleTabs = document.querySelectorAll(".prompt-role-tab");
    roleTabs.forEach(function (btn) {
        btn.classList.toggle("active", btn.dataset.role === activePromptRole);
    });
    var phaseTabs = document.querySelectorAll(".prompt-phase-tab");
    phaseTabs.forEach(function (btn) {
        btn.classList.toggle("active", btn.dataset.phase === activePromptPhase);
    });
    loadPromptEditorFromState();
}

function togglePromptEditor() {
    var container = document.getElementById("promptEditorContainer");
    if (!container) return;
    if (container.style.display === "none" || container.style.display === "") {
        openPromptEditor();
    } else {
        container.style.display = "none";
    }
}

function resetCurrentPromptToDefault() {
    var roleDefaults = DEFAULT_PROMPTS[activePromptRole] || {};
    var defaultVal = roleDefaults[activePromptPhase] || "";
    if (!currentPrompts[activePromptRole]) {
        currentPrompts[activePromptRole] = {};
    }
    currentPrompts[activePromptRole][activePromptPhase] = defaultVal;
    loadPromptEditorFromState();
    setConfirmPromptsStatus("");
}

function setConfirmPromptsStatus(message, isError) {
    var el = document.getElementById("confirmPromptsStatus");
    if (!el) return;
    el.textContent = message || "";
    el.className = "confirm-prompts-status" + (message ? (isError ? " confirm-prompts-status--error" : " confirm-prompts-status--ok") : "");
}

function confirmAllPrompts() {
    var hidden = document.getElementById("custom_prompts_json");
    if (!hidden) return;
    setConfirmPromptsStatus("");

    function promptsEqual(a, b) {
        var roles = ["honest", "byzantine"];
        var phases = ["system", "movement", "discussion", "voting"];
        for (var r = 0; r < roles.length; r++) {
            var role = roles[r];
            if (!a[role] || !b[role]) return false;
            for (var p = 0; p < phases.length; p++) {
                var phase = phases[p];
                var av = (a[role][phase] || "").trim();
                var bv = (b[role][phase] || "").trim();
                if (av !== bv) return false;
            }
        }
        return true;
    }

    try {
        if (promptsEqual(currentPrompts, DEFAULT_PROMPTS)) {
            hidden.value = "";
            setConfirmPromptsStatus("Using default prompts.");
        } else {
            var serialized = JSON.stringify(currentPrompts);
            hidden.value = serialized;
            var parsed = JSON.parse(serialized);
            if (parsed && typeof parsed === "object" && (parsed.honest || parsed.byzantine)) {
                setConfirmPromptsStatus("Confirmed! Your prompts will be sent when you launch.");
            } else {
                setConfirmPromptsStatus("Prompts saved but structure invalid.", true);
            }
        }
    } catch (e) {
        hidden.value = "";
        setConfirmPromptsStatus("Could not save prompts.", true);
    }
}

// Initialize on DOM ready: build default 4-agent table and bind CONFIRM + form
window.addEventListener("DOMContentLoaded", function () {
    filterModelsByMode();
    checkApiKeyStatus();

    var confirmBtn = document.getElementById("confirmAgentsBtn");
    if (confirmBtn) {
        confirmBtn.addEventListener("click", generateAgentTable);
    }

    var form = document.getElementById("configForm");
    if (form) {
        form.addEventListener("submit", onConfigSubmit);
    }

    var roleTabs = document.querySelectorAll(".prompt-role-tab");
    roleTabs.forEach(function (btn) {
        btn.addEventListener("click", function () {
            setActivePromptRole(btn.dataset.role);
        });
    });

    var phaseTabs = document.querySelectorAll(".prompt-phase-tab");
    phaseTabs.forEach(function (btn) {
        btn.addEventListener("click", function () {
            setActivePromptPhase(btn.dataset.phase);
        });
    });

    var editor = document.getElementById("promptEditorTextarea");
    if (editor) {
        editor.addEventListener("input", function () {
            if (!currentPrompts[activePromptRole]) {
                currentPrompts[activePromptRole] = {};
            }
            currentPrompts[activePromptRole][activePromptPhase] = editor.value;
            setConfirmPromptsStatus("");
        });
    }

    var resetPromptBtn = document.getElementById("resetPromptBtn");
    if (resetPromptBtn) {
        resetPromptBtn.addEventListener("click", resetCurrentPromptToDefault);
    }

    var confirmPromptsBtn = document.getElementById("confirmPromptsBtn");
    if (confirmPromptsBtn) {
        confirmPromptsBtn.addEventListener("click", confirmAllPrompts);
    }

    // Initialize prompt editor with defaults on load
    loadPromptEditorFromState();
});