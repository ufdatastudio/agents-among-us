# config/settings.py

NUM_ROUNDS = 5 # 7 total rounds for real simulations
MAX_MOVEMENT_PHASES = 4  # How many "ticks" of movement occur before a forced check/pause

# Map Connectivity Graph
ROOMS = {
    "Cafeteria": ["Weapons", "Navigation", "Storage", "Admin", "MedBay", "UpperEngine"],
    "Weapons": ["Cafeteria", "O2"],
    "Navigation": ["Shields", "O2"],
    "O2": ["Weapons", "Navigation", "Cafeteria"],
    "Shields": ["Navigation", "Communications", "Storage"],
    "Communications": ["Shields"],
    "Admin": ["Cafeteria", "Storage"],
    "Storage": ["Admin", "Shields", "Electrical", "LowerEngine", "Cafeteria"],
    "Electrical": ["Storage", "LowerEngine"],
    "LowerEngine": ["Storage", "Electrical", "Security", "Reactor"],
    "Security": ["LowerEngine", "UpperEngine", "Reactor"],
    "Reactor": ["Security", "UpperEngine", "LowerEngine"],
    "UpperEngine": ["Reactor", "Security", "MedBay", "Cafeteria"],
    "MedBay": ["UpperEngine", "Cafeteria"]
}

AGENT_COLORS = ["🔴", "🔵", "🟢", "💗", "🟠", "🟡", "⚫", "⚪", "🟣", "🟤"]
NUM_BYZ = 1
NUM_HONEST = 4
AGENT_LLM_CONFIG = [
    "meta-llama/Llama-3.1-8B-Instruct",
    # Add more models here
]

QUANTIZATION = True
LIVE_MAP = True