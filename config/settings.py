# config/settings.py
# Global settings and game constants (map, rounds, agent counts, quantization flags) used across the project.

NUM_ROUNDS = 10 # 7 total rounds for real simulations || front end now handles this, but just kept it here anyways
MAX_MOVEMENT_PHASES = 4  # How many "ticks" of movement occur before a forced check/pause

# new map with new names
ROOMS = {
    # --- left side ---
    "Clock": [
        "Logs", 
        "Air cooling",  # Changed from "Air Cooling"
        "Liquid cooling"  # Changed from "Liquid Cooling"
    ],
    "Air cooling": [  # Changed from "Air Cooling"
        "Clock", 
        "Logs", 
        "Diagnostics", 
        "Cpu",  # Changed from "CPU"
        "Liquid cooling"  # Changed from "Liquid Cooling"
    ],
    "Liquid cooling": [  # Changed from "Liquid Cooling"
        "Clock", 
        "Logs", 
        "Bus", 
        "Ssd",  # Changed from "SSD"
        "Air cooling"  # Changed from "Air Cooling"
    ],
    "Logs": [
        "Clock", 
        "Air cooling",  # Changed from "Air Cooling"
        "Liquid cooling"  # Changed from "Liquid Cooling"
    ],

    "Diagnostics": [
        "Air cooling",  # Changed from "Air Cooling"
        "Cpu"  # Changed from "CPU"
    ],
    "Bus": [
        "Liquid cooling",  # Changed from "Liquid Cooling"
        "Ssd"  # Changed from "SSD"
    ],
    "Cpu": [  # Changed from "CPU"
        "Air cooling",  # Changed from "Air Cooling"
        "Diagnostics", 
        "Gpu",  # Changed from "GPU"
        "Bios",  # Changed from "BIOS"
        "Ssd"  # Changed from "SSD"
    ],
    "Bios": [  # Changed from "BIOS"
        "Cpu",  # Changed from "CPU"
        "Ssd"  # Changed from "SSD"
    ],
    "Ssd": [  # Changed from "SSD"
        "Cpu",  # Changed from "CPU"
        "Bios",  # Changed from "BIOS"
        "Bus", 
        "Liquid cooling",  # Changed from "Liquid Cooling"
        "Firewall", 
        "Io"  # Changed from "IO"
    ],
    "Io": [  # Changed from "IO"
        "Ssd",  # Changed from "SSD"
        "Firewall"
    ],

    # --- right side ---
    "Gpu": [  # Changed from "GPU"
        "Cpu",  # Changed from "CPU"
        "Vrm",  # Changed from "VRM"
        "Network",
        "Firewall"
    ],
    "Vrm": [  # Changed from "VRM"
        "Gpu",  # Changed from "GPU"
        "Network", 
        "Firewall"
    ],
    "Network": [
        "Gpu",  # Changed from "GPU"
        "Vrm",  # Changed from "VRM"
        "Firewall"
    ],
    "Firewall": [
        "Network", 
        "Vrm",  # Changed from "VRM"
        "Ssd",  # Changed from "SSD"
        "Io",  # Changed from "IO"
        "Gpu"  # Changed from "GPU"
    ]
}

AGENT_COLORS = ["🔴", "🟠", "🟡", "🟩", "🟢", "🔷", "🔵", "🟣", "🟤", "💗", "⚪", "⚫"]
NUM_BYZ = 2
NUM_HONEST = 8


QUANTIZATION = False # FALSE WHEN ON MY MAC (SWITCH BACK TO TRUE)

# OLD SETTINGS BELOW:
'''
# config/settings.py
# Global settings and game constants (map, rounds, agent counts, quantization flags) used across the project.

NUM_ROUNDS = 10 # 7 total rounds for real simulations
MAX_MOVEMENT_PHASES = 4  # How many "ticks" of movement occur before a forced check/pause

# Map Connectivity Graph
# ROOMS = {
#     "Cafeteria": ["Weapons", "Navigation", "Storage", "Admin", "MedBay", "UpperEngine"],
#     "Weapons": ["Cafeteria", "O2"],
#     "Navigation": ["Shields", "O2"],
#     "O2": ["Weapons", "Navigation", "Cafeteria"],
#     "Shields": ["Navigation", "Communications", "Storage"],
#     "Communications": ["Shields"],
#     "Admin": ["Cafeteria", "Storage"],
#     "Storage": ["Admin", "Shields", "Electrical", "LowerEngine", "Cafeteria"],
#     "Electrical": ["Storage", "LowerEngine"],
#     "LowerEngine": ["Storage", "Electrical", "Security", "Reactor"],
#     "Security": ["LowerEngine", "UpperEngine", "Reactor"],
#     "Reactor": ["Security", "UpperEngine", "LowerEngine"],
#     "UpperEngine": ["Reactor", "Security", "MedBay", "Cafeteria"],
#     "MedBay": ["UpperEngine", "Cafeteria"]
# }

ROOMS = {
    # --- Left Side (Reactor & Engines) ---
    "Reactor": [
        "Security", 
        "UpperEngine", 
        "LowerEngine"
    ],
    "UpperEngine": [
        "Reactor", 
        "Security", 
        "MedBay", 
        "Cafeteria",
        "LowerEngine"  
    ],
    "LowerEngine": [
        "Reactor", 
        "Security", 
        "Electrical", 
        "Storage",
        "UpperEngine"  
    ],
    "Security": [
        "Reactor", 
        "UpperEngine", 
        "LowerEngine"
    ],

    "MedBay": [
        "UpperEngine", 
        "Cafeteria"
    ],
    "Electrical": [
        "LowerEngine", 
        "Storage"
    ],
    "Cafeteria": [
        "UpperEngine", 
        "MedBay", 
        "Weapons", 
        "Admin", 
        "Storage"
    ],
    "Admin": [
        "Cafeteria", 
        "Storage"
    ],
    "Storage": [
        "Cafeteria", 
        "Admin", 
        "Electrical", 
        "LowerEngine", 
        "Shields", 
        "Communications"
    ],
    "Communications": [
        "Storage", 
        "Shields"
    ],

    # --- Right Side (Navigation & Support) ---
    "Weapons": [
        "Cafeteria", 
        "O2", 
        "Navigation",
        "Shields"
    ],
    "O2": [
        "Weapons", 
        "Navigation", 
        "Shields"
    ],
    "Navigation": [
        "Weapons", 
        "O2", 
        "Shields"
    ],
    "Shields": [
        "Navigation", 
        "O2", 
        "Storage", 
        "Communications",
        "Weapons"
    ]
}

AGENT_COLORS = ["🔴", "🔵", "🟢", "💗", "🟠", "🟡", "⚫", "⚪", "🟣", "🟤"]
NUM_BYZ = 2
NUM_HONEST = 8


QUANTIZATION = False # Switch to False when on MacOS/non-NVIDIA GPU systems.
'''