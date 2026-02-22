# config/settings.py
# Global settings and game constants (map, rounds, agent counts, quantization flags) used across the project.

NUM_ROUNDS = 10 # 7 total rounds for real simulations || front end now handles this, but just kept it here anyways
MAX_MOVEMENT_PHASES = 4  # How many "ticks" of movement occur before a forced check/pause

# new map with new names
ROOMS = {
    # --- left side ---
    "Clock": [
        "Logs", 
        "Air cooling",  
        "Liquid cooling"  
    ],
    "Air cooling": [  
        "Clock", 
        "Logs", 
        "Diagnostics", 
        "Cpu",  
        "Liquid cooling"  
    ],
    "Liquid cooling": [  
        "Clock", 
        "Logs", 
        "Bus", 
        "Ssd",  
        "Air cooling"  
    ],
    "Logs": [
        "Clock", 
        "Air cooling",  
        "Liquid cooling"  
    ],

    "Diagnostics": [
        "Air cooling",  
        "Cpu"  
    ],
    "Bus": [
        "Liquid cooling",  
        "Ssd"  
    ],
    "Cpu": [  
        "Air cooling",  
        "Diagnostics", 
        "Gpu",  
        "Bios",  
        "Ssd"  
    ],
    "Bios": [  
        "Cpu",  
        "Ssd"  
    ],
    "Ssd": [  
        "Cpu", 
        "Bios", 
        "Bus", 
        "Liquid cooling",  
        "Firewall", 
        "Io"  
    ],
    "Io": [  
        "Ssd",  
        "Firewall"
    ],

    # --- right side ---
    "Gpu": [  
        "Cpu",  
        "Vrm",  
        "Network",
        "Firewall"
    ],
    "Vrm": [  
        "Gpu",  
        "Network", 
        "Firewall"
    ],
    "Network": [
        "Gpu",  
        "Vrm",  
        "Firewall"
    ],
    "Firewall": [
        "Network", 
        "Vrm",  
        "Ssd",  
        "Io",  
        "Gpu"  
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