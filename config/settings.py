# config/settings.py

NUM_ROUNDS = 10 
MAX_MOVEMENT_PHASES = 4  # How many "ticks" of movement occur before a forced check/pause


ROOMS = {
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



