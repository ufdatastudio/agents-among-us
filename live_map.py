import pygame
import json
import os
import math
import subprocess
import sys

# --- CONFIGURATION ---
WIDTH, HEIGHT = 2500, 1300
BACKGROUND_COLOR = (15, 15, 35) 
ROOM_COLOR = (40, 40, 80)
ROOM_BORDER = (100, 100, 200)
TEXT_COLOR = (255, 255, 255)
CONNECTION_COLOR = (80, 80, 80)
SIDEBAR_WIDTH = 400

STATE_FILE = "live_state.json"
MAP_OFFSET_X = 275
MAP_OFFSET_Y = -60

# Global Process Handle
SIMULATION_PROCESS = None

ROOM_COORDS = {
    # Left Side (Engines/Reactor)
    "Reactor": (200, 600),
    "UpperEngine": (450, 300),
    "LowerEngine": (450, 900),
    "Security": (550, 600),      # Nested between engines
    
    # Center-Left
    "MedBay": (750, 450),
    "Electrical": (750, 650),

    # Center (Spine)
    "Cafeteria": (1050, 400),
    "Admin": (1400, 600),        # Right of the main vertical hallway
    "Storage": (1050, 700),
    
    # Right Side
    "Weapons": (1650, 300),
    "O2": (1550, 600),
    "Navigation": (2100, 600),   # Far right point
    "Shields": (1650, 850),
    "Communications": (1400, 900)
}

ADJUSTED_COORDS = {k: (v[0] + MAP_OFFSET_X, v[1] + MAP_OFFSET_Y) for k, v in ROOM_COORDS.items()}

# --- UPDATED: Visual Connections ---
CONNECTIONS = [
    # Reactor Block
    ("Reactor", "UpperEngine"),
    ("Reactor", "LowerEngine"),
    ("Reactor", "Security"),
    ("Security", "UpperEngine"),
    ("Security", "LowerEngine"),
    ("UpperEngine", "LowerEngine"),
    
    # Upper Path
    ("UpperEngine", "MedBay"),
    ("UpperEngine", "Cafeteria"),
    ("MedBay", "Cafeteria"),
    
    # Lower Path
    ("LowerEngine", "Electrical"),
    ("LowerEngine", "Storage"),
    ("Electrical", "Storage"),
    
    # Center Spine
    ("Cafeteria", "Admin"),
    ("Cafeteria", "Storage"),
    ("Admin", "Storage"),
    
    # Right Side Connectivity
    ("Cafeteria", "Weapons"),
    ("Weapons", "O2"),
    ("Weapons", "Navigation"),
    ("Weapons", "Shields"), 
    
    ("O2", "Navigation"),
    ("O2", "Shields"),
    ("Navigation", "Shields"),
    
    # Bottom Right
    ("Storage", "Shields"),
    ("Storage", "Communications"),
    ("Shields", "Communications")
]

COLOR_MAP = {
    "🔴": (231, 76, 60), "🔵": (52, 152, 219), "🟢": (46, 204, 113),
    "💗": (255, 105, 180), "🟠": (230, 126, 34), "🟡": (241, 196, 15),
    "⚫": (80, 80, 80), "⚪": (236, 240, 241), "🟣": (155, 89, 182),
    "🟤": (121, 85, 72)
}

def load_state():
    if not os.path.exists(STATE_FILE): return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return None

def draw_skull(screen, x, y, color=(200, 200, 200), size=1.0):
    pygame.draw.circle(screen, color, (x, y), 10 * size)
    rect_w, rect_h = 12 * size, 8 * size
    pygame.draw.rect(screen, color, (x - rect_w/2, y + 4 * size, rect_w, rect_h))
    eye_color = (0, 0, 0)
    pygame.draw.circle(screen, eye_color, (x - 4*size, y + 1*size), 3 * size)
    pygame.draw.circle(screen, eye_color, (x + 4*size, y + 1*size), 3 * size)

def draw_map_background(screen, font):
    for start, end in CONNECTIONS:
        if start in ADJUSTED_COORDS and end in ADJUSTED_COORDS:
            pygame.draw.line(screen, CONNECTION_COLOR, ADJUSTED_COORDS[start], ADJUSTED_COORDS[end], 8)

    for name, (x, y) in ADJUSTED_COORDS.items():
        pygame.draw.circle(screen, (30, 30, 60), (x, y), 55)
        pygame.draw.circle(screen, ROOM_COLOR, (x, y), 50)
        pygame.draw.circle(screen, ROOM_BORDER, (x, y), 50, 3)
        text = font.render(name, True, (180, 180, 200))
        text_rect = text.get_rect(center=(x, y - 65))
        screen.blit(text, text_rect)

def draw_agents(screen, data):
    if not data: return
    agents = data["agents"]
    rooms_data = data["rooms"]
    room_counts = {name: 0 for name in ROOM_COORDS}

    for room_name, r_data in rooms_data.items():
        if room_name not in ADJUSTED_COORDS: continue
        base_x, base_y = ADJUSTED_COORDS[room_name]
        
        # Dead Bodies
        for body_name in r_data["bodies"]:
            c_str = agents.get(body_name, {}).get("color", "⚪")
            rgb = COLOR_MAP.get(c_str, (200, 200, 200))
            idx = room_counts[room_name]
            off_x, off_y = (idx % 3) * 30 - 30, (idx // 3) * 30 - 10
            draw_skull(screen, base_x + off_x, base_y + off_y, color=rgb, size=1.2)
            room_counts[room_name] += 1

        # Live Agents
        for agent_name in r_data["occupants"]:
            agent_data = agents.get(agent_name)
            if not agent_data: continue
            c_str = agent_data.get("color", "⚪")
            rgb = COLOR_MAP.get(c_str, (255, 255, 255))
            idx = room_counts[room_name]
            off_x, off_y = (idx % 3) * 30 - 30, (idx // 3) * 30 - 10
            pygame.draw.circle(screen, rgb, (base_x + off_x, base_y + off_y), 14)
            pygame.draw.circle(screen, (0,0,0), (base_x + off_x, base_y + off_y), 14, 2)
            room_counts[room_name] += 1

def draw_sidebar(screen, data, font, title_font):
    if not data: return
    sidebar_rect = pygame.Rect(0, 80, SIDEBAR_WIDTH, HEIGHT - 230)
    pygame.draw.rect(screen, (25, 25, 45), sidebar_rect)
    pygame.draw.line(screen, (100, 100, 100), (SIDEBAR_WIDTH, 80), (SIDEBAR_WIDTH, HEIGHT - 150), 2)

    t_surf = title_font.render("AGENT STATUS", True, (255, 255, 255))
    screen.blit(t_surf, (20, 90))
    
    col_agent_x, col_role_x, col_vote_x, col_stat_x = 50, 160, 260, 320
    h_color = (150, 150, 150)
    screen.blit(font.render("Agent", True, h_color), (col_agent_x, 130))
    screen.blit(font.render("Role", True, h_color), (col_role_x, 130))
    screen.blit(font.render("Votes", True, h_color), (col_vote_x, 130))
    screen.blit(font.render("Status", True, h_color), (col_stat_x, 130))
    pygame.draw.line(screen, (80, 80, 80), (10, 150), (SIDEBAR_WIDTH-10, 150), 1)

    agents = sorted(data["agents"].values(), key=lambda x: int(x["name"].split("_")[1]))
    y = 170
    for ag in agents:
        name, status, role = ag["name"], ag["status"], ag["role"]
        role_display = "Byz" if role == "byzantine" else "Hon"
        c_str = ag.get("color", "⚪")
        rgb = COLOR_MAP.get(c_str, (255, 255, 255))
        votes = ag["stats"].get("votes_received", 0)
        is_active = (status == "active")
        text_color = (255, 255, 255) if is_active else (80, 80, 80)
        
        pygame.draw.circle(screen, rgb if is_active else (80,80,80), (30, y + 10), 8)
        screen.blit(font.render(name, True, text_color), (col_agent_x, y))
        role_color = (255, 100, 100) if role == "byzantine" and is_active else text_color
        screen.blit(font.render(role_display, True, role_color), (col_role_x, y))
        v_color = (255, 255, 0) if votes > 0 else text_color
        screen.blit(font.render(str(votes), True, v_color), (col_vote_x + 10, y))
        
        stat_txt, stat_col = "ALIVE", (0, 255, 0)
        if status == "eliminated": stat_txt, stat_col = "DEAD", (200, 50, 50)
        elif status == "ejected": stat_txt, stat_col = "EJECTED", (200, 150, 50)
        if not is_active: stat_col = (100, 100, 100)
        screen.blit(font.render(stat_txt, True, stat_col), (col_stat_x, y))
        y += 35

def draw_log_panel(screen, data, font, bold_font):
    if not data: return
    panel_rect = pygame.Rect(0, HEIGHT - 200, WIDTH, 200)
    pygame.draw.rect(screen, (20, 20, 20), panel_rect)
    pygame.draw.line(screen, (255, 255, 255), (0, HEIGHT - 200), (WIDTH, HEIGHT - 200), 2)
    
    title = bold_font.render("Live Feed", True, (200, 200, 200))
    screen.blit(title, (20, HEIGHT - 190))
    
    event_log = data["global"].get("ui_event_log", [])
    y_pos = HEIGHT - 30
    for event in reversed(event_log):
        if y_pos < HEIGHT - 170: break
        msg, cat = event["msg"], event["type"]
        color = (200, 200, 200)
        if cat == "kill": color = (255, 50, 50)
        elif cat == "meeting": color = (255, 255, 0)
        elif cat == "eject": color = (255, 165, 0)
        elif cat == "vote": color = (100, 200, 255)
        elif cat == "chat": color = (100, 255, 255)
        line_surf = font.render(f"[{event['time']}] {msg}", True, color)
        screen.blit(line_surf, (20, y_pos))
        y_pos -= 25

def draw_header(screen, data, font):
    pygame.draw.rect(screen, (40, 40, 60), (0, 0, WIDTH, 80))
    gid, round_num, phase = "???", "0", "IDLE"
    if data:
        gid = data.get("game_id", "???")
        round_num = data["global"].get("round", 0)
        phase = data["global"].get("current_phase", "UNKNOWN")

    phase_color = (50, 50, 80)
    if phase == "MOVEMENT": phase_color = (50, 200, 50) 
    elif phase == "DISCUSSION": phase_color = (200, 200, 0)
    elif phase == "VOTING": phase_color = (255, 50, 50)
    elif phase == "GAME OVER": phase_color = (100, 100, 100)

    screen.blit(font.render(f"SIM ID: {gid}", True, (150, 150, 150)), (20, 25))
    screen.blit(font.render(f"ROUND: {round_num}", True, (255, 255, 255)), (300, 25))
    
    pygame.draw.rect(screen, phase_color, (WIDTH - 520, 20, 200, 40), border_radius=5)
    p_surf = font.render(f"{phase}", True, (255, 255, 255))
    text_rect = p_surf.get_rect(center=(WIDTH - 420, 40))
    screen.blit(p_surf, text_rect)

# --- CORRECTED BUTTON DRAWING ---
def draw_buttons(screen, font, start_rect, clear_rect):
    global SIMULATION_PROCESS
    is_running = SIMULATION_PROCESS is not None and SIMULATION_PROCESS.poll() is None

    # 1. Start/Stop Button
    btn_color = (200, 50, 50) if is_running else (50, 200, 50)
    btn_text = "STOP SIM" if is_running else "START SIM"
    
    pygame.draw.rect(screen, btn_color, start_rect, border_radius=5)
    s_surf = font.render(btn_text, True, (255, 255, 255))
    s_rect = s_surf.get_rect(center=start_rect.center)
    screen.blit(s_surf, s_rect)

    # 2. Clear Button
    pygame.draw.rect(screen, (50, 100, 150), clear_rect, border_radius=5)
    c_surf = font.render("CLEAR", True, (255, 255, 255))
    c_rect = c_surf.get_rect(center=clear_rect.center)
    screen.blit(c_surf, c_rect)

def run():
    global SIMULATION_PROCESS
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Among Us AI - Live Map")
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont("Consolas", 18)
    header_font = pygame.font.SysFont("Arial", 28, bold=True)
    btn_font = pygame.font.SysFont("Arial", 16, bold=True)
    log_font = pygame.font.SysFont("Consolas", 14)

    # --- FIX: DEFINE RECTS OUTSIDE LOOP ---
    btn_w, btn_h = 140, 40
    start_x = WIDTH - 300
    clear_x = WIDTH - 150
    y = 20
    
    # These now persist and can be clicked
    start_btn_rect = pygame.Rect(start_x, y, btn_w, btn_h)
    clear_btn_rect = pygame.Rect(clear_x, y, btn_w, btn_h)

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click Only
                    if start_btn_rect.collidepoint(event.pos):
                        # TOGGLE SIMULATION
                        if SIMULATION_PROCESS is not None and SIMULATION_PROCESS.poll() is None:
                            # STOP
                            SIMULATION_PROCESS.kill()
                            SIMULATION_PROCESS = None
                            print("[UI] Simulation Aborted.")
                        else:
                            # START
                            print("[UI] Starting main.py...")
                            # Use sys.executable to ensure we use the same python env
                            SIMULATION_PROCESS = subprocess.Popen([sys.executable, "main.py"])
                    
                    elif clear_btn_rect.collidepoint(event.pos):
                        # CLEAR UI
                        print("[UI] Clearing Data...")
                        if os.path.exists(STATE_FILE):
                            try:
                                os.remove(STATE_FILE)
                            except Exception as e:
                                print(f"Error removing state file: {e}")
        
        # 2. Load Data
        data = load_state()
        screen.fill(BACKGROUND_COLOR)
        
        # 3. Draw Elements
        draw_header(screen, data, header_font)
        
        # Pass the pre-defined rects to the draw function
        draw_buttons(screen, btn_font, start_btn_rect, clear_btn_rect)

        if data:
            draw_map_background(screen, font)
            draw_agents(screen, data)
            draw_sidebar(screen, data, font, header_font)
            draw_log_panel(screen, data, log_font, header_font)
        else:
            txt = header_font.render("Ready to Start Simulation...", True, (100, 100, 100))
            screen.blit(txt, (WIDTH//2 - 200, HEIGHT//2))

        pygame.display.flip()
        clock.tick(5)

    # Cleanup on exit
    if SIMULATION_PROCESS:
        SIMULATION_PROCESS.kill()
    pygame.quit()

if __name__ == "__main__":
    run()