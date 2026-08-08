"""
Agents Among Us - Flask Web Application
Complete backend integration with all API routes + ML Classifiers
"""

import csv
import copy
import json
import os
import random
import subprocess
import sys
import threading
from datetime import datetime

import glob
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for

from config.app_mode import get_allowed_providers, get_app_mode, should_load_dotenv

if should_load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

app = Flask(__name__)
app.secret_key = 'agents-among-us-secret-key-change-in-production'

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# paths
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MASTER_CSV = os.path.join(DATA_DIR, 'frontend_stats.csv')
LIVE_STATE_FILE = os.path.join(BACKEND_PATH, 'logs', 'live_state.json')
HUMAN_INPUT_DIR = os.path.join(BACKEND_PATH, 'logs', 'human_inputs')

current_game_process = None

# ---------------------------------------------------------------------------
# Human experiment input buffer (Checkpoint 2)
# Keeps last-submitted action/chat/vote for a given (game_id, agent_name, phase, round, tick).
# Engine wiring happens in later checkpoints.
# ---------------------------------------------------------------------------
HUMAN_INPUT_LOCK = threading.Lock()
HUMAN_INPUT_STORE = {
    "action": {},  # key -> payload
    "chat": {},    # key -> payload
    "vote": {},    # key -> payload
}

os.makedirs(DATA_DIR, exist_ok=True)


def read_stats_csv(csv_path=None):
    """
    Read frontend_stats.csv and return list of row dicts.
    Tolerates mixed column counts: older rows have 18 columns, newer rows
    have 21 or 22 (extra sgd_score, svm_score, lr_score). Normalizes so every
    row has the same keys.
    """
    path = csv_path or MASTER_CSV
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return []
            # Normalize header: if 18 cols, add sgd_score, svm_score, lr_score before last (timestamp)
            if len(header) == 18:
                header = header[:17] + ["sgd_score", "svm_score", "lr_score"] + header[17:]
            ncols = len(header)
            for row in reader:
                if len(row) == 18:
                    row = row[:17] + ["", "", ""] + row[17:18]
                elif len(row) > ncols:
                    row = row[:ncols]
                while len(row) < ncols:
                    row.append("")
                out.append(dict(zip(header, row)))
    except Exception as e:
        print(f"ERROR reading stats CSV: {e}")
    return out


STATS_CSV_GLOB = os.path.join(BACKEND_PATH, "logs", "*", "Game_*_Run0", "stats.csv")
THOUGHT_FIELDNAMES = [
    "round",
    "phase",
    "tick",
    "agent",
    "role",
    "model",
    "think",
    "output",
    "had_tags",
    "parse_ok",
]


def _game_id_from_stats_path(csv_file):
    """logs/<composition>/Game_<id>_Run0/stats.csv -> (<composition>, <id>)."""
    parts = csv_file.split(os.sep)
    composition = parts[-3]
    game_folder = parts[-2]
    game_id = game_folder.replace("Game_", "").replace("_Run0", "")
    return composition, game_id


def resolve_game_ended_at(stats_csv_path):
    """Prefer meta.json ended_at written at finalize; fallback to stats.csv mtime."""
    meta_path = os.path.join(os.path.dirname(stats_csv_path), "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ended_at = meta.get("ended_at")
            if ended_at:
                return str(ended_at)
        except Exception as e:
            print(f"WARNING: Could not read {meta_path}: {e}")
    try:
        return datetime.fromtimestamp(os.path.getmtime(stats_csv_path)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_game_log_file(game_id, filename):
    """Locate a per-game log/csv under logs/ using the common path layouts."""
    possible_paths = [
        os.path.join(BACKEND_PATH, "logs", f"Game_{game_id}", filename),
        os.path.join(BACKEND_PATH, "logs", "*", f"Game_{game_id}_Run0", filename),
        os.path.join(BACKEND_PATH, "logs", "*", f"Game_{game_id}", filename),
    ]
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def import_new_games_from_logs():
    """Scan finished games under logs/ and append any missing rows to frontend_stats.csv.

    Also backfills/corrects the timestamp column for existing games using each
    run's meta.json ended_at (or stats.csv mtime), so Refresh never stamps 'now'.

    Returns:
        int: number of newly imported games.
    """
    existing_data = read_stats_csv(MASTER_CSV)
    existing_game_ids = set(r.get("game_id") for r in existing_data if r.get("game_id"))

    csv_files = glob.glob(STATS_CSV_GLOB)
    ended_at_by_game = {}
    for csv_file in csv_files:
        _, game_id = _game_id_from_stats_path(csv_file)
        ended_at_by_game[game_id] = resolve_game_ended_at(csv_file)

    print("\nScanning for new games...")
    print(f"Found {len(csv_files)} total stats.csv files")
    print(f"Already have {len(existing_game_ids)} games in database")

    new_games = 0
    new_data = []

    for csv_file in csv_files:
        composition, game_id = _game_id_from_stats_path(csv_file)
        if game_id in existing_game_ids:
            continue
        try:
            df = pd.read_csv(csv_file)
            df.insert(0, "composition", composition)
            df.insert(1, "game_id", game_id)
            df["timestamp"] = ended_at_by_game.get(
                game_id, resolve_game_ended_at(csv_file)
            )
            new_data.append(df)
            new_games += 1
            print(f"  Added: {game_id} ({composition}) @ {df['timestamp'].iloc[0]}")
        except Exception as e:
            print(f"ERROR reading {csv_file}: {e}")

    if new_data:
        combined = pd.concat(new_data, ignore_index=True)
        if os.path.exists(MASTER_CSV):
            combined.to_csv(MASTER_CSV, mode="a", header=False, index=False)
        else:
            combined.to_csv(MASTER_CSV, mode="w", header=True, index=False)
        print(f"\nAdded {new_games} new games to database\n")
    else:
        print("\nNo new games found\n")

    # Correct timestamps for all known games (new + previously imported).
    all_rows = read_stats_csv(MASTER_CSV)
    if all_rows and ended_at_by_game:
        changed = False
        for row in all_rows:
            gid = row.get("game_id")
            if gid in ended_at_by_game and row.get("timestamp") != ended_at_by_game[gid]:
                row["timestamp"] = ended_at_by_game[gid]
                changed = True
        if changed:
            keys = list(all_rows[0].keys())
            with open(MASTER_CSV, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_rows)
            print("Updated simulation end timestamps from game meta/mtime\n")

    return new_games


def _json_error(http_status: int, code: str, message: str, details=None):
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if details is not None:
        payload["error"]["details"] = details
    return jsonify(payload), http_status


def _load_live_state():
    if not os.path.exists(LIVE_STATE_FILE):
        return None, ("STATE_MISSING", "live_state.json not found (game not started yet)")
    try:
        with open(LIVE_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError:
        return None, ("STATE_INVALID", "live_state.json contains invalid JSON")
    except Exception as e:
        return None, ("STATE_READ_FAILED", f"Could not read live_state.json: {e}")


def _normalize_phase(phase_val):
    if phase_val is None:
        return ""
    return str(phase_val).strip().upper()


def _require_fields(obj, fields):
    missing = []
    for f in fields:
        if f not in obj or obj[f] is None or (isinstance(obj[f], str) and obj[f].strip() == ""):
            missing.append(f)
    return missing


def _validate_human_submission(kind: str, payload: dict):
    """
    Strict validation for human experiment submissions.
    Ensures:
      - required fields exist
      - payload matches the running game's live_state.json
      - submitting agent is the human agent
      - phase and round match current state
    """
    required = ["game_id", "agent_name", "phase", "round", "tick", "action", "target"]
    missing = _require_fields(payload, required)
    if missing:
        return ("MISSING_FIELDS", "Missing required fields", {"missing": missing})

    state, err = _load_live_state()
    if err:
        code, msg = err
        return (code, msg, None)

    # Validate game id
    state_game_id = state.get("game_id")
    if str(payload["game_id"]) != str(state_game_id):
        return ("GAME_MISMATCH", "Payload game_id does not match running game", {"expected": state_game_id})

    g = state.get("global", {}) or {}
    current_phase = _normalize_phase(g.get("current_phase"))
    payload_phase = _normalize_phase(payload.get("phase"))
    if payload_phase != current_phase:
        return ("PHASE_MISMATCH", "Payload phase does not match current phase", {"expected": current_phase})

    try:
        payload_round = int(payload.get("round"))
    except Exception:
        return ("BAD_ROUND", "Field 'round' must be an integer", None)
    current_round = g.get("round", 0)
    if int(current_round) != payload_round:
        return ("ROUND_MISMATCH", "Payload round does not match current round", {"expected": current_round})

    agents = state.get("agents", {}) or {}
    agent_name = str(payload.get("agent_name"))
    agent_state = agents.get(agent_name)
    if not agent_state:
        return ("UNKNOWN_AGENT", "agent_name not found in state", None)

    human_enabled = bool(g.get("human_experiment", False))
    human_agent = g.get("human_agent")
    if not human_enabled or not human_agent:
        return ("HUMAN_MODE_OFF", "Human experiment is not enabled for this game", None)
    if agent_name != str(human_agent):
        return ("NOT_HUMAN_AGENT", "agent_name is not the configured human agent", {"human_agent": human_agent})
    if not bool(agent_state.get("is_human", False)):
        return ("AGENT_NOT_MARKED_HUMAN", "Agent is not marked is_human in state", None)

    # Basic sanity for tick
    try:
        _ = int(payload.get("tick"))
    except Exception:
        return ("BAD_TICK", "Field 'tick' must be an integer", None)

    # Kind-specific action sanity (minimal for now; detailed legality checks come with engine wiring)
    action = str(payload.get("action")).strip().lower()
    if kind == "action":
        allowed = {"move", "stay", "report", "button", "kill", "tag"}
        if action not in allowed:
            return ("BAD_ACTION", "Invalid action", {"allowed": sorted(list(allowed))})
    elif kind == "chat":
        if action != "say":
            return ("BAD_ACTION", "Chat submissions must use action='say'", {"expected": "say"})
    elif kind == "vote":
        if action != "vote":
            return ("BAD_ACTION", "Vote submissions must use action='vote'", {"expected": "vote"})

    return None


def _store_human_input(kind: str, payload: dict):
    key = (
        str(payload.get("game_id")),
        str(payload.get("agent_name")),
        _normalize_phase(payload.get("phase")),
        int(payload.get("round")),
        int(payload.get("tick")),
        kind,
    )
    with HUMAN_INPUT_LOCK:
        HUMAN_INPUT_STORE[kind][key] = {
            "received_at": datetime.now().isoformat(timespec="seconds"),
            **payload,
        }
        # Also persist to disk so the simulation subprocess (main.py) can consume it.
        # This enables single-player now and keeps the interface compatible with
        # future multiplayer transport changes.
        try:
            os.makedirs(HUMAN_INPUT_DIR, exist_ok=True)
            fp = os.path.join(HUMAN_INPUT_DIR, f"{payload.get('game_id')}.jsonl")
            line_obj = {"kind": kind, **payload, "received_at": datetime.now().isoformat(timespec="seconds")}
            with open(fp, "a", encoding="utf-8") as f:
                f.write(json.dumps(line_obj) + "\n")
        except Exception as e:
            # Non-fatal: keep in-memory acceptance semantics.
            print(f"WARNING: Could not persist human input: {e}")
    return key


@app.route('/api/human/action', methods=['POST'])
def human_action():
    payload = request.get_json(silent=True) or {}
    err = _validate_human_submission("action", payload)
    if err:
        code, msg, details = err
        # use 422 for semantic validation issues; 409 for phase/round mismatch
        status = 409 if code in {"PHASE_MISMATCH", "ROUND_MISMATCH", "GAME_MISMATCH"} else 422
        return _json_error(status, code, msg, details)
    key = _store_human_input("action", payload)
    return jsonify({"ok": True, "status": "accepted", "kind": "action", "key": list(key)})


@app.route('/api/human/chat', methods=['POST'])
def human_chat():
    payload = request.get_json(silent=True) or {}
    err = _validate_human_submission("chat", payload)
    if err:
        code, msg, details = err
        status = 409 if code in {"PHASE_MISMATCH", "ROUND_MISMATCH", "GAME_MISMATCH"} else 422
        return _json_error(status, code, msg, details)
    key = _store_human_input("chat", payload)
    return jsonify({"ok": True, "status": "accepted", "kind": "chat", "key": list(key)})


@app.route('/api/human/vote', methods=['POST'])
def human_vote():
    payload = request.get_json(silent=True) or {}
    err = _validate_human_submission("vote", payload)
    if err:
        code, msg, details = err
        status = 409 if code in {"PHASE_MISMATCH", "ROUND_MISMATCH", "GAME_MISMATCH"} else 422
        return _json_error(status, code, msg, details)
    key = _store_human_input("vote", payload)
    return jsonify({"ok": True, "status": "accepted", "kind": "vote", "key": list(key)})




@app.route('/') 
def index():
    return render_template('index.html')


@app.route('/config')
def config():
    return render_template('config.html', app_mode=get_app_mode())


@app.route('/game')
def game():
    game_id = session.get('game_id', 'unknown')
    num_agents = session.get('num_agents', 0)
    num_rounds = session.get('num_rounds', 0)
    composition = session.get('composition', '')
    
    byzantine_count = 0
    honest_count = 0
    if composition:
        try:
            comp_file = os.path.join(BACKEND_PATH, 'config', f'{composition}.json')
            if os.path.exists(comp_file):
                with open(comp_file, 'r') as f:
                    comp_data = json.load(f)
                    byzantine_count = comp_data.get('byzantine_count', 0)
                    honest_count = comp_data.get('honest_count', 0)
        except:
            pass
    
    return render_template('game.html', 
                         game_id=game_id,
                         num_agents=num_agents,
                         num_rounds=num_rounds,
                         byzantine_count=byzantine_count,
                         honest_count=honest_count)


@app.route('/stats')
def stats():
    return render_template('stats.html')


@app.route('/win')
def win():
    winner = request.args.get('winner', 'Unknown')
    return render_template('win.html', winner=winner)

@app.route('/start_game', methods=['POST'])
def start_game():
    """Launch the backend simulation with custom configuration"""
    global current_game_process
    
    try:
        # get form data
        num_agents = int(request.form.get('num_agents', 4))
        num_rounds = int(request.form.get('num_rounds', 10))
        num_ticks = int(request.form.get('num_ticks', 4))
        num_discussion_messages = int(request.form.get('num_discussion_messages', 2))
        game_id = request.form.get('game_id', '').strip()
        human_experiment = request.form.get('human_experiment') == 'true'
        human_agent = "Agent_0" if human_experiment else None
        requested_num_byzantines = None
        randomized_byzantines = set()

        # Thought-capture master switch from config UI.
        # Require-tag retry is always enabled whenever capture is on.
        capture_vals = request.form.getlist("capture_thoughts")
        if capture_vals:
            capture_thoughts = capture_vals[-1] == "true"
        else:
            capture_thoughts = True
        require_think_tags = capture_thoughts
        
        # === NEW: Get ML Classifier selections ===
        classifier_sgd = request.form.get('classifier_sgd') == 'true'
        classifier_svm = request.form.get('classifier_svm') == 'true'
        classifier_lr = request.form.get('classifier_lr') == 'true'
        
        enabled_classifiers = {
            'sgd': classifier_sgd,
            'svm': classifier_svm,
            'lr': classifier_lr
        }
        
        # auto-generate game_id if empty
        if not game_id:
            game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # parse agent configurations - PRESERVE EXACT ORDER AND AGENT NUMBERS
        agents = []
        byzantine_count = 0
        honest_count = 0

        if human_experiment:
            requested_num_byzantines = int(request.form.get('num_byzantines', 1))
            max_byz = (num_agents - 1) // 2
            if requested_num_byzantines < 1 or requested_num_byzantines > max_byz:
                return jsonify({
                    "error": f"Invalid num_byzantines={requested_num_byzantines}. Must be between 1 and {max_byz} for {num_agents} agents."
                }), 400
            randomized_byzantines = set(random.sample(range(num_agents), requested_num_byzantines))
        
        for i in range(num_agents):
            role = request.form.get(f'agent_{i}_role')
            if human_experiment:
                role = 'byzantine' if i in randomized_byzantines else 'honest'
                is_hybrid = False
            else:
                is_hybrid = role == 'honest' and request.form.get(f'agent_{i}_is_hybrid') == 'true'
            agent_name = f"Agent_{i}"
            agent = {
                'agent_num': i,  # Preserve exact agent number
                'model': request.form.get(f'agent_{i}_model'),
                'role': role,
                'color': request.form.get(f'agent_{i}_color'),
                'is_hybrid': is_hybrid,
                'is_human': bool(human_experiment and agent_name == human_agent),
            }
            agents.append(agent)
            
            if agent['role'] == 'byzantine':
                byzantine_count += 1
            else:
                honest_count += 1
        
        # create custom composition JSON with FULL agent configuration + classifiers
        composition = {
            "name": f"custom_{game_id}",
            "honest_count": honest_count,
            "byzantine_count": byzantine_count,
            "agents": agents,  # Full per-agent configuration
            "num_rounds": num_rounds,
            "num_ticks": num_ticks,
            "num_discussion_messages": num_discussion_messages,
            "enabled_classifiers": enabled_classifiers,
            "human_experiment": human_experiment,
            "human_agent": human_agent,
            "capture_thoughts": capture_thoughts,
            "require_think_tags": require_think_tags,
        }
        if human_experiment:
            composition["roles_randomized"] = True
            composition["num_byzantines"] = requested_num_byzantines

        # Optional: custom per-role, per-phase prompts (frontend overrides)
        raw_prompts = request.form.get('custom_prompts_json', '').strip()
        if raw_prompts:
            try:
                prompts_cfg = json.loads(raw_prompts)
                if isinstance(prompts_cfg, dict):
                    composition["prompts"] = prompts_cfg
            except Exception as e:
                print(f"WARNING: Failed to parse custom_prompts_json: {e}")
        
        # save composition to logs/ (writable in both local and container environments)
        game_configs_dir = os.path.join(BACKEND_PATH, 'logs', 'game_configs')
        os.makedirs(game_configs_dir, exist_ok=True)
        composition_file = os.path.join(game_configs_dir, f'custom_{game_id}.json')
        with open(composition_file, 'w') as f:
            json.dump(composition, f, indent=2)
        
        # Debug: Print configuration (compact summary + lineup)
        print(f"\n{'='*60}")
        print(f"GAME CONFIGURATION")
        print(f"{'='*60}")

        classifiers_enabled = [k.upper() for k, v in enabled_classifiers.items() if v]
        observers_label = ", ".join(classifiers_enabled) if classifiers_enabled else "NONE"
        prompts_mode = "Custom" if "prompts" in composition else "Default"

        print(f"Game ID: {game_id}")
        print(f"Agents: {num_agents} ({byzantine_count} Byzantine, {honest_count} Honest)")
        print(f"Rounds: {num_rounds}")
        print(f"Ticks: {num_ticks}")
        print(f"Discussion messages: {num_discussion_messages}")
        print(f"Observers: {observers_label}")
        print(f"Prompts: {prompts_mode}")
        print(f"Human experiment: {'ON (Agent_0)' if human_experiment else 'OFF'}")
        if human_experiment:
            print(f"Human hidden-role mode: ON (randomized Byzantine count = {requested_num_byzantines})")
        print(f"Thought capture: {'ON' if capture_thoughts else 'OFF'} (require_think_tags={require_think_tags})")

        print("\nAgent lineup:")
        for agent in agents:
            role_label = "Byzantine" if agent['role'] == 'byzantine' else "Honest"
            hybrid_note = " | hybrid" if agent.get('is_hybrid') else ""
            print(f"  Agent_{agent['agent_num']}: {role_label}{hybrid_note} | {agent['model']} | {agent['color']}")

        
        # store in session
        session['game_id'] = game_id
        session['composition'] = f"custom_{game_id}"
        session['num_agents'] = num_agents
        session['num_rounds'] = num_rounds
        session['num_ticks'] = num_ticks
        session['num_discussion_messages'] = num_discussion_messages

        # reset live state so we don't show a previous game's snapshot
        try:
            if os.path.exists(LIVE_STATE_FILE):
                os.remove(LIVE_STATE_FILE)
            # reset human input queue for this run (if any)
            human_fp = os.path.join(BACKEND_PATH, 'logs', 'human_inputs', f"{game_id}_Run0.jsonl")
            if os.path.exists(human_fp):
                os.remove(human_fp)
        except Exception as e:
            print(f"WARNING: Could not clear live_state.json: {e}")
        
        # Build command
        cmd = [
            sys.executable,
            os.path.join(BACKEND_PATH, 'main.py'),
            '--composition_name', f'custom_{game_id}',
            '--game_id', game_id,
            '--job_index', '0',
            '--num_rounds', str(num_rounds),
            '--num_ticks', str(num_ticks),
            '--num_discussion_messages', str(num_discussion_messages)
        ]
        
        print(f"{'='*60}")
        print(f"LAUNCHING GAME: {game_id}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}")
        
        # Pass API keys to subprocess via environment
        env = os.environ.copy()
        api_key_fields = {
            'navigator_api_key': 'NAVIGATOR_TOOLKIT_API_KEY',
            'anthropic_api_key': 'ANTHROPIC_API_KEY',
            'openai_api_key': 'OPENAI_API_KEY',
        }
        for form_field, env_var in api_key_fields.items():
            val = request.form.get(form_field, '').strip()
            if val:
                env[env_var] = val

        # start game process in background
        current_game_process = subprocess.Popen(
            cmd,
            cwd=BACKEND_PATH,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # start background thread to stream output to terminal
        def stream_output(process):
            """Stream subprocess output to terminal in real-time"""
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        print(line.rstrip())
                        sys.stdout.flush()
            except Exception as e:
                print(f"ERROR streaming output: {e}")
            finally:
                try:
                    process.stdout.close()
                except Exception:
                    pass
                # Wait for finalize_stats / stats.csv to flush, then auto-import.
                try:
                    process.wait(timeout=30)
                except Exception:
                    pass
                try:
                    import time as _time
                    _time.sleep(0.75)
                    added = import_new_games_from_logs()
                    if added:
                        print(f"Auto-imported {added} finished game(s) into stats")
                except Exception as e:
                    print(f"WARNING: Auto stats import failed: {e}")
        
        output_thread = threading.Thread(target=stream_output, args=(current_game_process,), daemon=True)
        output_thread.start()
        
        return redirect(url_for('game'))
        
    except Exception as e:
        print(f"ERROR starting game: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stop_game', methods=['POST'])
def stop_game():
    """Stop the currently running game"""
    global current_game_process
    
    try:
        if current_game_process:
            current_game_process.terminate()
            current_game_process.wait(timeout=5)
            current_game_process = None
            return jsonify({'status': 'stopped'})
        else:
            return jsonify({'status': 'no_game_running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game_state')
def get_game_state():
    """Read live_state.json and return current game state"""
    global current_game_process
    
    try:
        # check if backend process is still running
        if current_game_process is not None:
            poll_result = current_game_process.poll()
            if poll_result is not None:
                if poll_result != 0:
                    current_game_process = None
                    return jsonify({
                        'status': 'error',
                        'message': f'Backend process crashed (exit code: {poll_result}). Check terminal for errors.',
                        'process_ended': True
                    })
                else:
                    current_game_process = None
        
        # check if live_state.json exists
        if not os.path.exists(LIVE_STATE_FILE):
            return jsonify({
                'status': 'waiting',
                'message': 'Waiting for game to start...'
            })
        
        with open(LIVE_STATE_FILE, 'r') as f:
            state = json.load(f)

        # Human experiments: never send private CoT to the browser unless the
        # researcher explicitly enables reveal (debug overlay / ?reveal_thoughts=1).
        # Thoughts remain on disk in live_state.json / thought.log for analysis.
        if _should_redact_thoughts_for_client(state):
            state = _redact_thoughts_from_client_state(state)
        
        return jsonify(state)
        
    except json.JSONDecodeError as e:
        return jsonify({
            'status': 'error',
            'message': 'Invalid JSON in live_state.json'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def _should_redact_thoughts_for_client(state):
    global_state = (state or {}).get("global") or {}
    if not global_state.get("human_experiment"):
        return False
    reveal = str(request.args.get("reveal_thoughts", "")).strip().lower()
    return reveal not in ("1", "true", "yes")


def _redact_thoughts_from_client_state(state):
    redacted = copy.deepcopy(state)
    redacted["thought_history"] = []
    redacted["latest_thoughts"] = {}
    global_state = redacted.setdefault("global", {})
    global_state["thoughts_redacted"] = True
    return redacted


@app.route('/api/game_status')
def get_game_status():
    """Check if game is still running"""
    global current_game_process
    
    if current_game_process:
        poll = current_game_process.poll()
        if poll is None:
            return jsonify({'running': True})
        else:
            current_game_process = None
            return jsonify({'running': False, 'exit_code': poll})
    else:
        return jsonify({'running': False})


@app.route('/api/stats/all')
def get_all_stats():
    """Return all data from frontend_stats.csv (tolerates 18 vs 21 column rows)."""
    try:
        data = read_stats_csv(MASTER_CSV)
        return jsonify(data)
    except Exception as e:
        print(f"ERROR reading stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/refresh', methods=['POST'])
def refresh_stats():
    """Scan logs/ folder for new games and append to frontend_stats.csv"""
    try:
        new_games = import_new_games_from_logs()
        return jsonify({'new_games': new_games})
    except Exception as e:
        print(f"ERROR refreshing stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/export')
def export_stats():
    """Download frontend_stats.csv"""
    try:
        if not os.path.exists(MASTER_CSV):
            return "No statistics available", 404
        
        return send_file(
            MASTER_CSV,
            mimetype='text/csv',
            as_attachment=True,
            download_name='agents_among_us_stats.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/export_game')
def export_game():
    """Download CSV for a specific game"""
    try:
        game_id = request.args.get('game_id')
        if not game_id:
            return "game_id parameter required", 400
        
        data = read_stats_csv(MASTER_CSV)
        game_rows = [r for r in data if r.get("game_id") == game_id]
        if not game_rows:
            return f"No data found for game: {game_id}", 404
        
        temp_file = os.path.join(DATA_DIR, f'temp_{game_id}.csv')
        keys = list(game_rows[0].keys())
        with open(temp_file, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(game_rows)
        return send_file(
            temp_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'game_{game_id}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/export_discussion')
def export_discussion():
    """Download discussion_chat.csv for a specific game"""
    try:
        game_id = request.args.get('game_id')
        if not game_id:
            return "game_id parameter required", 400
        
        discussion_file = find_game_log_file(game_id, "discussion_chat.csv")
        
        if not discussion_file or not os.path.exists(discussion_file):
            return f"No discussion data found for game: {game_id}", 404
        
        return send_file(
            discussion_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'discussion_{game_id}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/export_thoughts')
def export_thoughts():
    """Download agent thought capture data for a game as CSV."""
    try:
        game_id = request.args.get('game_id')
        if not game_id:
            return "game_id parameter required", 400

        thought_file = find_game_log_file(game_id, "thought.log")
        if not thought_file or not os.path.exists(thought_file):
            return f"No thoughts data found for game: {game_id}", 404

        rows = []
        with open(thought_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed thought line {line_no}: {e}")
                    continue
                rows.append({
                    "round": rec.get("round", ""),
                    "phase": rec.get("phase", ""),
                    "tick": rec.get("tick", ""),
                    "agent": rec.get("agent", ""),
                    "role": rec.get("role", ""),
                    "model": rec.get("model", ""),
                    "think": rec.get("think", ""),
                    "output": rec.get("output", ""),
                    "had_tags": rec.get("had_tags", ""),
                    "parse_ok": rec.get("parse_ok", ""),
                })

        temp_file = os.path.join(DATA_DIR, f'temp_thoughts_{game_id}.csv')
        with open(temp_file, "w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=THOUGHT_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        return send_file(
            temp_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'thoughts_{game_id}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats/clear', methods=['POST'])
def clear_stats():
    """Delete frontend_stats.csv"""
    try:
        if os.path.exists(MASTER_CSV):
            os.remove(MASTER_CSV)
            print("Deleted frontend_stats.csv")
            return jsonify({'status': 'cleared'})
        else:
            return jsonify({'status': 'no_data'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/game/<game_id>')
def get_game_stats(game_id):
    """Get stats for a specific game"""
    try:
        if not os.path.exists(MASTER_CSV):
            return jsonify([])
        
        df = pd.read_csv(MASTER_CSV)
        game_data = df[df['game_id'] == game_id]
        
        return jsonify(game_data.to_dict('records'))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/check_api_keys')
def check_api_keys():
    """Check which API keys are set in the environment."""
    return jsonify({
        'navigator': bool(os.environ.get('NAVIGATOR_TOOLKIT_API_KEY')),
        'anthropic': bool(os.environ.get('ANTHROPIC_API_KEY')),
        'openai': bool(os.environ.get('OPENAI_API_KEY')),
    })


@app.route('/api/app_mode')
def app_mode_info():
    """Return the current APP_MODE and allowed providers."""
    mode = get_app_mode()
    allowed = get_allowed_providers()
    return jsonify({
        'mode': mode,
        'allowed_providers': sorted(allowed) if allowed else None,
    })


@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'ok',
        'backend_path': BACKEND_PATH,
        'data_dir': DATA_DIR,
        'stats_exists': os.path.exists(MASTER_CSV),
        'live_state_exists': os.path.exists(LIVE_STATE_FILE)
    })


@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("AGENTS AMONG US")
    print("="*60)
    print(f"Backend Path: {BACKEND_PATH}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Stats Database: {MASTER_CSV}")
    print(f"Live State File: {LIVE_STATE_FILE}")
    print("="*60)
    print(f"Open: http://localhost:8080")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=8080, debug=False)