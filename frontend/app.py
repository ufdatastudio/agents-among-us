"""
Agents Among Us - Flask Web Application
Complete backend integration with all API routes + ML Classifiers
"""

import csv
import json
import os
import subprocess
import sys
import threading
from datetime import datetime

import glob
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for

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
LIVE_STATE_FILE = os.path.join(BACKEND_PATH, 'live_state.json')

current_game_process = None

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




@app.route('/') 
def index():
    return render_template('index.html')


@app.route('/config')
def config():
    return render_template('config.html')


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
        game_id = request.form.get('game_id', '').strip()
        
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
        
        for i in range(num_agents):
            agent = {
                'agent_num': i,  # Preserve exact agent number
                'model': request.form.get(f'agent_{i}_model'),
                'role': request.form.get(f'agent_{i}_role'),
                'color': request.form.get(f'agent_{i}_color')
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
            "enabled_classifiers": enabled_classifiers  # NEW: ML Classifiers
        }
        
        # save composition to temporary file
        game_configs_dir = os.path.join(BACKEND_PATH, 'config', 'game_configs')
        os.makedirs(game_configs_dir, exist_ok=True)
        composition_file = os.path.join(BACKEND_PATH, 'config', 'game_configs', f'custom_{game_id}.json')
        with open(composition_file, 'w') as f:
            json.dump(composition, f, indent=2)
        
        # Debug: Print configuration
        print(f"\n{'='*60}")
        print(f"AGENT CONFIGURATION SENT TO BACKEND:")
        print(f"{'='*60}")
        for agent in agents:
            role_label = "Byzantine" if agent['role'] == 'byzantine' else "Honest"
            print(f"  Agent_{agent['agent_num']}: {role_label} | {agent['model']} | {agent['color']}")
        
        # Print classifier config
        classifiers_enabled = [k.upper() for k, v in enabled_classifiers.items() if v]
        if classifiers_enabled:
            print(f"\nML Classifiers Enabled: {', '.join(classifiers_enabled)}")
        else:
            print(f"\nML Classifiers: DISABLED")
        print(f"{'='*60}\n")
        
        # store in session
        session['game_id'] = game_id
        session['composition'] = f"custom_{game_id}"
        session['num_agents'] = num_agents
        session['num_rounds'] = num_rounds
        
        # reset live state so we don't show a previous game's snapshot
        try:
            if os.path.exists(LIVE_STATE_FILE):
                os.remove(LIVE_STATE_FILE)
        except Exception as e:
            print(f"WARNING: Could not clear live_state.json: {e}")
        
        # Build command
        cmd = [
            sys.executable,
            os.path.join(BACKEND_PATH, 'main.py'),
            '--composition_name', f'custom_{game_id}',
            '--game_id', game_id,
            '--job_index', '0',
            '--num_rounds', str(num_rounds)
        ]
        
        print(f"{'='*60}")
        print(f"LAUNCHING GAME: {game_id}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Agents: {num_agents} ({byzantine_count} Byzantine, {honest_count} Honest)")
        print(f"Rounds: {num_rounds}")
        print(f"{'='*60}\n")
        
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
                process.stdout.close()
        
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
        # load existing game IDs to avoid duplicates (use tolerant reader)
        existing_data = read_stats_csv(MASTER_CSV)
        existing_game_ids = set(r.get("game_id") for r in existing_data if r.get("game_id"))
        
        # find all stats.csv files in logs
        pattern = os.path.join(BACKEND_PATH, 'logs', '*', 'Game_*_Run0', 'stats.csv')
        csv_files = glob.glob(pattern)
        
        print(f"\nScanning for new games...")
        print(f"Found {len(csv_files)} total stats.csv files")
        print(f"Already have {len(existing_game_ids)} games in database")
        
        new_games = 0
        new_data = []
        
        for csv_file in csv_files:
            # extract metadata from path: logs/tiny_test/Game_test_005_Run0/stats.csv
            parts = csv_file.split(os.sep)
            composition = parts[-3]
            game_folder = parts[-2]
            game_id = game_folder.replace('Game_', '').replace('_Run0', '')
            
            if game_id in existing_game_ids:
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # add metadata columns
                df.insert(0, 'composition', composition)
                df.insert(1, 'game_id', game_id)
                df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                new_data.append(df)
                new_games += 1
                print(f"  Added: {game_id} ({composition})")
                
            except Exception as e:
                print(f"ERROR reading {csv_file}: {e}")
        
        # append new data to master CSV
        if new_data:
            combined = pd.concat(new_data, ignore_index=True)
            
            if os.path.exists(MASTER_CSV):
                combined.to_csv(MASTER_CSV, mode='a', header=False, index=False)
            else:
                combined.to_csv(MASTER_CSV, mode='w', header=True, index=False)
            
            print(f"\nAdded {new_games} new games to database\n")
        else:
            print(f"\nNo new games found\n")
        
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
        
        # Find the discussion_chat.csv file for this game
        possible_paths = [
            os.path.join(BACKEND_PATH, 'logs', f'Game_{game_id}', 'discussion_chat.csv'),
            os.path.join(BACKEND_PATH, 'logs', '*', f'Game_{game_id}_Run0', 'discussion_chat.csv')
        ]
        
        discussion_file = None
        for pattern in possible_paths:
            matches = glob.glob(pattern)
            if matches:
                discussion_file = matches[0]
                break
        
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