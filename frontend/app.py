from flask import Flask, render_template, jsonify, request, redirect, url_for, session, send_file
import subprocess
import os
import json
import sys
import glob
import pandas as pd
from datetime import datetime
import signal

app = Flask(__name__)
app.secret_key = 'agents-among-us-secret-key-change-in-production'

# paths
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MASTER_CSV = os.path.join(DATA_DIR, 'frontend_stats.csv')
LIVE_STATE_FILE = os.path.join(BACKEND_PATH, 'live_state.json')

current_game_process = None

os.makedirs(DATA_DIR, exist_ok=True)


# page routes

@app.route('/') 
def index():
    return render_template('index.html') #home page (main menu)


@app.route('/config')
def config():
    return render_template('config.html') # settings (config) page


@app.route('/game')
def game():
    game_id = session.get('game_id', 'unknown') 
    return render_template('game.html', game_id=game_id) # main game view


@app.route('/stats')
def stats():
    return render_template('stats.html') # stats page


@app.route('/win')
def win():
    winner = request.args.get('winner', 'Unknown')
    return render_template('win.html', winner=winner) # win screen (not implemented yet)


# game control api routes

@app.route('/start_game', methods=['POST'])
# launch the backend simulation with custom configuration. Receives form data from config page, creates custom composition, starts game.
def start_game():
    global current_game_process
    
    try:
        # get form data
        num_agents = int(request.form.get('num_agents', 4))
        num_rounds = int(request.form.get('num_rounds', 10))
        game_id = request.form.get('game_id', '').strip()
        
        # auto-generate game_id if empty
        if not game_id:
            game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # parse agent configurations
        agents = []
        byzantine_count = 0
        honest_count = 0
        
        for i in range(num_agents):
            agent = {
                'model': request.form.get(f'agent_{i}_model'),
                'role': request.form.get(f'agent_{i}_role'),
                'color': request.form.get(f'agent_{i}_color')
            }
            agents.append(agent)
            
            if agent['role'] == 'byzantine':
                byzantine_count += 1
            else:
                honest_count += 1
        
        # create custom composition JSON
        composition = {
            "name": f"custom_{game_id}",
            "honest_count": honest_count,
            "byzantine_count": byzantine_count,
            "agents": agents,
            "num_rounds": num_rounds
        }
        
        # save composition to temporary file
        composition_file = os.path.join(BACKEND_PATH, 'config', f'custom_{game_id}.json')
        with open(composition_file, 'w') as f:
            json.dump(composition, f, indent=2)
        
        # store in session
        session['game_id'] = game_id
        session['composition'] = f"custom_{game_id}"
        session['num_agents'] = num_agents
        session['num_rounds'] = num_rounds
        
        # build command to run backend
        cmd = [
            sys.executable,  # Use same Python interpreter
            os.path.join(BACKEND_PATH, 'main.py'),
            '--composition_name', f'custom_{game_id}',
            '--game_id', game_id,
            '--job_index', '0'
        ]
        
        print(f"\n{'='*60}")
        print(f"🚀 LAUNCHING GAME: {game_id}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Agents: {num_agents} ({byzantine_count} Byzantine, {honest_count} Honest)")
        print(f"Rounds: {num_rounds}")
        print(f"{'='*60}\n")
        
        # start game process in background
        current_game_process = subprocess.Popen(
            cmd,
            cwd=BACKEND_PATH,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # redirect to game page
        return redirect(url_for('game'))
        
    except Exception as e:
        print(f"❌ ERROR starting game: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/stop_game', methods=['POST'])
# stop the currently running game
def stop_game():
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
# read live_state.json and return current game state. This is polled by the game page every 2 seconds.
def get_game_state():
    try:
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
#check if game is still running
def get_game_status():
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


# =============================================================================
# STATISTICS API ROUTES
# =============================================================================

@app.route('/api/stats/all')
def get_all_stats():
    """
    Return all data from frontend_stats.csv.
    Used by stats page to display tables.
    """
    try:
        if not os.path.exists(MASTER_CSV):
            return jsonify([])
        
        # Read CSV
        df = pd.read_csv(MASTER_CSV)
        
        # Convert to list of dicts
        data = df.to_dict('records')
        
        return jsonify(data)
        
    except Exception as e:
        print(f"❌ ERROR reading stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats/refresh', methods=['POST'])
def refresh_stats():
    """
    Scan logs/ folder for new games and append to frontend_stats.csv.
    Returns count of new games added.
    """
    try:
        # Load existing game IDs to avoid duplicates
        existing_game_ids = set()
        if os.path.exists(MASTER_CSV):
            existing_df = pd.read_csv(MASTER_CSV)
            existing_game_ids = set(existing_df['game_id'].unique())
        
        # Find all stats.csv files in logs
        pattern = os.path.join(BACKEND_PATH, 'logs', '*', 'Game_*_Run0', 'stats.csv')
        csv_files = glob.glob(pattern)
        
        print(f"\n🔍 Scanning for new games...")
        print(f"Found {len(csv_files)} total stats.csv files")
        print(f"Already have {len(existing_game_ids)} games in database")
        
        new_games = 0
        new_data = []
        
        for csv_file in csv_files:
            # Extract metadata from path
            # logs/tiny_test/Game_test_005_Run0/stats.csv
            parts = csv_file.split(os.sep)
            composition = parts[-3]
            game_folder = parts[-2]
            game_id = game_folder.replace('Game_', '').replace('_Run0', '')
            
            # Skip if already in database
            if game_id in existing_game_ids:
                continue
            
            try:
                # Read the game CSV
                df = pd.read_csv(csv_file)
                
                # Add metadata columns
                df.insert(0, 'composition', composition)
                df.insert(1, 'game_id', game_id)
                df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                new_data.append(df)
                new_games += 1
                print(f"  ✅ Added: {game_id} ({composition})")
                
            except Exception as e:
                print(f"  ❌ Error reading {csv_file}: {e}")
        
        # Append new data to master CSV
        if new_data:
            combined = pd.concat(new_data, ignore_index=True)
            
            if os.path.exists(MASTER_CSV):
                # Append without header
                combined.to_csv(MASTER_CSV, mode='a', header=False, index=False)
            else:
                # Create with header
                combined.to_csv(MASTER_CSV, mode='w', header=True, index=False)
            
            print(f"\n✅ Added {new_games} new games to database\n")
        else:
            print(f"\n📊 No new games found\n")
        
        return jsonify({'new_games': new_games})
        
    except Exception as e:
        print(f"❌ ERROR refreshing stats: {e}")
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


@app.route('/api/stats/clear', methods=['POST'])
def clear_stats():
    """Delete frontend_stats.csv (with confirmation)"""
    try:
        if os.path.exists(MASTER_CSV):
            os.remove(MASTER_CSV)
            print("🗑️  Deleted frontend_stats.csv")
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


# =============================================================================
# UTILITY ROUTES
# =============================================================================

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


# error handling

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# main

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎮 AGENTS AMONG US - WEB INTERFACE")
    print("="*60)
    print(f"📂 Backend Path: {BACKEND_PATH}")
    print(f"📂 Data Directory: {DATA_DIR}")
    print(f"📊 Stats Database: {MASTER_CSV}")
    print(f"🎯 Live State File: {LIVE_STATE_FILE}")
    print("="*60)
    print(f"🌐 Open: http://localhost:3000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=3000, use_reloader=True)