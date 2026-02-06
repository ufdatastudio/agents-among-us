"""
Flask app for Agents Among Us Frontend
Serves the UI and provides API endpoints for game control
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import subprocess
import os
import json
import sys

app = Flask(__name__)
app.secret_key = 'agents-among-us-secret-key'

# Path to the backend (parent directory of frontend)
BACKEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Store running game process
current_game_process = None


@app.route('/')
def index():
    """Home page - main menu"""
    return render_template('index.html')


@app.route('/config')
def config():
    """Configuration page"""
    return render_template('config.html')


@app.route('/game')
def game():
    """Main game view"""
    game_id = session.get('game_id', 'unknown')
    return render_template('game.html', game_id=game_id)


@app.route('/stats')
def stats():
    """Statistics page"""
    return render_template('stats.html')


@app.route('/win')
def win():
    """Win screen"""
    winner = request.args.get('winner', 'Unknown')
    return render_template('win.html', winner=winner)


if __name__ == '__main__':
    print(" Open: http://localhost:3000")
    app.run(debug=True, port=3000)