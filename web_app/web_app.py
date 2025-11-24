# Flask-based server for running and visualizing the ByzantineBrains simulation.
import sys
import os
import time
from uuid import uuid4
from flask import Flask, Response, render_template, stream_with_context, jsonify, request
from core.simulation import setup_game, get_current_state, _current_game
from config.settings import NUM_ROUNDS
from game.game_loop import run_game_round, finalize_log, generate_map_html, generate_agent_status_html

app = Flask(__name__)
active_game_id = None

# Captures stdout and buffers it for streaming to the client.
class RealTimeStream:
    def __init__(self):
        self.buffer = []

    def write(self, text):
        lines = text.splitlines()
        self.buffer.extend(lines)

    def flush(self):
        for line in self.get():
            if line.strip():
                yield f"data: {line.strip()}\n\n"

    def get(self):
        output = self.buffer[:]
        self.buffer.clear()
        return output

# Serves the main simulation UI with the initial map.
@app.route("/")
def index():
    initial_map = generate_map_html()
    return render_template("index.html", map_html=initial_map)

# Runs the full simulation, streaming each round's output to the UI.
@app.route("/run")
def run():
    selected_model = request.args.get("model", "All")
    @stream_with_context
    def generate():
        global active_game_id
        stream = RealTimeStream()
        original_stdout = sys.stdout
        sys.stdout = stream

        start_time = time.time() # START TIMER
        game_id = str(uuid4())
        active_game_id = game_id
        agents, agents_state, state = setup_game(game_id, selected_model)
        round_num = 1
        try:
            while round_num <= NUM_ROUNDS:
                print(f"--- Round {round_num} ---")
                yield from stream.flush()
                yield from run_game_round(game_id, round_num, state, agents, agents_state, stream)
                # End simulation early if win condition is met.
                alive_byzantines = [a for a in agents if agents_state[a.name]["role"] == "byzantine" and not state[a.name]["eliminated"]]
                alive_honest = [a for a in agents if agents_state[a.name]["role"] == "honest" and not state[a.name]["eliminated"]]
                if not alive_byzantines or len(alive_byzantines) >= len(alive_honest):
                    break
                round_num += 1
                time.sleep(0.25)

            end_time = time.time() # 2. Record the end time
            duration = end_time - start_time # 3. Calculate duration

            # VICTORY CONDITION - MAY BE REDUNDANT
            alive_byzantines = [a for a in agents if agents_state[a.name]["role"] == "byzantine" and not state[a.name].get("eliminated")]
            alive_honest = [a for a in agents if agents_state[a.name]["role"] == "honest" and not state[a.name].get("eliminated")]
            if not alive_byzantines:
                victory_condition = "Honest Agents Win"
            elif len(alive_byzantines) >= len(alive_honest):
                victory_condition = "Byzantine Agents Win"
            else:
                victory_condition = "Max Rounds Reached - Honest Agents Win"
            
            for agent in agents:
                if state[agent.name].get("eliminated"):
                    agents_state[agent.name]["survived"] = False

            finalize_log(game_id, duration, victory_condition)


            print(f"✔ Simulation complete for game_id: {game_id}")
            yield from stream.flush()
        finally:
            sys.stdout = original_stdout
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')

# Returns the current map and agent status for periodic frontend updates.
@app.route("/map")
def map_endpoint():
    agents, state = get_current_state()
    return jsonify(
        map_html=generate_map_html(state, agents),
        agent_status=generate_agent_status_html(state, agents)
    )

# Launch the Flask app with threading enabled for live updates.
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
