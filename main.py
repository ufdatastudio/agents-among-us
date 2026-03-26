# main.py
# Entrypoint for running controller sessions: parses args, sets controller environment, and orchestrates game simulations and IPC.
import argparse
from datetime import datetime
import os
import platform

from config.app_mode import should_load_dotenv

if should_load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Set LLM_MODE default if not already set (e.g. by submit_globus.sh)
IS_MAC = platform.system() == "Darwin"
if IS_MAC:
    os.environ.setdefault("LLM_MODE", "LOCAL")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ.setdefault("LLM_MODE", "LOCAL")
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Enable this for distributed GPU runs

import time
from uuid import uuid4
from config.settings import NUM_ROUNDS as DEFAULT_NUM_ROUNDS, MAX_MOVEMENT_PHASES
from config.model_composition import COMPOSITION
from core.game_engine import GameEngine
from core.llm import ModelManager
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0, help="Slurm Array Task ID (0-99)")
    parser.add_argument("--composition_name", type=str, required=True, help="Name of the composition to run")
    parser.add_argument("--game_id", type=str, required=True, help="Shared Session ID for IPC")
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS, help="Number of rounds to play")
    parser.add_argument("--num_ticks", type=int, default=MAX_MOVEMENT_PHASES, help="Movement ticks per round")
    parser.add_argument("--num_discussion_messages", type=int, default=2, help="Messages per agent per discussion")
    args = parser.parse_args()

    num_rounds = args.num_rounds
    num_ticks = args.num_ticks
    num_discussion_messages = args.num_discussion_messages

    selected_composition = next((c for c in COMPOSITION if c["name"] == args.composition_name), None)

    # Check game_configs/ folders for custom JSONs (logs/ is writable in containers)
    if selected_composition is None:
        import json
        search_dirs = [
            os.path.join(os.path.dirname(__file__), 'logs', 'game_configs'),
            os.path.join(os.path.dirname(__file__), 'config', 'game_configs'),
        ]
        for search_dir in search_dirs:
            game_configs_file = os.path.join(search_dir, f'{args.composition_name}.json')
            if os.path.exists(game_configs_file):
                with open(game_configs_file, 'r') as f:
                    selected_composition = json.load(f)
                    print(f"Loaded custom composition from: {game_configs_file}", flush=True)
                break

    # Fallback: check config/ root folder
    if selected_composition is None:
        import json
        config_file = os.path.join(os.path.dirname(__file__), 'config', f'{args.composition_name}.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                selected_composition = json.load(f)
                print(f"Loaded custom composition from: {config_file}", flush=True)

    if selected_composition is None:
        raise ValueError(f"Composition '{args.composition_name}' not found in COMPOSITION list or config directory.")

    if isinstance(selected_composition, dict) and 'num_discussion_messages' in selected_composition:
        num_discussion_messages = selected_composition['num_discussion_messages']

    manager = ModelManager.get_instance()
    manager.set_game_context(args.game_id, args.composition_name)

    if manager.mode == "GLOBUS":
        manager.init_globus_executor()

    unique_run_id = f"{args.game_id}_Run{args.job_index}"

    engine = GameEngine(
        game_id=unique_run_id,
        num_agents=selected_composition['honest_count'] + selected_composition['byzantine_count'],
        num_rounds=num_rounds,
        num_ticks=num_ticks,
        num_discussion_messages=num_discussion_messages
    )
   
    engine.setup(composition=selected_composition)
    final_result = None

    for round_num in range(1, num_rounds + 1):
        # Run Movement (Sync)
        meeting_called = engine.run_movement_phase(round_num)
                
        # Did anyone die during movement?
        final_result = engine.check_win_condition()
        if final_result:
            break
            
        if meeting_called:
            engine.run_discussion_phase(round_num)
            
            final_result = engine.check_win_condition()
            if final_result:
                break

    if not final_result:
        final_result = "Honest Agents Win, Max Rounds Reached"
        engine.finalize_stats(final_result) 
    print(f"Game Over. Result: {final_result}")

if __name__ == "__main__":
    main()