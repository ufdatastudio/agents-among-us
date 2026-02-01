# main.py
# Entrypoint for running controller sessions: parses args, sets controller environment, and orchestrates game simulations and IPC.
import argparse
from datetime import datetime
import os
import platform

# Auto-detect: LOCAL on Mac, CONTROLLER on HiperGator

IS_MAC = platform.system() == "Darwin"
if IS_MAC:
    os.environ["LLM_MODE"] = "LOCAL"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Empty on Mac, not -1
else:
    os.environ["LLM_MODE"] = "CONTROLLER"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from uuid import uuid4
from config.settings import NUM_ROUNDS
from config.model_composition import COMPOSITION
from game.game_engine import GameEngine
from core.llm import ModelManager
import random

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0, help="Slurm Array Task ID (0-99)")
    parser.add_argument("--composition_name", type=str, required=True, help="Name of the composition to run")
    parser.add_argument("--game_id", type=str, required=True, help="Shared Session ID for IPC")
    args = parser.parse_args()

    selected_composition = next((c for c in COMPOSITION if c["name"] == args.composition_name), None)

    manager = ModelManager.get_instance()
    manager.set_game_context(args.game_id, args.composition_name)
    unique_run_id = f"{args.game_id}_Run{args.job_index}"


    engine = GameEngine(
        game_id=unique_run_id, 
        num_agents=selected_composition['honest_count'] + selected_composition['byzantine_count']
    )

    # Preload Model
    #manager = ModelManager.get_instance()
    # all_models_list = selected_composition['honest_model'] + selected_composition['byzantine_model']
    # unique_models = set(all_models_list)
    # for model in unique_models:
    #     manager.load_model(model)
    
    engine.setup(composition=selected_composition)
    final_result = None

    for round_num in range(1, NUM_ROUNDS + 1):
        # Run Movement (Sync)
        meeting_called = engine.run_movement_phase(round_num)
                
        # Check 1: Did anyone die during movement?
        final_result = engine.check_win_condition()
        if final_result:
            break
            
        if meeting_called:
            # Run Discussion
            engine.run_discussion_phase(round_num)
            
            final_result = engine.check_win_condition()
            if final_result:
                break

    if not final_result:
        final_result = "Honest Agents Win, Max Rounds Reached"
        engine.finalize_stats(final_result) 
    print(f"Game Over. Result: {final_result}")

    #manager.unload_all_models()

if __name__ == "__main__":
    main()