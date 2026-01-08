# main.py
import argparse
from datetime import datetime
import time
from uuid import uuid4
from config.settings import NUM_ROUNDS
from config.model_composition import COMPOSITION
from game.game_engine import GameEngine
from core.llm import ModelManager
import random

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0, help="Slurm Array Task ID")
    args = parser.parse_args()

    scenario_idx = 4
    selected_composition = COMPOSITION[scenario_idx]
    game_id = f"{selected_composition['name']}_Job{args.job_index}_{datetime.now().strftime('%m%d_%H%M')}"
    engine = GameEngine(
        game_id=game_id, 
        num_agents=selected_composition['honest_count'] + selected_composition['byzantine_count']
    )
    
    # Preload Model
    manager = ModelManager.get_instance()
    unique_models = list(set([selected_composition['honest_model'], selected_composition['byzantine_model']]))
    for model in unique_models:
        manager.load_model(model)
    
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
        time.sleep(1)

    if not final_result:
        final_result = "Honest Agents Win, Max Rounds Reached"
        engine.finalize_stats(final_result) 
    print(f"Game Over. Result: {final_result}")

if __name__ == "__main__":
    main()