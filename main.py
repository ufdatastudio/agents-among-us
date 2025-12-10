# main.py
import argparse
import time
from uuid import uuid4
from config.settings import NUM_ROUNDS, AGENT_LLM_CONFIG
from game.game_engine import GameEngine
from core.llm import ModelManager
import random

def main():
    # Generate random number between 1000-9999 for game ID
    game_id = random.randint(1000, 9999)
    
    # Init Engine (LogManager created inside)
    engine = GameEngine(game_id)
    
    # Preload Model
    manager = ModelManager.get_instance()
    for model in AGENT_LLM_CONFIG:
        manager.load_model(model)
    
    engine.setup(model_list=AGENT_LLM_CONFIG)
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
            
            # Check 2: Did anyone get ejected?
            # (Only need to check this if a discussion actually happened)
            final_result = engine.check_win_condition()
            if final_result:
                break
        time.sleep(1)

    if not final_result:
        final_result = "Honest Agents Win, Max Rounds Reached" 
    print(f"Game Over. Result: {final_result}")

if __name__ == "__main__":
    main()