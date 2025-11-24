# main.py
import time
import argparse
from uuid import uuid4
from core.simulation import setup_game
from config.settings import NUM_ROUNDS
from game.game_loopv3 import run_game_round, finalize_log

def main(model_name):
    """
    Sets up and runs the ByzantineBrains simulation from the command line.
    """
    start_time = time.time()
    game_id = str(uuid4())
    
    # Set up the game state and agents
    agents, agents_state, state = setup_game(game_id, model_name)
    
    round_num = 1
    while round_num <= NUM_ROUNDS:
        print(f"\n{'='*15} Round {round_num} {'='*15}")
        
        # Run a single round of the game
        run_game_round(game_id, round_num, state, agents, agents_state)
        
        # Check for win conditions to end the simulation early
        alive_byzantines = [
            a for a in agents 
            if agents_state[a.name]["role"] == "byzantine" and not state[a.name]["eliminated"]
        ]
        alive_honest = [
            a for a in agents 
            if agents_state[a.name]["role"] == "honest" and not state[a.name]["eliminated"]
        ]
        
        if not alive_byzantines or len(alive_byzantines) >= len(alive_honest):
            print("\n--- Win condition met. Ending simulation. ---")
            break
            
        round_num += 1
        time.sleep(1) # Pause between rounds for readability

    # --- Determine Final Victory Condition ---
    end_time = time.time()
    duration = end_time - start_time
    
    alive_byzantines = [
        a for a in agents 
        if agents_state[a.name]["role"] == "byzantine" and not state[a.name].get("eliminated")
    ]
    alive_honest = [
        a for a in agents 
        if agents_state[a.name]["role"] == "honest" and not state[a.name].get("eliminated")
    ]

    if not alive_byzantines:
        victory_condition = "Honest Agents Win"
    elif len(alive_byzantines) >= len(alive_honest):
        victory_condition = "Byzantine Agents Win"
    else:
        victory_condition = "Max Rounds Reached - Honest Agents Win"
        
    print(f"\n🏁 Simulation Complete: {victory_condition}")
    finalize_log(game_id, duration, victory_condition)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ByzantineBrains simulation.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct", 
        help="The model identifier to use for the agents (e.g., 'meta-llama/Llama-3.1-8B-Instruct')."
    )
    args = parser.parse_args()
    
    main(args.model)