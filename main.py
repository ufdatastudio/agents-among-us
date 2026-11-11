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
    #os.environ.setdefault("LLM_MODE", "LOCAL")
    os.environ.setdefault("LLM_MODE", "CONTROLLER") # FOR DISTRIBUTED GPU RUNS
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # FOR DISTRIBUTED GPU RUNS, COMMENT OUT

import time
from uuid import uuid4
from config.settings import NUM_ROUNDS as DEFAULT_NUM_ROUNDS, MAX_MOVEMENT_PHASES
from config.model_composition import COMPOSITION
from core.game_engine import GameEngine
from core.llm import ModelManager
from core.situation_room import RoomConfig
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0, help="Slurm Array Task ID (0-99)")
    parser.add_argument("--composition_name", type=str, required=True, help="Name of the composition to run")
    parser.add_argument("--game_id", type=str, required=True, help="Shared Session ID for IPC")
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS, help="Number of rounds to play")
    parser.add_argument("--num_ticks", type=int, default=MAX_MOVEMENT_PHASES, help="Movement ticks per round")
    parser.add_argument("--num_discussion_messages", type=int, default=2, help="Messages per agent per discussion")
    parser.add_argument("--skip_discussion", action="store_true", help="Skip discussion phase and vote purely on action logs")
    parser.add_argument("--discussion_mode", choices=("turns", "timed"), default="turns", help="turns keeps turn-taking; timed runs the situation room")
    parser.add_argument("--room_budget_s", type=float, default=180.0, help="Length of a timed discussion in room seconds")
    parser.add_argument("--room_tick_s", type=float, default=10.0, help="Room seconds between decision points")
    parser.add_argument("--room_max_words", type=int, default=40, help="Words kept from one statement in the room")
    parser.add_argument("--room_overlap", choices=("clean", "interleaved", "noisy"), default="clean", help="How overlapping speech is shown to the listener")
    parser.add_argument("--room_seed", type=int, default=0, help="Seed for onsets, shuffles and dropout in the room")
    parser.add_argument("--seed", type=int, default=None, help="Scenario seed. Games with one seed start every agent in the same room")
    parser.add_argument("--room_dropout", type=float, default=0.0, help="Fraction of words a listener misses in noisy mode")
    parser.add_argument("--room_tag_policy", choices=("every_chunk", "first_chunk"), default="every_chunk", help="Speaker tags in noisy mode")
    parser.add_argument("--room_workers", type=int, default=None, help="Agents that generate at once in the room (default: all, or 1 when the model runs in this process)")
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

    skip_discussion = args.skip_discussion
    if isinstance(selected_composition, dict) and 'skip_discussion' in selected_composition:
        skip_discussion = selected_composition['skip_discussion']

    if isinstance(selected_composition, dict) and 'num_discussion_messages' in selected_composition:
        num_discussion_messages = selected_composition['num_discussion_messages']

    discussion_mode = args.discussion_mode
    if isinstance(selected_composition, dict) and 'discussion_mode' in selected_composition:
        discussion_mode = selected_composition['discussion_mode']

    manager = ModelManager.get_instance()
    manager.set_game_context(args.game_id, args.composition_name)

    # A model loaded in this process cannot serve several agents at once.
    room_workers = args.room_workers if args.room_workers is not None else (1 if manager.mode == "LOCAL" else None)
    room_config = RoomConfig(
        budget_s=args.room_budget_s,
        tick_s=args.room_tick_s,
        max_words=args.room_max_words,
        overlap=args.room_overlap,
        dropout=args.room_dropout,
        tag_policy=args.room_tag_policy,
        seed=args.room_seed,
    )

    if manager.mode == "GLOBUS":
        manager.init_globus_executor()

    unique_run_id = f"{args.game_id}_Run{args.job_index}"

    engine = GameEngine(
        game_id=unique_run_id,
        num_agents=selected_composition['honest_count'] + selected_composition['byzantine_count'],
        num_rounds=num_rounds,
        num_ticks=num_ticks,
        num_discussion_messages=num_discussion_messages,
        skip_discussion=skip_discussion,
        discussion_mode=discussion_mode,
        room_config=room_config,
        room_workers=room_workers,
        seed=args.seed,
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