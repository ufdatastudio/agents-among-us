# worker.py
import os
os.environ["LLM_MODE"] = "WORKER"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import time
import json
import glob
import argparse
from core.llm import ModelManager

def run_worker(game_id, model_names_str, comp_name):
    # Set mode to LOCAL so it actually loads models
    os.environ["LLM_MODE"] = "LOCAL" 
    model_list = [m.strip() for m in model_names_str.split(',') if m.strip()]
    print(f"--- Starting Worker for Game {game_id} ---")

    manager = ModelManager.get_instance()
    
    for model_name in model_list:
        manager.load_model(model_name)
    
    ipc_path = os.path.join("logs", comp_name, f"Game_{game_id}", "ipc")
    # try-except for makedirs to prevent race conditions on startup
    try:
        os.makedirs(ipc_path, exist_ok=True)
    except OSError:
        pass
    
    print(f"Worker Ready. Watching: {ipc_path}")

    while True:
        # Find all .json files that are NOT responses
        files = glob.glob(os.path.join(ipc_path, "*.json"))
        requests = [f for f in files if not f.endswith("_response.json") and not f.endswith(".lock")]
        
        for req_file in requests:
            lock_file = req_file + ".lock"
            
            # 1. Attempt to Lock file (Atomic rename)
            try:
                os.rename(req_file, lock_file)
            except OSError:
                # File already taken by another worker
                continue

            try:
                # 2. Read Request
                with open(lock_file, "r") as f:
                    data = json.load(f)
                
                # 3. Check if this worker handles this model
                if data["model_name"] not in model_list:
                    # Not my job, unlock it
                    try:
                        os.rename(lock_file, req_file)
                    except OSError:
                        # If we can't rename it back, it might be gone. 
                        # Just ignore it to prevent crashing.
                        pass
                    continue
                
                print(f"Processing request: {data['id']}")
                
                # 4. Generate
                response_text = manager.generate(
                    data["model_name"],
                    data["system_prompt"],
                    data["user_prompt"],
                    data["temperature"]
                )
                
                # 5. Write Response
                response_path = os.path.join(ipc_path, f"{data['id']}_response.json")
                with open(response_path, "w") as f:
                    json.dump({"id": data["id"], "response": response_text}, f)
                
                # 6. Remove Lock/Request file
                # leave the response for the Controller to pick up
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
                
            except Exception as e:
                print(f"Error processing loop: {e}")
                # Attempt to restore file if something crashed mid-read
                if os.path.exists(lock_file):
                    try:
                        os.rename(lock_file, req_file)
                    except OSError:
                        pass
                
        time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=str, required=True)
    parser.add_argument("--model_names", type=str, required=True)
    parser.add_argument("--comp_name", type=str, required=True)
    args = parser.parse_args()
    
    run_worker(args.game_id, args.model_names, args.comp_name)