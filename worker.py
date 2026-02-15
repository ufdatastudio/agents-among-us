# worker.py
import os
import traceback
os.environ["LLM_MODE"] = "WORKER"
import sys
import time
import json
import glob
import argparse
from core.llm import ModelManager
import gc       
import torch    

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def run_worker(game_id, model_names_str, comp_name):
    # Set mode to LOCAL so it actually loads models
    os.environ["LLM_MODE"] = "LOCAL" 
    model_list = [m.strip() for m in model_names_str.split(',') if m.strip()]
    print(f"--- Starting Worker for Game {game_id} ---")

    manager = ModelManager.get_instance()
    
    for model_name in model_list:
        manager.load_model(model_name)
  
    ipc_path = os.path.join("logs", comp_name, f"Game_{game_id}", "ipc")
    #  prevent race conditions on startup
    try:
        os.makedirs(ipc_path, exist_ok=True)
    except OSError:
        pass
    
    for model_name in model_list:
        sanitized_name = model_name.replace("/", "_").replace("-", "_")
        ready_file = os.path.join(ipc_path, f"ready_{sanitized_name}.signal")
        with open(ready_file, 'w') as f:
            pass

    while True:
        files = glob.glob(os.path.join(ipc_path, "*.json"))
        relevant_files = []
        for f in files:
            if f.endswith("_response.json") or f.endswith(".lock"):
                continue
        
            if any(m.replace("/", "_").replace("-", "_") in f for m in model_list):
                relevant_files.append(f)

        for req_file in relevant_files:
            lock_file = req_file + ".lock"
            
            # 1. Attempt to Lock file (Atomic rename)
            try:
                os.rename(req_file, lock_file)
            except OSError:
                # File already taken by another worker
                continue

            try:
                with open(lock_file, "r") as f:
                    data = json.load(f)
                
                if data["model_name"] not in model_list:
                    # Not my job, unlock it
                    try:
                        os.rename(lock_file, req_file)
                    except OSError:
                        #  ignore it to prevent crashing.
                        pass
                    continue
                                
                response_text = manager.generate(
                    data["model_name"],
                    data["system_prompt"],
                    data["user_prompt"],
                    data["temperature"]
                )
                
                response_path = os.path.join(ipc_path, f"{data['id']}_response.json")
                temp_path = response_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump({"id": data["id"], "response": response_text}, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.rename(temp_path, response_path)
                
                try:
                    os.remove(lock_file)
                except OSError:
                    pass
                flush_memory()
                
            except Exception as e:
                print(f"Error processing loop: {e}")
                if os.path.exists(lock_file):
                    try:
                        os.rename(lock_file, req_file)
                    except OSError:
                        pass
                flush_memory()
                
        time.sleep(0.1)

if __name__ == "__main__":

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--game_id", type=str, required=True)
        parser.add_argument("--model_names", type=str, required=True)
        parser.add_argument("--comp_name", type=str, required=True)
        args = parser.parse_args()
        run_worker(args.game_id, args.model_names, args.comp_name)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()     
        sys.stderr.flush()