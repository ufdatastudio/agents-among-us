import argparse
import sys
from model_composition import COMPOSITION

def get_models_for_composition(comp_name):
    unique_models = set()    
    found = False
    for entry in COMPOSITION:
        if entry["name"] == comp_name:
            unique_models.update(entry["honest_model"])
            unique_models.update(entry["byzantine_model"])
            found = True
            break
            
    if not found:
        sys.stderr.write(f"Warning: Composition '{comp_name}' not found in COMPOSITION list. Check flags in model_composition.py.\n")
        return []

    return list(unique_models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("comp_name", type=str, help="The composition name (e.g. Llama3.3 or Llama3_Qwen2)")
    args = parser.parse_args()

    models = get_models_for_composition(args.comp_name)
    
    if models:
        print(" ".join(models))
    else:
        sys.exit(1)