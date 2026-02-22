#!/bin/bash

compositions=(
    "Llama3_Apertus"
)
for comp in "${compositions[@]}"; do
    echo "Submitting batch for: $comp"
    sbatch --export=ALL,COMP_NAME="$comp" --job-name="$comp" submit_games.sh
    sleep 1 
done