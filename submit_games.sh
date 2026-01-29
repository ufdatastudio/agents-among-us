#!/bin/bash
#SBATCH --job-name=amongus_sim        
#SBATCH --output=job_logs/slurm_%A_%a.log
#SBATCH --nodes=5-5                    
#SBATCH --ntasks=5
#SBATCH --gpus-per-node=1                      
#SBATCH --cpus-per-task=8                 
#SBATCH --mem=128gb                   
#SBATCH --time=5:00:00               
#SBATCH --partition=hpg-b200
#SBATCH --array=0-4

JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SESSION_ID="Session_${JOB_ID}_${TASK_ID}_${COMP_NAME}"

IPC_DIR="logs/${COMP_NAME}/Game_${SESSION_ID}/ipc"
mkdir -p "$IPC_DIR"


module purge
module load conda
conda activate amongus
#export HF_HUB_ENABLE_HF_TRANSFER=1



MODELS_PER_GPU=2
MODEL_LIST=$(python config/generate_batch_list.py)
MODEL_ARRAY=($MODEL_LIST) # Convert string to array
TOTAL_MODELS=${#MODEL_ARRAY[@]}

# Loop with step of MODELS_PER_GPU
for (( i=0; i<TOTAL_MODELS; i+=MODELS_PER_GPU )); do
    
    # Slice the array starting at index 'i' for 'MODELS_PER_GPU' elements
    CURRENT_MODELS=("${MODEL_ARRAY[@]:$i:$MODELS_PER_GPU}")

    # Join the array elements with a comma
    IFS=',' 
    MODELS_TO_PASS="${CURRENT_MODELS[*]}"
    unset IFS 

    echo "Launching worker with models: $MODELS_TO_PASS"

    # Pass the comma-separated list to --model_names
    srun --ntasks=1 \
         --nodes=1 \
         --exclusive \
         --gpus-per-task=1 \
         --cpu-bind=none \
         python worker.py \
             --game_id "$SESSION_ID" \
             --model_names "$MODELS_TO_PASS" \
             --comp_name "$COMP_NAME" &
         
    # Small buffer to prevent race conditions during srun allocation
    sleep 10
done

sleep 60
echo "Running Composition: $COMP_NAME"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

GAMES_PER_JOB=20
START_INDEX=$(( TASK_ID * GAMES_PER_JOB ))

for (( j=0; j<GAMES_PER_JOB; j++ )); do
    CURRENT_GAME_IDX=$(( START_INDEX + j ))
    GAME_ID="${SESSION_ID}_Run${j}"
    
    echo ">>> Starting Game $j (Global Index $CURRENT_GAME_IDX)"
    
    python main.py \
        --composition_name "$COMP_NAME" \
        --job_index $CURRENT_GAME_IDX \
        --game_id "$SESSION_ID" 
        
    echo "Game $j finished."
    rm -f "$IPC_DIR"/*
done

rmdir "$IPC_DIR" 2>/dev/null