#!/bin/bash
#SBATCH --job-name=amongus_sim        
#SBATCH --output=job_logs/slurm_%A_%a.log
#SBATCH --nodes=3-3                    
#SBATCH --ntasks=3
#SBATCH --gpus-per-node=1                      
#SBATCH --cpus-per-task=8                 
#SBATCH --mem=120gb                   
#SBATCH --time=1:10:00
#SBATCH --array=0-5
#SBATCH --partition=hpg-b200
JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SESSION_ID="Session_${JOB_ID}_${TASK_ID}_${COMP_NAME}"

IPC_DIR="logs/${COMP_NAME}/Game_${SESSION_ID}/ipc"
if [[ -n "$COMP_NAME" && -n "$SESSION_ID" ]]; then
    rm -rf "$IPC_DIR" 
fi

mkdir -p "$IPC_DIR"

module purge
module load conda
conda activate amongus
sleep 5
PYTHON_EXE=$(which python)

MODELS_PER_GPU=4
GAMES_PER_JOB=5
MODEL_LIST=$(python config/generate_batch_list.py "$COMP_NAME")
if [ $? -ne 0 ]; then
    echo "Error: Could not determine models for composition '$COMP_NAME'"
    exit 1
fi
MODEL_ARRAY=($MODEL_LIST) 
TOTAL_MODELS=${#MODEL_ARRAY[@]}

mapfile -t NODE_ARRAY < <(scontrol show hostnames $SLURM_JOB_NODELIST)

for (( i=0; i<TOTAL_MODELS; i+=MODELS_PER_GPU )); do

    NODE_IDX=$(( i / MODELS_PER_GPU ))    
    CURRENT_NODE=${NODE_ARRAY[$NODE_IDX]}

    CURRENT_MODELS=("${MODEL_ARRAY[@]:$i:$MODELS_PER_GPU}")
    
    IFS=',' 
    MODELS_TO_PASS="${CURRENT_MODELS[*]}"
    unset IFS 

    echo "Launching Worker on node: $CURRENT_NODE with models: $MODELS_TO_PASS"
    srun --ntasks=1 \
         --nodes=1 \
         -exclusive \
         --gpus-per-task=1 \
         --nodelist=$CURRENT_NODE \
         --cpu-bind=none \
         $PYTHON_EXE worker.py \
             --game_id "$SESSION_ID" \
             --model_names "$MODELS_TO_PASS" \
             --comp_name "$COMP_NAME" &
         
     sleep 10
    
done



MAX_RETRIES=120  
COUNT=0        

while [ $COUNT -lt $MAX_RETRIES ]; do
    # Check if any ready signal exists in the specific IPC dir
    NUM_READY=$(ls "$IPC_DIR"/ready_*.signal 2>/dev/null | wc -l)
    if [ "$NUM_READY" -eq "$TOTAL_MODELS" ]; then
        echo "Workers are ready! Starting game..."
        break
    fi
    echo "Workers still loading... ($COUNT/$MAX_RETRIES)"
    sleep 10
    COUNT=$((COUNT+1))
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo "Error: Workers timed out initializing."
    kill $(jobs -p)
    exit 1
fi

echo "Running Composition: $COMP_NAME"

START_INDEX=$(( TASK_ID * GAMES_PER_JOB ))

for (( j=0; j<GAMES_PER_JOB; j++ )); do
    CURRENT_GAME_IDX=$(( START_INDEX + j ))
    
    echo ">>> Starting Game $j (Global Index $CURRENT_GAME_IDX)"
    
    $PYTHON_EXE main.py \
        --composition_name "$COMP_NAME" \
        --job_index $CURRENT_GAME_IDX \
        --game_id "$SESSION_ID" 
        
    echo "Game $j finished."
    rm -f "$IPC_DIR"/*
done
rmdir "$IPC_DIR" 2>/dev/null
kill $(jobs -p) 2>/dev/null || true
wait
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,MaxRSS,Elapsed,State
sleep 10