#!/bin/bash
#SBATCH --job-name=amongus_sim        # Name of the job as it appears in the queue
#SBATCH --output=job_logs/slurm_%A_%a.log # Combined output AND error log
#SBATCH --nodes=1                     # Run on 1 node
#SBATCH --ntasks=1                    # Run 1 task
#SBATCH --cpus-per-task=4             # 4 CPU cores per game instance
#SBATCH --mem=64gb                    # 64GB RAM per game instance
#SBATCH --time=02:00:00               # 2 hour time limit per game
#SBATCH --partition=hpg-b200          # Partition for GPU jobs 
#SBATCH --gpus=1                      # Request 1 GPU per game
#SBATCH --array=0-99%25              # Run 100 jobs (0-99), but only 25 at a time

# Print debug info to the log
pwd; hostname; date
echo "Running Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
# 1. Load Modules
module purge
module load conda

# 2. Activate Environment
conda activate amongus

# 4. Run the Game
# pass the SLURM_ARRAY_TASK_ID as the 'job_index'.
# main.py will use this to select the LLM composition and name the log files.
python main.py --job_index $SLURM_ARRAY_TASK_ID

echo "Job $SLURM_ARRAY_TASK_ID finished at $(date)"