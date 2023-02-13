#!/bin/bash
#SBATCH --job-name=run_two
#SBATCH --error=run_two/%j.err
#SBATCH --output=run_two/%j.log
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-99
#SBATCH --nodes=1


idx=$SLURM_ARRAY_TASK_ID
source activate lamom2
srun python run_two.py $idx
