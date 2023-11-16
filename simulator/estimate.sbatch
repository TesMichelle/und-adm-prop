#!/bin/bash
#SBATCH --job-name="estimate_accurancy"
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:0
#SBATCH --output="estimate_accurancy/s5d5_%a.log"
#SBATCH --error="estimate_accurancy/s5d5_%a.err"
#SBATCH --array="0-99"

source activate lamom2
idx=$SLURM_ARRAY_TASK_ID
srun python estimate.py sample_accurancy/s5d5_${idx} ${idx}
