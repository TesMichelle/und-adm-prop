#!/bin/bash
#SBATCH --job-name="sample_accurancy"
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:0
#SBATCH --output="sample_accurancy/s5d5_%a.log"
#SBATCH --error="sample_accurancy/s5d5_%a.err"
#SBATCH --array="0-99"

source activate lamom2
idx=$SLURM_ARRAY_TASK_ID
srun python reg_est_sim.py 5 5 ${idx}
