#!/bin/bash
#SBATCH --job-name="mstoadmixture"
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:0
#SBATCH --output="adm_test/s5e5%a.log"
#SBATCH --error="adm_test/s5e5%a.err
#SBATCH --array="0-99"

source activate lamom2
idx=$SLURM_ARRAY_TASK_ID
srun python mstoadmixture.py ${idx}
