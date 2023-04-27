#! /bin/bash
#SBATCH --job-name="distribution_sampling"
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:0
#SBATCH --output="sampled_distribution/chr1_test_%a.log"
#SBATCH --error="sampled_distribution/chr1_test_%a.err
#SBATCH --array="0-99"

source activate lamom2
idx=$SLURM_ARRAY_TASK_ID
srun python sampled_distribution.py ${idx} 1
