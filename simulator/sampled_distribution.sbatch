#! /bin/bash
#SBATCH --job-name="distribution_sampling"
#SBATCH --cpus-per-task=1
#SBATCH --time=0-12:0
#SBATCH --output="sd_2_3/allchr_%a.log"
#SBATCH --error="sd_2_3/allchr_%a.err"
#SBATCH --array="0-999"

source activate lamom2
idx=$SLURM_ARRAY_TASK_ID
ts=1
dur=2
srun python sampled_distribution.py ${idx} 2.2336 $ts $dur
srun python sampled_distribution.py ${idx} 2.1466 $ts $dur
srun python sampled_distribution.py ${idx} 2.0409 $ts $dur
srun python sampled_distribution.py ${idx} 1.9202 $ts $dur
srun python sampled_distribution.py ${idx} 1.8722 $ts $dur
srun python sampled_distribution.py ${idx} 1.6800 $ts $dur
srun python sampled_distribution.py ${idx} 1.6636 $ts $dur
srun python sampled_distribution.py ${idx} 1.8114 $ts $dur
srun python sampled_distribution.py ${idx} 1.5822 $ts $dur
srun python sampled_distribution.py ${idx} 1.7468 $ts $dur
srun python sampled_distribution.py ${idx} 1.2571 $ts $dur
srun python sampled_distribution.py ${idx} 1.2020 $ts $dur
srun python sampled_distribution.py ${idx} 1.4173 $ts $dur
srun python sampled_distribution.py ${idx} 1.3404 $ts $dur
srun python sampled_distribution.py ${idx} 1.2849 $ts $dur
srun python sampled_distribution.py ${idx} 1.1771 $ts $dur
srun python sampled_distribution.py ${idx} 1.0773 $ts $dur
srun python sampled_distribution.py ${idx} 1.0827 $ts $dur
srun python sampled_distribution.py ${idx} 0.6279 $ts $dur
srun python sampled_distribution.py ${idx} 0.7411 $ts $dur
