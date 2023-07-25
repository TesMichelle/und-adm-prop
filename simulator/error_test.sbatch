#!/bin/bash
#SBATCH --job-name="mstoadmixture"
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:0
#SBATCH --output="adm_test/s5e5%a.log"
#SBATCH --error="adm_test/s5e5%a.err"
#SBATCH --array="0-2"

source activate lamom2
idx=$SLURM_ARRAY_TASK_ID
srun python errortest.py ${idx}

find error_test/seed_${idx} -name '*.gz' \
| xargs -tI{} plink --vcf {} --recode 12 --make-bed --out {} --const-fid

find error_test/seed_${idx} -name '*.bed' \
plink --bfile {} --indep-pairwise 50 10 0.1

find error_test/seed_${idx} -name '*.bed' \
plink --bfile {} --extract plink.prune.in --make-bed --out prunedData

find error_test/seed_${idx} -name 'prunedData.bed' \
| xargs -tI{} admixture {} --supervised 2 -j4
