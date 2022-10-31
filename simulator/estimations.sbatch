#!/bin/bash
#SBATCH --job-name=props_ests            # Название задачи
#SBATCH --error=mom_ests/%j.err        # Файл для вывода ошибок
#SBATCH --output=mom_ests%j.log       # Файл для вывода результатов
#SBATCH --time=1:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --array=0-63                        # Массив подзадач с номерами от 1 до 20
#SBATCH --nodes=1


idx=$SLURM_ARRAY_TASK_ID
module load Python/Anaconda_v11.2021        # Загрузка модуля Anaconda
source activate lamom
srun python run_est.py $idx             # Выполнение расчёта
