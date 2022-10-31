#!/bin/bash
#SBATCH --job-name=props_sims            # Название задачи
#SBATCH --error=props_sims-%j.err        # Файл для вывода ошибок
#SBATCH --output=props_sims-%j.log       # Файл для вывода результатов
#SBATCH --time=1:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --array=1-5                        # Массив подзадач с номерами от 1 до 20
#SBATCH --nodes=10


idx=$SLURM_ARRAY_TASK_ID
module load Python/Anaconda_v11.2021        # Загрузка модуля Anaconda
source activate props
srun python runsims.py $idx             # Выполнение расчёта
