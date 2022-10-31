#!/bin/bash
#SBATCH --job-name=sim_single           # Название задачи
#SBATCH --error=sim_single/%j.err        # Файл для вывода ошибок
#SBATCH --output=sim_single/%j.log       # Файл для вывода результатов
#SBATCH --time=5:00:00                      # Максимальное время выполнения
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу
#SBATCH --array=0-99                        # Массив подзадач с номерами от 1 до 20
#SBATCH --nodes=1


idx=$SLURM_ARRAY_TASK_ID
module load Python/Anaconda_v11.2021        # Загрузка модуля Anaconda
source activate lamom
srun python run_sim_single.py 1 $idx             # Выполнение расчёта
srun python run_sim_single.py 2 $idx             # Выполнение расчёта
srun python run_sim_single.py 3 $idx             # Выполнение расчёта
srun python run_sim_single.py 4 $idx             # Выполнение расчёта
srun python run_sim_single.py 5 $idx             # Выполнение расчёта
srun python run_sim_single.py 6 $idx             # Выполнение расчёта
srun python run_sim_single.py 7 $idx             # Выполнение расчёта
srun python run_sim_single.py 8 $idx             # Выполнение расчёта
srun python run_sim_single.py 9 $idx             # Выполнение расчёта
srun python run_sim_single.py 10 $idx             # Выполнение расчёта
