#!/bin/bash
#SBATCH --job-name="repsurf"            # Название задачи
#SBATCH --account=proj_1549          # Идентификатор проекта
#SBATCH --time=00:05:00            # Максимальное время выполнения
#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=2          # Требуемое кол-во CPU

srun sh scripts/scanobjectnn/repsurf_ssg_umb.sh