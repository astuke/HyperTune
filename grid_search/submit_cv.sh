#!/bin/bash
#SBATCH --job-name submit_cv
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -e mpi_%A.err
#SBATCH -o mpi_%A.out

python cv_1.py &
python cv_2.py &
python cv_3.py &
python cv_4.py &
python cv_5.py
