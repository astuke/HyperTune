#!/bin/bash
#SBATCH --time=14-00:00:00
#SBATCH -p longrun
#SBATCH --mem-per-cpu=32000
#SBATCH -n 5
#SBATCH -e gridsearch.err
#SBATCH -o gridsearch.out

python gridsearch_mbtr.py
