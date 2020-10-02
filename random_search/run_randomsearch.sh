#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH -p small
#SBATCH --mem-per-cpu=16000
#SBATCH -n 5
#SBATCH -e randsearch.err
#SBATCH -o randsearch.out

python random_search_mbtr.py
