#!/bin/bash
#SBATCH --ntasks-per-node=5
#SBATCH --time=14-00:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH -p longrun
#SBATCH -n 1

boss op boss.in
