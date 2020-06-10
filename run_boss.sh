#!/bin/bash
#SBATCH --job-name run_H_sobol_50
#SBATCH --ntasks-per-node=5
#SBATCH --time=14-00:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH -p longrun
#SBATCH -n 1
#SBATCH -e mpi_%A.err
#SBATCH -o mpi_%A.out
#SBATCH --mail-type=END
#SBATCH --mail-user=annika.stuke@aalto.fi

boss op modified.rst
