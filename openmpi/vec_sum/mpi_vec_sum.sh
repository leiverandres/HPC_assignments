#!/bin/bash
#
#SBATCH --job-name=vectorSum
#SBATCH --output=res_mpi_vec_sum.out
#SBATCH --ntasks=3
#SBATCH --nodes=3
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1

mpirun mpi_vec_sum
