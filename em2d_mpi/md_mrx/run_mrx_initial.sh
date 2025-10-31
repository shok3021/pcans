#!/bin/bash

#PBS -q A_dev
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=12:mpiprocs=12:ompthreads=1:mem=6gb
#PBS -l walltime=00:5:00
module load intel/2025.1
cd /home/shok/pcans/em2d_mpi/md_mrx
mpirun -np 12 -ppn 12 ./a.out