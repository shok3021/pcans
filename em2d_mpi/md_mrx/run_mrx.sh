#!/bin/bash

#PBS -q A_S
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=12:mem=64gb
#PBS -l walltime=03:00:00
module load intel/2025.1
/home/shok/pcans/em2d_mpi/md_mrx/a.out