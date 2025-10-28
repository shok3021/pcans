#!/bin/bash

#PBS -q A_dev_i
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=12:mem=6gb
#PBS -l walltime=00:5:00
module load intel/2025.1
./a.out