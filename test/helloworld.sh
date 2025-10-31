#!/bin/bash
#PBS -q A_dev
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=00:5:00

module load intel/2025.1

module load openmpi/5.0.7/gcc11.5.0

# Change to the submission directory
cd ${PBS_O_WORKDIR}

mpif90 -o mpi_hello_world mpi_hello_world.f90

mpirun ./mpi_hello_world