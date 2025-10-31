#!/bin/bash
#PBS -q A_dev
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=4:mem=6gb
#PBS -l walltime=00:5:00

module load intel/2025.1

# Load the Open MPI module you tested interactively
module load openmpi/5.0.7/gcc11.5.0

# --- Debug Info ---
echo "--- DEBUG INFO (Attempt Q - Using Open MPI) ---"
echo "Using Open MPI module: openmpi/5.0.7/gcc11.5.0"
echo "Using mpirun -np 4"
echo "------------------------------"

# Change to the submission directory
cd ${PBS_O_WORKDIR}

# Recompile with Open MPI's mpif90 just in case
# (It's good practice to compile and run with the same MPI)
mpif90 -o mpi_hello_world mpi_hello_world.f90

# Run with Open MPI's mpirun, explicitly specifying 4 processes
mpirun -np 4 ./mpi_hello_world
