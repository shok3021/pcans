#!/bin/bash
#PBS -q A_dev
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=4:mem=6gb
#PBS -l walltime=00:5:00

# 最初にOpen MPIでコンパイルするために必要な環境をロード
module load intel/2025.1
# NOTE: intel/2025.1が自動でロードするIntel MPIの環境変数を
# Open MPIで上書きするために、Open MPIを最後にロードする
module load openmpi/5.0.7/gcc11.5.0

# --- Debug Info ---
echo "--- DEBUG INFO (Attempt Q - Using Open MPI) ---"
echo "Using Open MPI module: openmpi/5.0.7/gcc11.5.0"
echo "Using mpirun -np 4"
echo "------------------------------"

# Change to the submission directory
cd ${PBS_O_WORKDIR}

# Recompile with Open MPI's mpif90
mpif90 -o mpi_hello_world mpi_hello_world.f90

# Run with Open MPI's mpirun
# Open MPIは通常、PBS環境で-npを省略しても自動的にスロット数を検出します。
# 明示的に指定する場合は、要求した数(4)と一致させます。
mpirun -np 4 ./mpi_hello_world