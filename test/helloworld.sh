#!/bin/bash

#PBS -q A_dev
#PBS -P NIFS25KIST067
#PBS -l select=1:ncpus=4:mem=6gb
#PBS -l walltime=00:5:00
module load intel/2025.1
cd $PBS_O_WORKDIR


# 1. ホスト名を取得
HOSTNAME=$(head -n 1 $PBS_NODEFILE)

# --- デバッグ情報 ---
echo "--- DEBUG INFO (Attempt H) ---"
echo "Forcing bootstrap method to 'ssh' (bypassing PBS PMI)"
echo "Hostname is: $HOSTNAME"
echo "------------------------------"

# ↓↓↓ これが今回の解決策 ↓↓↓
# PBSのPMI連携をあきらめ、ssh を強制的に使わせる
export I_MPI_HYDRA_BOOTSTRAP=ssh

# "sa107" というホストで 4 プロセス起動しろ、と明示
mpirun -np 4 -hosts $HOSTNAME ./mpi_hello_world
