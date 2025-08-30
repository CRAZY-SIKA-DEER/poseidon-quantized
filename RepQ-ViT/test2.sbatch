#!/bin/bash
#SBATCH --job-name=code_tunnel
#SBATCH --partition=workq
#SBATCH --time=7:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1          # matches "gpu:4" shown by sinfo
#SBATCH --cpus-per-task=8     # request 8 CPU cores
#SBATCH --mem=64G             # request 64 GB memory
#SBATCH --output=code_tunnel_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch_cuda_env

export PYTHONPATH=$(pwd)
export TMPDIR=$PWD/tmp
mkdir -p "$TMPDIR"

srun python RepQ-ViT/RepQ_NS-PwC_2048.py
