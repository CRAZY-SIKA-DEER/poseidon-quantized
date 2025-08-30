#!/bin/bash
#SBATCH --job-name=code_tunnel
#SBATCH --gpus=1
#SBATCH --time=7:50:00
#SBATCH --output=code_tunnel_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch_cuda_env

export PYTHONPATH=$(pwd)
export TMPDIR=$PWD/tmp
mkdir -p $TMPDIR
python RepQ-ViT/RepQ_NS-PwC_2048.py




