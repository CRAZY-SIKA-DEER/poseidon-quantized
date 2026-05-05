#!/bin/bash -l
#SBATCH --job-name=download_poseidon_data
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=23:59:00
#SBATCH --output=logs/download_data-%j.out
#SBATCH --error=logs/download_data-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH

python -u download_selected.py