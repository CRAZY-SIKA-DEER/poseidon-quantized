#!/bin/bash -l
#SBATCH --job-name=assemble_poseidon_data
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/assemble_data-%j.out
#SBATCH --error=logs/assemble_data-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH

python -u assemble_selected.py