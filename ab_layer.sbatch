#!/bin/bash -l
#SBATCH --job-name=run_layerwise_ppq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:59:00
#SBATCH --output=logs/ppq_poseidon-%j.out
#SBATCH --error=logs/ppq_poseidon-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq


PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH python -u SAPQ/run_sapq_layerwise.py