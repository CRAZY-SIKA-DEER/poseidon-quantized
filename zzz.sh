#!/bin/bash -l
#SBATCH --job-name=sapqGn_CE_RPUI9
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:59:00
#SBATCH --output=logs/sapqGn_CE_RPUI-%j.out
#SBATCH --error=logs/sapqGn_CE_RPUI-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH

export PPQ_MODEL_PATH=models/CE-RPUI-L
export PPQ_DATA_PATH=dataset/CE-RPUI
export PPQ_DATASET_NAME=fluids.compressible.RiemannKelvinHelmholtz
export SAPQ_PRIOR_MODE=block_sens

python -u SAPQ/run_sapq_network_global_raw.py