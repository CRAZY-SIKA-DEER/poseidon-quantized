#!/bin/bash -l
#SBATCH --job-name=run_global_sobo
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:59:00
#SBATCH --output=logs/ppq_poseidon-%j.out
#SBATCH --error=logs/ppq_poseidon-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

#PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH python PPQ/run_ppq_poseidon_layerwise.py
#PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH python precalculated_ranges/collect_data_ranges.py
#PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH python BRECQ/quant/main_poseidon_brecq.py --model_path models/NS-PwC-L --dataset_name fluids.incompressible.PiecewiseConstants --data_path dataset/NS-PwC --device cuda --n_bits_w 4 --channel_wise --calib_batchsize 2 --calib_steps 2 --val_batchsize 2 --val_steps 2 --batch_size_recon 2 --iters_w 10 --test_before_calibration
#PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH python SAPQ/run_sapq_poseidon.py
PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH python -u SAPQ/run_sapq_network_global.py
