#!/bin/bash -l
#SBATCH --job-name=brecq_poseidon
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:59:00
#SBATCH --output=logs/ppq_poseidon-%j.out
#SBATCH --error=logs/ppq_poseidon-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

# # for MSE
# python BRECQ/quant/main_poseidon_brecq.py \
#   --model_path models/NS-PwC-L \
#   --dataset_name fluids.incompressible.PiecewiseConstants \
#   --data_path dataset/NS-PwC \
#   --device cuda \
#   --n_bits_w 4 \
#   --channel_wise \
#   --calib_batchsize 2 \
#   --calib_steps 2 \
#   --val_batchsize 2 \
#   --val_steps 2 \
#   --batch_size_recon 2 \
#   --iters_w 10 \
#   --opt_mode mse \
#   --test_before_calibration


# for Fisher Diagonal
python BRECQ/quant/main_poseidon_brecq.py \
  --model_path models/NS-PwC-L \
  --dataset_name fluids.incompressible.PiecewiseConstants \
  --data_path dataset/NS-PwC \
  --device cuda \
  --n_bits_w 4 \
  --channel_wise \
  --calib_batchsize 2 \
  --calib_steps 2 \
  --val_batchsize 2 \
  --val_steps 2 \
  --batch_size_recon 2 \
  --iters_w 10 \
  --opt_mode fisher_diag \
  --test_before_calibration



# # for Fisher Full
# python BRECQ/quant/main_poseidon_brecq.py \
#   --model_path models/NS-PwC-L \
#   --dataset_name fluids.incompressible.PiecewiseConstants \
#   --data_path dataset/NS-PwC \
#   --device cuda \
#   --n_bits_w 4 \
#   --channel_wise \
#   --calib_batchsize 2 \
#   --calib_steps 2 \
#   --val_batchsize 2 \
#   --val_steps 2 \
#   --batch_size_recon 2 \
#   --iters_w 10 \
#   --opt_mode fisher_full \
#   --test_before_calibration