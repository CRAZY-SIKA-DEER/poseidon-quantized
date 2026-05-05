#!/bin/bash -l
#SBATCH --job-name=vicon_ppq_layerwise_ns2d
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:59:00
#SBATCH --output=logs/vicon_ppq_layerwise_all-%j.out
#SBATCH --error=logs/vicon_ppq_layerwise_all-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized/VICON
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

CKPT=/home/u6ey/yiheng.u6ey/poseidon-quantized/models/vicon/vicon.pth

# # -------------------------
# # NS2D
# # -------------------------
# python -u run_ppq_vicon_layerwise.py \
#   +ppq.ckpt_path=${CKPT} \
#   +ppq.dataset=NS2D \
#   +ppq.percentile_prob=1e-4 \
#   +ppq.num_epochs=50 \
#   +ppq.num_mc_samples=10 \
#   +ppq.base_lr=9.1e-4 \
#   +ppq.eta=1e-6 \
#   +ppq.init_bits=8 \
#   +ppq.bmax_bits=20 \
#   +ppq.log_every=1 \
#   +ppq.gamma=0.005 \
#   board=0 plot=0 amp=0

# # -------------------------
# # COMPRESSIBLE2D
# # change percentile if best P is different
# # -------------------------
# python -u run_ppq_vicon_layerwise.py \
#   +ppq.ckpt_path=${CKPT} \
#   +ppq.dataset=COMPRESSIBLE2D \
#   +ppq.percentile_prob=1e-2 \
#   +ppq.num_epochs=10 \
#   +ppq.num_mc_samples=10 \
#   +ppq.base_lr=9.1e-4 \
#   +ppq.eta=1e-6 \
#   +ppq.init_bits=8 \
#   +ppq.bmax_bits=20 \
#   +ppq.log_every=1 \
#   +ppq.gamma=0.005 \
#   board=0 plot=0 amp=0

# # -------------------------
# # EULER2D
# # change percentile if best P is different
# # -------------------------
# python -u run_ppq_vicon_layerwise.py \
#   +ppq.ckpt_path=${CKPT} \
#   +ppq.dataset=EULER2D \
#   +ppq.percentile_prob=1e-2 \
#   +ppq.num_epochs=50 \
#   +ppq.num_mc_samples=10 \
#   +ppq.base_lr=9.1e-4 \
#   +ppq.eta=1e-6 \
#   +ppq.init_bits=8 \
#   +ppq.bmax_bits=20 \
#   +ppq.log_every=1 \
#   +ppq.gamma=0.005 \
#   board=0 plot=0 amp=0