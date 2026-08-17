#!/bin/bash -l
#SBATCH --job-name=download_aurora_era5
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=23:59:00
#SBATCH --output=logs/aurora_era5_download-%j.out
#SBATCH --error=logs/aurora_era5_download-%j.err

set -euo pipefail

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs dataset/aurora/era5_025/raw

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:${PYTHONPATH:-}

# Aurora uses two input states to predict the next state. For 1024 calibration
# windows plus 1024 validation/test windows, we need 2048 + 2 = 2050 time
# points. With four 6-hourly ERA5 states per day, this is 513 days. We use
# 514 days by default to provide a small buffer.
START_DATE="${AURORA_ERA5_START_DATE:-2023-01-01}"
NUM_DAYS="${AURORA_ERA5_NUM_DAYS:-514}"
OUTPUT_DIR="${AURORA_ERA5_OUTPUT_DIR:-dataset/aurora/era5_025/raw}"

echo "[INFO] Downloading Aurora ERA5 subset"
echo "[INFO] START_DATE=${START_DATE}"
echo "[INFO] NUM_DAYS=${NUM_DAYS}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] Times: 00:00 06:00 12:00 18:00"
echo "[INFO] Existing complete files will be skipped."

python -u download_aurora_era5_subset.py \
    --output-dir "${OUTPUT_DIR}" \
    --start-date "${START_DATE}" \
    --num-days "${NUM_DAYS}" \
    --times 00:00 06:00 12:00 18:00

echo "[DONE] Aurora ERA5 download finished."
