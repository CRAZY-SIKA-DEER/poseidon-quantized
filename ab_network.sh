#!/bin/bash -l
#SBATCH --job-name=launch_sapq_global
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_sapq_global-%j.out
#SBATCH --error=logs/launch_sapq_global-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

PRIOR_LIST=(
    "ppq"
    "block_no_sens"
    "block_sens"
)

for prior_mode in "${PRIOR_LIST[@]}"; do
    echo "Submitting SAPQ global with prior_mode=${prior_mode}"

    sbatch --job-name="sapq_${prior_mode}" \
        --nodes=1 \
        --gpus=1 \
        --time=23:59:00 \
        --output="logs/network_${prior_mode}-%j.out" \
        --error="logs/network_${prior_mode}-%j.err" \
        --wrap="
            cd /home/u6ey/yiheng.u6ey/poseidon-quantized
            source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
            conda activate ppq

            export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH
            export SAPQ_PRIOR_MODE=${prior_mode}

            python -u SAPQ/run_sapq_network_global.py
        "

    sleep 0.5
done

echo "All SAPQ global jobs submitted."