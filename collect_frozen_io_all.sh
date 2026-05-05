#!/bin/bash -l
#SBATCH --job-name=launch_cache_io
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_cache_io-%j.out
#SBATCH --error=logs/launch_cache_io-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

RUN_LIST=(
#   "NS-PwC-L|NS-PwC|fluids.incompressible.PiecewiseConstants"
  "NS-SVS-L|NS-SVS|fluids.incompressible.VortexSheet"
#   "NS-BB-L|NS-BB|fluids.incompressible.BrownianBridge"
#   "CE-RM-L|CE-RM|fluids.compressible.RichtmyerMeshkov"
#   "CE-RPUI-L|CE-RPUI|fluids.compressible.RiemannKelvinHelmholtz"
#   "Wave-Gauss-L|Wave-Gauss|wave.Gaussians"
#   "Wave-Layer-L|Wave-Layer|wave.Layer"
)

for item in "${RUN_LIST[@]}"; do
    IFS="|" read -r model dataset dataset_name <<< "$item"

    echo "Submitting cache IO: model=${model}, dataset=${dataset}, dataset_name=${dataset_name}"

    sbatch --job-name="cache_${model}" \
        --nodes=1 \
        --gpus=1 \
        --time=23:59:00 \
        --output="logs/cache_${model}-%j.out" \
        --error="logs/cache_${model}-%j.err" \
        --wrap="
            cd /home/u6ey/yiheng.u6ey/poseidon-quantized
            source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
            conda activate ppq

            export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH

            export PPQ_MODEL_PATH=models/${model}
            export PPQ_DATA_PATH=dataset/${dataset}
            export PPQ_DATASET_NAME=${dataset_name}

            python -u PPQ/caching_layer_wise_io.py
        "

    sleep 0.5
done

echo "All cache IO jobs submitted."