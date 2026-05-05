#!/bin/bash -l
#SBATCH --job-name=launch_layerwise_ppq
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_layerwise_ppq-%j.out
#SBATCH --error=logs/launch_layerwise_ppq-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

RUN_LIST=(
  "NS-SVS-L|NS-SVS|fluids.incompressible.VortexSheet"
#   "NS-BB-L|NS-BB|fluids.incompressible.BrownianBridge"
#   "CE-RPUI-L|CE-RPUI|fluids.compressible.RiemannKelvinHelmholtz"
#   "Wave-Gauss-L|Wave-Gauss|wave.Gaussians"
#   "Wave-Layer-L|Wave-Layer|wave.Layer"
)

for item in "${RUN_LIST[@]}"; do
    IFS="|" read -r model dataset dataset_name <<< "$item"

    echo "Submitting layerwise PPQ: model=${model}, dataset=${dataset}, dataset_name=${dataset_name}"

    sbatch --job-name="layer_ppq_${model}" \
        --nodes=1 \
        --gpus=1 \
        --time=23:59:00 \
        --output="logs/layer_ppq_${model}-%j.out" \
        --error="logs/layer_ppq_${model}-%j.err" \
        --wrap="
            cd /home/u6ey/yiheng.u6ey/poseidon-quantized
            source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
            conda activate ppq

            export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH

            export PPQ_MODEL_PATH=models/${model}
            export PPQ_DATA_PATH=dataset/${dataset}
            export PPQ_DATASET_NAME=${dataset_name}
            export SAPQ_PRIOR_MODE=ppq

            python -u SAPQ/run_ppq_all.py
        "

    sleep 0.5
done

echo "All layerwise PPQ jobs submitted."