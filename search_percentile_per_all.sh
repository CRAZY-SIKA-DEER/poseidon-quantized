#!/bin/bash -l
#SBATCH --job-name=launch_search_p
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_search_p-%j.out
#SBATCH --error=logs/launch_search_p-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

RUN_LIST=(
  "NS-SVS-L|NS-SVS|fluids.incompressible.VortexSheet"
  "NS-BB-L|NS-BB|fluids.incompressible.BrownianBridge"
  "CE-RM-L|CE-RM|fluids.compressible.RichtmyerMeshkov"
  "CE-RPUI-L|CE-RPUI|fluids.compressible.RiemannKelvinHelmholtz"
  "Wave-Gauss-L|Wave-Gauss|wave.Gaussians"
  "Wave-Layer-L|Wave-Layer|wave.Layer"
)

for item in "${RUN_LIST[@]}"; do
    IFS="|" read -r model dataset dataset_name <<< "$item"

    echo "Submitting percentile search: model=${model}, dataset=${dataset}"

    sbatch --job-name="search_${model}" \
        --nodes=1 \
        --gpus=1 \
        --time=23:59:00 \
        --output="logs/search_${model}-%j.out" \
        --error="logs/search_${model}-%j.err" \
        --wrap="
            cd /home/u6ey/yiheng.u6ey/poseidon-quantized
            source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
            conda activate ppq

            export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH

            export PPQ_MODEL_PATH=models/${model}
            export PPQ_DATA_PATH=dataset/${dataset}
            export PPQ_DATASET_NAME=${dataset_name}

            python -u precalculated_ranges/search_percentile_prob.py
        "

    sleep 0.5
done

echo "All percentile search jobs submitted."