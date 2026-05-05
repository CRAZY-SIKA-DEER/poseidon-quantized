#!/bin/bash -l
#SBATCH --job-name=launch_brecq
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_brecq-%j.out
#SBATCH --error=logs/launch_brecq-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

RUN_LIST=(
  "NS-SVS-L|NS-SVS|fluids.incompressible.VortexSheet"
  "NS-BB-L|NS-BB|fluids.incompressible.BrownianBridge"
  "CE-RPUI-L|CE-RPUI|fluids.compressible.RiemannKelvinHelmholtz"
  "Wave-Gauss-L|Wave-Gauss|wave.Gaussians"
  "Wave-Layer-L|Wave-Layer|wave.Layer"
  "NS-PwC-L|NS-PwC|fluids.incompressible.PiecewiseConstants"
)

BIT_LIST=(4 8)
ITER_LIST=(1000 5000 10000)

for item in "${RUN_LIST[@]}"; do
    IFS="|" read -r model dataset dataset_name <<< "$item"

    for bits in "${BIT_LIST[@]}"; do
        for iters in "${ITER_LIST[@]}"; do

            echo "Submitting BRECQ: model=${model}, dataset=${dataset}, w${bits}, iters=${iters}"

            sbatch --job-name="brecq_${model}_w${bits}_i${iters}" \
                --nodes=1 \
                --gpus=1 \
                --time=23:59:00 \
                --output="logs/brecq_${model}_w${bits}_i${iters}-%j.out" \
                --error="logs/brecq_${model}_w${bits}_i${iters}-%j.err" \
                --wrap="
                    cd /home/u6ey/yiheng.u6ey/poseidon-quantized
                    source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
                    conda activate ppq

                    export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH

                    python -u BRECQ/quant/main_poseidon_brecq.py \
                      --model_path models/${model} \
                      --dataset_name ${dataset_name} \
                      --data_path dataset/${dataset} \
                      --device cuda \
                      --n_bits_w ${bits} \
                      --channel_wise \
                      --calib_batchsize 2 \
                      --calib_steps 512 \
                      --val_batchsize 2 \
                      --val_steps 2 \
                      --batch_size_recon 2 \
                      --iters_w ${iters} \
                      --opt_mode fisher_diag \
                      --test_before_calibration
                "

            sleep 0.5
        done
    done
done

echo "All BRECQ jobs submitted."