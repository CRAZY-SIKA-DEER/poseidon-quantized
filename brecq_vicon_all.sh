#!/bin/bash -l
#SBATCH --job-name=launch_vicon_brecq
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_vicon_brecq-%j.out
#SBATCH --error=logs/launch_vicon_brecq-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

DATASET_LIST=(
  "NS2D"
  "COMPRESSIBLE2D"
  "EULER2D"
)

BIT_LIST=(4)
ITER_LIST=(2000)

CKPT_PATH="models/vicon/vicon.pth"

for dataset in "${DATASET_LIST[@]}"; do
    for bits in "${BIT_LIST[@]}"; do
        for iters in "${ITER_LIST[@]}"; do

            SCALE_PATH="brecq_artifacts/VICON/weight_scales/w${bits}_channelwise_mse80.pt"

            echo "Submitting VICON BRECQ: dataset=${dataset}, w${bits}, iters=${iters}"

            sbatch --job-name="vicon_brecq_${dataset}_w${bits}_i${iters}" \
                --nodes=1 \
                --gpus=1 \
                --time=23:59:00 \
                --output="logs/vicon_brecq_${dataset}_w${bits}_i${iters}-%j.out" \
                --error="logs/vicon_brecq_${dataset}_w${bits}_i${iters}-%j.err" \
                --wrap="
                    cd /home/u6ey/yiheng.u6ey/poseidon-quantized

                    source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
                    conda activate ppq

                    export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH

                    python -u BRECQ/run_vicon_brecq.py \
                      --dataset_name ${dataset} \
                      --ckpt_path ${CKPT_PATH} \
                      --n_bits_w ${bits} \
                      --channel_wise \
                      --calib_batchsize 2 \
                      --calib_steps 512 \
                      --recon_iters ${iters} \
                      --recon_batch_size 32 \
                      --opt_mode mse \
                      --asym \
                      --device cuda \
                      --weight_scales_path ${SCALE_PATH}
                "

            sleep 0.5
        done
    done
done

echo "All VICON BRECQ jobs submitted."