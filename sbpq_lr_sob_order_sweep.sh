#!/bin/bash -l
#SBATCH --job-name=launch_sbpq_lr_sob
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_sbpq_lr_sob-%j.out
#SBATCH --error=logs/launch_sbpq_lr_sob-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

RUN_LIST=(
  "NS-PwC-L|NS-PwC|fluids.incompressible.PiecewiseConstants"
  # "NS-SVS-L|NS-SVS|fluids.incompressible.VortexSheet"
  # "NS-BB-L|NS-BB|fluids.incompressible.BrownianBridge"
  # "CE-RPUI-L|CE-RPUI|fluids.compressible.RiemannKelvinHelmholtz"
  # "Wave-Gauss-L|Wave-Gauss|wave.Gaussians"
  # "Wave-Layer-L|Wave-Layer|wave.Layer"
)

TARGET_BITS=(
  "4"
  "8"
)

SOBOLEV_ORDERS=(
  "0"
  "1"
  "2"
  "3"
)

LEARNING_RATES=(
  "1e-2"
  "3e-3"
  "1e-3"
  "3e-4"
  "1e-4"
  "3e-5"
  "1e-5"
  "3e-6"
  "1e-6"
)

job_count=0

for item in "${RUN_LIST[@]}"; do
    IFS="|" read -r model dataset dataset_name <<< "$item"

    for bits in "${TARGET_BITS[@]}"; do
        for sob_order in "${SOBOLEV_ORDERS[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                lr_tag="${lr//./p}"
                lr_tag="${lr_tag//-/m}"

                job_name="sbpq_${model}_B${bits}_sob${sob_order}_lr${lr_tag}"

                echo "Submitting ${job_name}: dataset=${dataset}, dataset_name=${dataset_name}"

                sbatch --job-name="${job_name}" \
                    --nodes=1 \
                    --gpus=1 \
                    --time=23:59:00 \
                    --output="logs/${job_name}-%j.out" \
                    --error="logs/${job_name}-%j.err" \
                    --wrap="
                        cd /home/u6ey/yiheng.u6ey/poseidon-quantized

                        source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
                        conda activate ppq

                        export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH

                        export SBPQ_MODEL_PATH=models/${model}
                        export SBPQ_DATA_PATH=dataset/${dataset}
                        export SBPQ_DATASET_NAME=${dataset_name}

                        export SBPQ_REFERENCE_BITS=${bits}
                        export SBPQ_INIT_BITS=${bits}
                        export SBPQ_DELTA_BITS=2
                        export SBPQ_BETA_KAPPA=10
                        export SBPQ_BETA_PRIOR_SCALE=1

                        export SBPQ_SOBOLEV_ORDER=${sob_order}
                        export SBPQ_SOBOLEV_NORM=l1

                        export SBPQ_LEARNING_RATE=${lr}
                        export SBPQ_NUM_MC_SAMPLES=10
                        export SBPQ_ETA=1e-6

                        export SBPQ_CALIB_BATCH_SIZE=2
                        export SBPQ_CALIB_STEPS=512
                        export SBPQ_SENSITIVITY_BATCHES=512
                        export SBPQ_VAL_BATCH_SIZE=128
                        export SBPQ_VAL_STEPS=2
                        export SBPQ_NUM_OPTIMIZATION_STEPS=20
                        export SBPQ_NUM_WORKERS=0

                        python -u run_sbpq_poseidon.py
                    "

                job_count=$((job_count + 1))
                sleep 0.5
            done
        done
    done
done

echo "Submitted ${job_count} SBPQ learning-rate/Sobolev-order sweep jobs."
