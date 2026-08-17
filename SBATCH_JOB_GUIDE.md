# Slurm Job Script Guide

This guide explains how to write correct `.sh` launcher files for this
repository on Isambard. It is intended for a new GPT/Codex conversation that
needs to create or submit experiment jobs.

## Key Distinction

Do not confuse these two script types:

1. `vscode_gpu.sh`
   - Requests a GPU node for a VS Code tunnel.
   - Runs `code tunnel`.
   - It is not an experiment launcher.
   - Do not use it when asked to submit training/evaluation jobs.

2. Normal experiment launcher `.sh` files
   - Examples: `download_selected.sh`, `ppq_poseidon_all.sh`,
     `sapq_poseidon_all.sh`, `collect_range_all.sh`.
   - These set up the repo environment and submit or run Python experiments.
   - Use this style for SBPQ, SAPQ, PPQ, range collection, evaluation, etc.

## Required Environment Setup

Every normal experiment script should contain this setup before running Python:

```bash
cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH
```

The `PYTHONPATH` export is required so local packages such as `SBPQ`, `SAPQ`,
`PPQ`, `scOT`, and local utility modules resolve correctly.

## Single Job Script Pattern

Use this when the `.sh` file itself runs one job.

```bash
#!/bin/bash -l
#SBATCH --job-name=sbpq_poseidon_nspwc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=23:59:00
#SBATCH --output=logs/sbpq_poseidon_nspwc-%j.out
#SBATCH --error=logs/sbpq_poseidon_nspwc-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
conda activate ppq

export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:$PYTHONPATH

export SBPQ_MODEL_PATH=models/NS-PwC-L
export SBPQ_DATA_PATH=dataset/NS-PwC
export SBPQ_DATASET_NAME=fluids.incompressible.PiecewiseConstants

python -u run_sbpq_poseidon.py
```

Submit it with:

```bash
sbatch script_name.sh
```

## Launcher Script Pattern For Multiple Jobs

Use this when one launcher submits many independent GPU jobs in parallel.
The launcher itself usually needs no GPU.

```bash
#!/bin/bash -l
#SBATCH --job-name=launch_sbpq_poseidon
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --time=1:00:00
#SBATCH --output=logs/launch_sbpq_poseidon-%j.out
#SBATCH --error=logs/launch_sbpq_poseidon-%j.err

cd /home/u6ey/yiheng.u6ey/poseidon-quantized
mkdir -p logs

RUN_LIST=(
  "NS-PwC-L|NS-PwC|fluids.incompressible.PiecewiseConstants"
  "NS-SVS-L|NS-SVS|fluids.incompressible.VortexSheet"
  "NS-BB-L|NS-BB|fluids.incompressible.BrownianBridge"
  "CE-RPUI-L|CE-RPUI|fluids.compressible.RiemannKelvinHelmholtz"
  "Wave-Gauss-L|Wave-Gauss|wave.Gaussians"
  "Wave-Layer-L|Wave-Layer|wave.Layer"
)

for item in "${RUN_LIST[@]}"; do
    IFS="|" read -r model dataset dataset_name <<< "$item"

    echo "Submitting SBPQ: model=${model}, dataset=${dataset}, dataset_name=${dataset_name}"

    sbatch --job-name="sbpq_${model}" \
        --nodes=1 \
        --gpus=1 \
        --time=23:59:00 \
        --output="logs/sbpq_${model}-%j.out" \
        --error="logs/sbpq_${model}-%j.err" \
        --wrap="
            cd /home/u6ey/yiheng.u6ey/poseidon-quantized

            source /home/u6ey/yiheng.u6ey/miniforge3/etc/profile.d/conda.sh
            conda activate ppq

            export PYTHONPATH=/home/u6ey/yiheng.u6ey/poseidon-quantized:\$PYTHONPATH

            export SBPQ_MODEL_PATH=models/${model}
            export SBPQ_DATA_PATH=dataset/${dataset}
            export SBPQ_DATASET_NAME=${dataset_name}

            python -u run_sbpq_poseidon.py
        "

    sleep 0.5
done

echo "All SBPQ jobs submitted."
```

Important detail: inside `--wrap`, escape `$PYTHONPATH` as `\$PYTHONPATH`.
Otherwise the outer launcher shell expands it too early.

## Naming And Logs

Use explicit names so jobs and logs are easy to identify:

```bash
--job-name="sbpq_${model}"
--output="logs/sbpq_${model}-%j.out"
--error="logs/sbpq_${model}-%j.err"
```

`%j` is the Slurm job ID. Keep it in log filenames.

Good prefixes:

- `sbpq_...` for SBPQ training/evaluation
- `sapq_...` for SAPQ
- `ppq_...` for PPQ
- `ranges_...` for range collection
- `eval_...` for evaluation-only jobs
- `launch_...` for CPU launcher jobs that submit child jobs

## Environment Variables

For SBPQ Poseidon jobs, prefer environment variables over hard-coded config
edits:

```bash
export SBPQ_MODEL_PATH=models/NS-PwC-L
export SBPQ_DATA_PATH=dataset/NS-PwC
export SBPQ_DATASET_NAME=fluids.incompressible.PiecewiseConstants
```

Useful SBPQ debug/smoke overrides:

```bash
export SBPQ_CALIB_STEPS=1
export SBPQ_VAL_STEPS=1
export SBPQ_SENSITIVITY_BATCHES=1
export SBPQ_NUM_OPTIMIZATION_STEPS=2
export SBPQ_NUM_MC_SAMPLES=1
export SBPQ_CALIB_BATCH_SIZE=1
export SBPQ_VAL_BATCH_SIZE=1
export SBPQ_NUM_WORKERS=0
```

Do not include those smoke overrides in a real full training script unless the
user explicitly asks for a short debug run.

## Validation Rule

For fair comparison between FP, SBPQ, and fixed-bit baselines, the Python code
should evaluate all methods on the same validation batches. A shuffled
validation dataloader or repeatedly calling a shuffled iterator can make 16-bit
appear better than FP by accident.

The preferred Python-side pattern is:

```python
frozen_val_batches = list(val_iterator())
evaluate_fp(frozen_val_batches)
evaluate_sbpq(frozen_val_batches)
evaluate_fixed_baselines(frozen_val_batches)
```

## Before Submitting

Check these items:

- The script uses `#!/bin/bash -l`.
- It has correct `#SBATCH` resources.
- It creates `logs/`.
- It activates conda env `ppq`.
- It exports `PYTHONPATH`.
- It uses the correct method-specific env vars, e.g. `SBPQ_*` for SBPQ.
- It writes output/error logs with meaningful names and `%j`.
- It does not use `vscode_gpu.sh` unless the user specifically asks for a VS
  Code tunnel node.

## Submitting And Checking Jobs

Submit:

```bash
sbatch script_name.sh
```

Check current jobs:

```bash
squeue --me
```

Inspect a running/completed log:

```bash
tail -f logs/job_name-JOBID.out
```

