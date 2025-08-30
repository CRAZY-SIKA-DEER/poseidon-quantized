#!/bin/bash

#SBATCH --job-name=evaluate_dynamic_quantize
#SBATCH --output=eval_dynamic_quantize_%j.out
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# Make sure Python knows where to find scOT
export PYTHONPATH=$(pwd)

# Run the evaluation
python scOT/compression/dynamic_quantize/evaluation_dynamic_quantize.py