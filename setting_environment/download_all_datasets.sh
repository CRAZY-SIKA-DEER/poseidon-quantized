#!/bin/bash

# List of datasets you want
datasets=(
    "NS-Gauss"
    "NS-Sines"
    "CE-RP"
    "CE-KH"
    "CE-Gauss"
    "NS-PwC"
    "NS-SVS"
    "NS-BB"
    "NS-SL"
    "NS-Tracer-PwC"
    "FNS-KF"
    "CE-RPUI"
    "CE-RM"
    "GCE-RT"
    "Wave-Layer"
    "Wave-Gauss"
    "ACE"
    "SE-AF"
    "Poisson-Gauss"
    "Helmholtz"
)

# Create parent directory if it doesn't exist
mkdir -p datasets

# Loop through each dataset and download
for dataset in "${datasets[@]}"
do
    echo "Downloading $dataset..."
    huggingface-cli download camlab-ethz/$dataset --repo-type dataset --local-dir datasets/$dataset
done

echo "✅ All datasets downloaded!"