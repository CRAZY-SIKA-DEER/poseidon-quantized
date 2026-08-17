#!/bin/bash
#SBATCH --job-name=code_tunnel
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=code_tunnel_%j.out

~/opt/vscode_cli/code tunnel --name "yiheng-isambard-gpu1"
