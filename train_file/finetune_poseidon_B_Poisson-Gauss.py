import subprocess

# Setup your configs here
config = "run_T.yaml"
model = "Poseidon-T"
run_name = "finetune_wave_layer_T"


# DONT CHANGE CODE HERE
command = [
    "accelerate", "launch",
    "scOT/train.py",
    "--config", "configs/run_B_Poisson-Gauss.yaml",
    "--wandb_run_name", "finetune_Poisson-Gauss_run_B",
    "--wandb_project_name", "PoseidonFinetune_Poisson-Gauss_B",
    "--checkpoint_path", "checkpoints/finetune_Poisson-Gauss_B",
    "--data_path", "datasets/Poisson-Gauss",
    "--finetune_from", "camlab-ethz/Poseidon-B",
    "--replace_embedding_recovery"
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Training script failed with return code {e.returncode}")