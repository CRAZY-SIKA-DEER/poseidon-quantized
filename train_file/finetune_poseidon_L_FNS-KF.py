import subprocess

# Setup your configs here
config = "run_T.yaml"
model = "Poseidon-T"
run_name = "finetune_wave_layer_T"


# DONT CHANGE CODE HERE
command = [
    "accelerate", "launch",
    "scOT/train.py",
    "--config", "configs/run_L_FNS-KF.yaml",
    "--wandb_run_name", "finetune_FNS-KF_run_L",
    "--wandb_project_name", "PoseidonFinetune_FNS-KF_L",
    "--checkpoint_path", "checkpoints/finetune_FNS-KF_L",
    "--data_path", "datasets/FNS-KF",
    "--finetune_from", "camlab-ethz/Poseidon-L",
    "--replace_embedding_recovery"
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Training script failed with return code {e.returncode}")