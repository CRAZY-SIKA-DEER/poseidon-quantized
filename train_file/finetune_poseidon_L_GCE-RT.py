import subprocess

# Setup your configs here
config = "run_T.yaml"
model = "Poseidon-T"
run_name = "finetune_wave_layer_T"


# DONT CHANGE CODE HERE
command = [
    "accelerate", "launch",
    "scOT/train.py",
    "--config", "configs/run_L_GCE-RT.yaml",
    "--wandb_run_name", "finetune_GCE-RT_run_L",
    "--wandb_project_name", "PoseidonFinetune_GCE-RT_L",
    "--checkpoint_path", "checkpoints/finetune_GCE-RT_L",
    "--data_path", "datasets/GCE-RT",
    "--finetune_from", "camlab-ethz/Poseidon-L",
    "--replace_embedding_recovery"
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Training script failed with return code {e.returncode}")