import subprocess

# Setup your configs here
config = "run_T.yaml"
model = "Poseidon-T"
run_name = "finetune_wave_layer_T"


# DONT CHANGE CODE HERE
command = [
    "accelerate", "launch",
    "scOT/train.py",
    "--config", "configs/run_B_CE-RPUI.yaml",
    "--wandb_run_name", "finetune_CE-RPUI_run_B",
    "--wandb_project_name", "PoseidonFinetune_CE-RPUI_B",
    "--checkpoint_path", "checkpoints/finetune_CE-RPUI_B",
    "--data_path", "datasets/CE-RPUI",
    "--finetune_from", "camlab-ethz/Poseidon-B",
    "--replace_embedding_recovery"
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Training script failed with return code {e.returncode}")