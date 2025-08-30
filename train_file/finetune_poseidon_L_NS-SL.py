import subprocess

# Setup your configs here
config = "run_T.yaml"
model = "Poseidon-T"
run_name = "finetune_wave_layer_T"


# DONT CHANGE CODE HERE
command = [
    "accelerate", "launch",
    "scOT/train.py",
    "--config", "configs/run_L_NS-SL_2048.yaml",
    "--wandb_run_name", "finetune_NS-SL_run_L_2048",
    "--wandb_project_name", "PoseidonFinetune_NS-SL_L",
    "--checkpoint_path", "checkpoints/finetune_NS-SL_L_2048",
    "--data_path", "datasets/NS-SL",
    "--finetune_from", "camlab-ethz/Poseidon-L",
    "--replace_embedding_recovery",
    # if resuem training then use thsi command
    "--resume_training",
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Training script failed with return code {e.returncode}")