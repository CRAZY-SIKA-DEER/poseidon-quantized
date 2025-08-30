import subprocess

# Setup your configs here
config = "run_T.yaml"
model = "Poseidon-T"
run_name = "finetune_wave_layer_T"


# DONT CHANGE CODE HERE
command = [
    "accelerate", "launch",
    "scOT/train.py",
    "--config", "configs/run_B_NS-BB_2048.yaml",
    "--wandb_run_name", "finetune_NS-BB_run_B_2048",
    "--wandb_project_name", "PoseidonFinetune_NS-BB_B",
    "--checkpoint_path", "checkpoints/finetune_NS-BB_B_2048",
    "--data_path", "datasets/NS-BB",
    "--finetune_from", "camlab-ethz/Poseidon-B",
    "--replace_embedding_recovery",
    # if resume training then use this comannd
    "--resume_training",
]

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as e:
    print(f"Training script failed with return code {e.returncode}")