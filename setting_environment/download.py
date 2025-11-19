#!/usr/bin/env python3
import os
import subprocess
import sys


def run(cmd: str):
    """Run a shell command with logging and error checking."""
    print(f"\n=== Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}: {cmd}")
        sys.exit(result.returncode)


def ensure_dir(path: str):
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def download_datasets():
    print("\n==============================")
    print("📦  Downloading NS datasets...")
    print("==============================\n")

    # NS-SVS
    ensure_dir("dataset/NS-SVS")
    run("hf download camlab-ethz/NS-SVS --repo-type dataset --local-dir dataset/NS-SVS")
    run("python dataset/NS-SVS/assemble_data.py "
        "--input_dir dataset/NS-SVS "
        "--output_file dataset/NS-SVS/NS-SVS.nc")

    # NS-BB
    ensure_dir("dataset/NS-BB")
    run("hf download camlab-ethz/NS-BB --repo-type dataset --local-dir dataset/NS-BB")
    run("python dataset/NS-BB/assemble_data.py "
        "--input_dir dataset/NS-BB "
        "--output_file dataset/NS-BB/NS-BB.nc")

    # NS-SL
    ensure_dir("dataset/NS-SL")
    run("hf download camlab-ethz/NS-SL --repo-type dataset --local-dir dataset/NS-SL")
    run("python dataset/NS-SL/assemble_data.py "
        "--input_dir dataset/NS-SL "
        "--output_file dataset/NS-SL/NS-SL.nc")


def download_models():
    print("\n==============================")
    print("🤖  Downloading Poseidon models...")
    print("==============================\n")

    # L models
    run(
        "wandb artifact get "
        "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-PwC_L_2048/"
        "poseidon_finetune_NS-PwC_L_2048_ckpt368800:v0 "
        "--root models/NS-PwC"
    )

    run(
        "wandb artifact get "
        "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-SVS_L_2048/"
        "poseidon_finetune_NS-SVS_L_2048_ckpt368800:v0 "
        "--root models/NS-SVS"
    )

    run(
        "wandb artifact get "
        "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-BB_L_2048/"
        "poseidon_finetune_NS-BB_L_2048_ckpt368800:v0 "
        "--root models/NS-BB"
    )

    # B models
    run(
        "wandb artifact get "
        "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-PwC_B_2048/"
        "poseidon_finetune_NS-PwC_B_2048_ckpt368800:v0 "
        "--root models/NS-PwC-B"
    )

    run(
        "wandb artifact get "
        "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-BB_B_2048/"
        "poseidon_finetune_NS-BB_B_2048_ckpt368800:v0 "
        "--root models/NS-BB-B"
    )

    run(
        "wandb artifact get "
        "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-SVS_B_2048/"
        "poseidon_finetune_NS-SVS_B_2048_ckpt368800:v0 "
        "--root models/NS-SVS-B"
    )

    run(
        "wandb artifact get "
        "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-SL_B_2048/"
        "poseidon_finetune_NS-SL_B_2048_ckpt368800:v0 "
        "--root models/NS-SL-B"
    )


def main():
    print("\n==========================================")
    print("Poseidon Dataset & Model Download Utility")
    print("==========================================\n")

    print("Make sure you have installed:")
    print("  - huggingface_hub (provides `hf` CLI)")
    print("  - wandb (for downloading model artifacts)")
    print("\nLogin first if needed:")
    print("  hf login")
    print("  wandb login\n")

    download_datasets()
    download_models()

    print("\n🎉 All datasets and models downloaded successfully!\n")


if __name__ == "__main__":
    main()
