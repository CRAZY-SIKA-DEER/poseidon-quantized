from pathlib import Path
from huggingface_hub import snapshot_download

DATA_ROOT = Path("/lus/lfs1aip2/projects/u6ey/yiheng.u6ey/poseidon/dataset")

datasets = [
    "CE-RM",
    "CE-RPUI",
    "Wave-Gauss",
    "Wave-Layer",
    "NS-BB",
]

for name in datasets:
    repo_id = f"camlab-ethz/{name}"
    local_dir = DATA_ROOT / name

    print(f"[Downloading] {repo_id} -> {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"[Done] {name}")