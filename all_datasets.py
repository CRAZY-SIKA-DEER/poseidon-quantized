from huggingface_hub import snapshot_download

datasets = [
    "NS-PwC", "NS-SVS", "NS-BB", "NS-SL", "FNS-KF", "CE-RPUI", "CE-RM",
    "SE-AF", "GCE-RT", "Wave-Layer", "Wave-Gauss", "ACE", "Poisson-Gauss", "Helmholtz"
]

for name in datasets:
    repo_id = f"camlab-ethz/{name}"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=f"poseidon_main/camlab-ethz/down_streams/{name}",
        local_dir_use_symlinks=False
    )
