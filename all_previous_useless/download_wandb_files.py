import wandb, os
ART = "yihengzeng-university-college-london-ucl-/PoseidonFinetune_NS-PwC_L_2048/poseidon_finetune_NS-PwC_L_2048_ckpt368800:v0"
OUT = "models/NS-PwC"
os.makedirs(OUT, exist_ok=True)
api = wandb.Api(timeout=300)  # increase HTTP timeout
art = api.artifact(ART, type="model")
for name in ["pytorch_model.bin", "optimizer.pt"]:
    print(f"Downloading {name}...")
    art.get_path(name).download(root=OUT)  # resumes and writes straight to OUT
print("Done.")
