import argparse, torch, os
from torch import nn
from torch.utils.data import DataLoader
from torch.quantization import (
    get_default_qat_qconfig,
    prepare_qat,
    convert,
    disable_observer,
    enable_observer,
)
from scOT.model import ScOT                                # Poseidon backbone
from scOT.problems.base import get_dataset


MODEL_PATH   = "checkpoints/finetune_NS-PwC_L_2048/PoseidonFinetune_NS-PwC_L/finetune_NS-PwC_run_L_2048/checkpoint-368800"
DATA_PATH    = "datasets/NS-PwC"
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"


from scOT.model import ScOT
model = ScOT.from_pretrained(MODEL_PATH)
names = [name for name, _ in model.named_modules()]

with open("poseidon_module_names.txt", "w") as f:
    f.write("\n".join(names))

print(f"Saved {len(names)} module names to poseidon_module_names.txt")
