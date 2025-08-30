import torch
from torch.ao.quantization import quantize, get_default_qconfig
from scOT.model import ScOT
from scOT.problems.base import get_dataset
from torch.utils.data import DataLoader

MODEL_PATH   = "checkpoints/finetune_NS-PwC_L_2048/PoseidonFinetune_NS-PwC_L/finetune_NS-PwC_run_L_2048/checkpoint-368800"
DATA_PATH    = "datasets/NS-PwC"
DATASET_NAME = "fluids.incompressible.PiecewiseConstants"
#  Monkey‑patch fuse_model into ScOT
def apply_fuse(self):
    # iterate over your backbone’s blocks
    for name, block in self.base_model.named_children():
        if hasattr(block, 'conv1') and hasattr(block, 'bn1') and hasattr(block, 'relu1'):
            fuse_modules(block, ['conv1', 'bn1', 'relu1',
                                 'conv2', 'bn2', 'relu2'], inplace=True)

# inject it
ScOT.fuse_model = apply_fuse


# 1. Build & fuse
model_fp32 = ScOT.from_pretrained(MODEL_PATH).eval()
model_fp32.fuse_model()

# 2. Assign QConfig
model_fp32.qconfig = get_default_qconfig("qnnpack")

# 3. Build calibration loader
calib_ds = get_dataset(dataset=DATASET_NAME, which="train",
                       num_trajectories=256, data_path=DATA_PATH)
calib_loader = DataLoader(calib_ds, batch_size=16, shuffle=True)

# 4. Define calibration function
def calibrate(m):
    with torch.no_grad():
        for i,batch in enumerate(calib_loader):
            if i>=20: break
            m(pixel_values=batch["pixel_values"],
              time        =batch["time"],
              pixel_mask  =batch["pixel_mask"],
              labels      =batch["labels"])

# 5) Quantize in one shot
qmodel = quantize(
    model_fp32,
    run_fn=calibrate,
    run_args=(),            # <-- required
    inplace=False,
)

# 6. Save the entire module
torch.save(qmodel, "poseidon_int8_full.pth")


import torch
# 1. Load your quantized module
qmodel = torch.load("poseidon_int8_full.pth", weights_only=False)
qmodel.eval().to("cpu")

# 2. Inference
out = qmodel(pixel_values=xb, time=tm, pixel_mask=pm, labels=yb)
