#!/usr/bin/env python
import os
import torch
from scOT.model import ScOT, ScOTConfig
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

# ── USER EDIT ────────────────────────────────────────────────────────────────
FLOAT_MODEL_DIR = "checkpoints/finetune_NS-PwC_B/PoseidonFinetune_NS-PwC_B/finetune_NS-PwC_run_B"
QAT_MODEL_PATH  = "qat_int8_poseidon.pt"
# ────────────────────────────────────────────────────────────────────────────────

def get_ckpt_size(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / 1e6

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def load_float_model(model_dir: str):
    config = ScOTConfig.from_pretrained(model_dir)
    # this will look for pytorch_model.bin (or similar) under model_dir
    model = ScOT.from_pretrained(model_dir, config=config)
    return model

def load_qat_model(config_dir: str, state_path: str):
    # 1) build float-QAT graph
    cfg = ScOTConfig.from_pretrained(config_dir)
    m_q = ScOT(cfg).train()
    torch.backends.quantized.engine = "qnnpack"
    m_q.qconfig = get_default_qat_qconfig("qnnpack")
    m_q = prepare_qat(m_q, inplace=False)
    # 2) convert to quantized modules
    m_q = convert(m_q.eval(), inplace=False)
    # 3) load your int8 weights
    sd = torch.load(state_path, map_location="cpu")
    m_q.load_state_dict(sd)
    return m_q


def count_state_dict_elements(model: torch.nn.Module) -> int:
    total = 0
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            total += v.numel()
    return total


if __name__ == "__main__":
    '''
    1.	157,729,988 is just the total number of weight-values (i.e. parameters) in your float32 model.
    Every convolution, linear layer, embedding, etc., contributes a certain number of weights and biases, and if you add them all up you get ∼1.58×10⁸ “elements.”
	2.	Why the QAT model’s state_dict() looks like it has only a few thousand “parameters”:
	•	In a quantized model, the actual int8 weights live inside C++ “packed” buffers, not as ordinary torch.Tensors that show up in Python’s state_dict().
	•	What does appear in the Python checkpoint are only the tiny floating-point tensors for each layer’s scale and zero_point (and maybe biases). 
    Those add up to only a few thousand numbers, which is exactly what you saw.
	3.	What “elements” means here:
	•	When we say a model has N parameters or N elements, we mean “if I took every tensor in its state_dict() and summed up tensor.numel(), what do I get?”
	•	For the float model, every weight-tensor is in that dict, so you count ~1.58×10⁸ elements.
	•	For the quantized model, only the small scale/zero_point buffers are in the dict, so you count only ~8×10⁴ elements—even though 
    under the hood the full int8 weights still exist, they just aren’t exposed as Python tensors.

    Bottom line: your quantized checkpoint really does contain all the same layers and weights (now packed into int8 buffers), but PyTorch doesn’t show 
    those buffers in the Python state_dict(), so it looks like “fewer parameters” even though the model architecture is unchanged.
    '''
    # 1) on-disk sizes
    float_ckpt = os.path.join(FLOAT_MODEL_DIR, "pytorch_model.bin")
    if not os.path.exists(float_ckpt):
        # sometimes it’s named differently
        float_ckpt = os.path.join(FLOAT_MODEL_DIR, "model.bin")
    print(f"Float checkpoint: {float_ckpt}")
    print(f"  size = {get_ckpt_size(float_ckpt):.2f} MB")

    print(f"QAT checkpoint:   {QAT_MODEL_PATH}")
    print(f"  size = {get_ckpt_size(QAT_MODEL_PATH):.2f} MB\n")

    # 2) parameter counts
    print("Loading float model...")
    float_model = load_float_model(FLOAT_MODEL_DIR)
    print(f"  # parameters = {count_parameters(float_model):,}")

    print("Loading QAT-quantized model...")
    qat_model = load_qat_model(FLOAT_MODEL_DIR, QAT_MODEL_PATH)
    print(f"  # parameters = {count_parameters(qat_model):,}")

    print("Float total elements:", count_state_dict_elements(float_model))
    print("Quantized total elements:", count_state_dict_elements(qat_model))