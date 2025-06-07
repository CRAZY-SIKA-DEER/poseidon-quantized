import torch
import os
import argparse
from scOT.model import ScOT

def load_model(model_size):
    print("\n[INFO] Loading model...")
    model_name = f"camlab-ethz/Poseidon-{model_size}"
    print(f"[INFO] Loading from {model_name}")
    model = ScOT.from_pretrained(
        model_name,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )
    model.eval()
    print("[INFO] Model loaded and set to eval mode.")
    return model

def quantize_model_dynamic(model):
    print("\n[INFO] Starting dynamic quantization setup...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("[INFO] Dynamic quantization done.")
    return quantized_model

def get_file_size(filepath):
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='T', help="Model size: T (Tiny), B (Base), or L (Large)")
    args = parser.parse_args()

    model_size = args.model_size.upper()  # e.g., T, B, L

    print("[INFO] Starting dynamic quantization script...")

    model = load_model(model_size)
    orig_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Original model parameters: {orig_param_count}")

    # Save original model state_dict
    orig_model_file = f"poseidon_{model_size.lower()}_original.pt"
    torch.save(model.state_dict(), orig_model_file)
    print(f"[INFO] Original model saved as '{orig_model_file}'.")

    quantized_model = quantize_model_dynamic(model)
    quant_param_count = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)
    print(f"Quantized model parameters: {quant_param_count}")

    quant_model_file = f"quantized_poseidon_{model_size.lower()}_dynamic.pt"
    torch.save(quantized_model.state_dict(), quant_model_file)
    print(f"[INFO] Quantized model saved as '{quant_model_file}'.")

    # Show file sizes
    orig_model_size = get_file_size(orig_model_file)
    quant_model_size = get_file_size(quant_model_file)

    print(f"\n[SUMMARY]")
    print(f"Original model size: {orig_model_size:.2f} MB")
    print(f"Quantized model size: {quant_model_size:.2f} MB")
    print(f"Compression ratio: {orig_model_size / quant_model_size:.2f}× smaller!")