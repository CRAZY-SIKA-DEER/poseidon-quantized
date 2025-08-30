import torch
import os
import argparse
from scOT.model import ScOT

def load_model(model_size, state_dict_path=None):
    print("\n[INFO] Loading model architecture...")
    model_name = f"camlab-ethz/Poseidon-{model_size}"
    model = ScOT.from_pretrained(model_name, trust_remote_code=True)
    print("[INFO] Model architecture loaded.")

    if state_dict_path is not None:
        print(f"[INFO] Loading weights from '{state_dict_path}'...")
        checkpoint = torch.load(state_dict_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print("[INFO] Weights loaded successfully.")
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
    parser.add_argument('--state_dict', type=str, default=None, help="Path to a pretrained .pt file to load (optional)")
    args = parser.parse_args()

    model_size = args.model_size.upper()  # 'T', 'B', or 'L'
    state_dict_path = args.state_dict  # This can be None

    print("[INFO] Starting dynamic quantization script...")

    model = load_model(model_size, state_dict_path)

    orig_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Original model parameters: {orig_param_count}")

    if state_dict_path is None:
        orig_model_file = f"poseidon_{model_size.lower()}_original.pt"
        torch.save(model.state_dict(), orig_model_file)
        print(f"[INFO] Original model saved as '{orig_model_file}'.")
    else:
        orig_model_file = state_dict_path

    quantized_model = quantize_model_dynamic(model)

    quant_param_count = sum(p.numel() for p in quantized_model.parameters() if p.requires_grad)
    print(f"Quantized model parameters: {quant_param_count}")

    # ⚡⚡ NEW PART: Save quantized model in Hugging Face format ⚡⚡
    save_dir = f"quantized_poseidon_{model_size.lower()}_dynamic"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    quant_model_file = f"quantized_poseidon_{model_size.lower()}_dynamic.pth"
    torch.save(quantized_model.state_dict(), quant_model_file)   # <-- save only weights
    print(f"[INFO] Quantized model state_dict saved as '{quant_model_file}'.")

    # Show file sizes (for original model if saved as .pt, quantized as directory size)
    orig_model_size = get_file_size(orig_model_file)

    # ⚠️ get directory size for the new saved quantized model
    quant_model_size = get_file_size(quant_model_file)

    print(f"\n[SUMMARY]")
    print(f"Original model size: {orig_model_size:.2f} MB")
    print(f"Quantized model size: {quant_model_size:.2f} MB")
    print(f"Compression ratio: {orig_model_size / quant_model_size:.2f}× smaller!")