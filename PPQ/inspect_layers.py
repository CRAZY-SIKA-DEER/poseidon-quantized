import sys
import os
import torch
import torch.nn as nn
import json
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scOT.model import ScOT


# ========================
# Model Loading (your existing code)
# ========================
def load_poseidon_model(model_path: str, device: str = "cuda"):
    """
    Load the Poseidon ScOT model for quantization.
    
    Args:
        model_path: Path to pretrained model
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        model: Loaded model in eval mode
        device: Torch device object
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = ScOT.from_pretrained(model_path).to(device)
    model.eval()
    torch.set_float32_matmul_precision("high")
    
    print(f"Model loaded on device: {device}")
    print(f"Model type: {type(model)}")
    
    return model, device


# ========================
# Step 1: Inspect Quantizable Layers
# ========================
def inspect_quantizable_layers(model, exclude_patterns=None):
    """
    Traverse model and identify all Conv2d, ConvTranspose2d, and Linear layers.
    
    Args:
        model: PyTorch model
        exclude_patterns: List of substring patterns to exclude from layer names
                         (e.g., ['norm', 'time'] to exclude normalization/time layers)
    
    Returns:
        layer_info: dict mapping layer names to layer specifications (JSON-serializable)
    """
    if exclude_patterns is None:
        exclude_patterns = []
    
    layer_info = {}
    
    for name, module in model.named_modules():
        # Check if this is a quantizable layer type
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            # Apply exclusion filter
            should_exclude = any(pattern.lower() in name.lower() for pattern in exclude_patterns)
            if should_exclude:
                continue
            
            # Collect layer information (JSON-serializable only)
            info = {
                'type': type(module).__name__,
            }
            
            # Get shape information based on layer type
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                info['in_channels'] = module.in_channels
                info['out_channels'] = module.out_channels
                # Convert tuple to list for JSON serialization
                info['kernel_size'] = list(module.kernel_size) if isinstance(module.kernel_size, tuple) else module.kernel_size
                info['stride'] = list(module.stride) if isinstance(module.stride, tuple) else module.stride
                info['padding'] = list(module.padding) if isinstance(module.padding, tuple) else module.padding
                if hasattr(module, 'groups'):
                    info['groups'] = module.groups
            
            elif isinstance(module, nn.Linear):
                info['in_features'] = module.in_features
                info['out_features'] = module.out_features
            
            # Calculate parameter count
            info['num_parameters'] = sum(p.numel() for p in module.parameters())
            
            layer_info[name] = info
    
    return layer_info


# ========================
# Step 2: Print Architecture Summary
# ========================
def print_architecture_summary(layer_info):
    """
    Print a readable summary of the architecture hierarchy.
    
    Args:
        layer_info: Dictionary from inspect_quantizable_layers
    """
    print("\n" + "="*80)
    print("QUANTIZABLE LAYERS SUMMARY")
    print("="*80)
    
    # Group layers by type
    by_type = defaultdict(list)
    for name, info in layer_info.items():
        by_type[info['type']].append(name)
    
    # Count total parameters
    total_params = sum(info['num_parameters'] for info in layer_info.values())
    
    print(f"\nTotal quantizable layers: {len(layer_info)}")
    print(f"Total parameters in quantizable layers: {total_params:,}")
    print(f"\nBreakdown by layer type:")
    for layer_type, names in sorted(by_type.items()):
        print(f"  {layer_type}: {len(names)} layers")
    
    # Group by hierarchy
    print("\n" + "-"*80)
    print("LAYER DISTRIBUTION BY COMPONENT")
    print("-"*80)
    
    # Organize by major components
    components = {
        'Embeddings': [],
        'Encoder': [],
        'Residual Blocks': [],
        'Decoder': [],
        'Patch Recovery': [],
        'Other': []
    }
    
    for name, info in sorted(layer_info.items()):
        if 'embedding' in name.lower():
            components['Embeddings'].append(name)
        elif 'encoder' in name.lower():
            components['Encoder'].append(name)
        elif 'residual' in name.lower():
            components['Residual Blocks'].append(name)
        elif 'decoder' in name.lower():
            components['Decoder'].append(name)
        elif 'patch_recovery' in name.lower() or 'recovery' in name.lower():
            components['Patch Recovery'].append(name)
        else:
            components['Other'].append(name)
    
    # Print each component
    for component_name, layer_names in components.items():
        if not layer_names:
            continue
        print(f"\n  [{component_name}]: {len(layer_names)} layers")
    
    print("\n" + "="*80 + "\n")


# ========================
# Step 3: Select Layers for Quantization
# ========================
def select_layers_for_quantization(layer_info, exclude_additional=None):
    """
    Select which layers to quantize based on additional filtering criteria.
    
    Args:
        layer_info: Dictionary from inspect_quantizable_layers
        exclude_additional: Additional list of exact layer names to exclude
    
    Returns:
        quantize_layers: List of layer names to quantize
    """
    if exclude_additional is None:
        exclude_additional = []
    
    quantize_layers = []
    
    for name in layer_info.keys():
        if name not in exclude_additional:
            quantize_layers.append(name)
    
    print(f"\nSelected {len(quantize_layers)} layers for quantization")
    if exclude_additional:
        print(f"Excluded {len(exclude_additional)} additional layers")
    
    return quantize_layers


# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    # Configuration
    model_path = "models/NS-PwC"
    device = "cuda"
    
    # Load model
    print("Loading Poseidon model...")
    model, device = load_poseidon_model(model_path, device)
    
    # Inspect layers
    print("\nInspecting model architecture...")
    
    # Option 1: Inspect all Conv2d/Linear layers
    layer_info_all = inspect_quantizable_layers(model, exclude_patterns=[])
    
    # Option 2: Exclude time-conditional normalization layers
    layer_info_filtered = inspect_quantizable_layers(
        model, 
        exclude_patterns=['norm', 'time']  # Exclude layers with 'norm' or 'time' in name
    )
    
    # Print summary (limited output)
    print("\n" + "="*80)
    print("OPTION 1: ALL QUANTIZABLE LAYERS (including norm/time)")
    print_architecture_summary(layer_info_all)
    
    print("\n" + "="*80)
    print("OPTION 2: FILTERED (excluding norm/time layers)")
    print_architecture_summary(layer_info_filtered)
    
    # Select layers for quantization
    quantize_layers = select_layers_for_quantization(layer_info_filtered)
    
    # ========================
    # Save to JSON (human-readable, can view in any text editor)
    # ========================
    print("\nSaving detailed layer info to JSON files...")
    
    # Save all layers
    with open('layer_info_all.json', 'w') as f:
        json.dump(layer_info_all, f, indent=2)
    print("  ✓ Saved 'layer_info_all.json'")
    
    # Save filtered layers
    with open('layer_info_filtered.json', 'w') as f:
        json.dump(layer_info_filtered, f, indent=2)
    print("  ✓ Saved 'layer_info_filtered.json'")
    
    # Save selected layer names list
    with open('quantize_layers.json', 'w') as f:
        json.dump(quantize_layers, f, indent=2)
    print("  ✓ Saved 'quantize_layers.json'")
    
    # ========================
    # Save lightweight .pt file (just layer names for later use)
    # ========================
    print("\nSaving layer names to lightweight .pt file...")
    torch.save({
        'quantize_layers': quantize_layers,
        'layer_count': len(quantize_layers)
    }, 'quantize_layers.pt')
    print("  ✓ Saved 'quantize_layers.pt' (lightweight, just names)")
    
    print("\n" + "="*80)
    print("DONE! Files created:")
    print("  - layer_info_all.json (full details, all layers)")
    print("  - layer_info_filtered.json (full details, filtered)")
    print("  - quantize_layers.json (list of layer names to quantize)")
    print("  - quantize_layers.pt (lightweight, for code use)")
    print("\nYou can open the JSON files in any text editor to see all layers!")
    print("="*80 + "\n")
