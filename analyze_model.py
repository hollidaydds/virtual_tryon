import torch
import torch.nn as nn
import numpy as np

def analyze_pretrained_weights(model_path):
    """Analyze the pre-trained model weights in detail"""
    print(f"\nAnalyzing pre-trained weights from: {model_path}")
    
    # Load state dict
    state_dict = torch.load(model_path)
    
    # Group parameters by component
    components = {}
    for key, value in state_dict.items():
        # Split key into component parts
        parts = key.split('.')
        component = parts[0]
        if component not in components:
            components[component] = []
        components[component].append((key, value))
    
    # Analyze each component
    for component, params in components.items():
        print(f"\n=== {component} Component Analysis ===")
        
        # Get layer statistics
        for name, param in params:
            print(f"\nLayer: {name}")
            print(f"Shape: {param.shape}")
            print(f"Data type: {param.dtype}")
            print(f"Number of parameters: {param.numel()}")
            
            # Get numerical statistics
            if param.dim() > 0:  # Skip 0-dim tensors
                print(f"Mean: {param.mean().item():.6f}")
                print(f"Std: {param.std().item():.6f}")
                print(f"Min: {param.min().item():.6f}")
                print(f"Max: {param.max().item():.6f}")
            
            # Special analysis for specific layers
            if 'regression' in name:
                print("\nDetailed regression layer analysis:")
                print(f"Input features: {param.shape[1] if len(param.shape) > 1 else 'N/A'}")
                print(f"Output features: {param.shape[0] if len(param.shape) > 0 else 'N/A'}")
            
            if 'correlation' in name:
                print("\nDetailed correlation layer analysis:")
                if len(param.shape) == 4:
                    print(f"Input channels: {param.shape[1]}")
                    print(f"Spatial dimensions: {param.shape[2]}x{param.shape[3]}")

def analyze_layer_connectivity():
    """Analyze the expected connectivity between layers"""
    print("\n=== Layer Connectivity Analysis ===")
    
    # Feature extraction output size (from debug prints)
    feat_h, feat_w = 16, 12
    feat_c = 512
    print(f"\nFeature extraction output size: {feat_c}x{feat_h}x{feat_w}")
    
    # Correlation tensor size
    corr_size = feat_h * feat_w
    print(f"Expected correlation size: {corr_size}x{corr_size}")
    
    # Regression input size
    reg_in = corr_size * corr_size
    print(f"Expected regression input size: {reg_in}")
    
    # Verify shapes match
    print("\nShape verification:")
    print(f"Correlation output elements: {corr_size * corr_size} = {reg_in}")
    print(f"This should match regression input channels: 49152")

if __name__ == "__main__":
    model_path = "models/gmm_final.pth"
    analyze_pretrained_weights(model_path)
    analyze_layer_connectivity()
