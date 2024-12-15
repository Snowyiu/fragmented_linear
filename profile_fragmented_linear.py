import torch
import torch.nn as nn
import time
from typing import Dict, List
from contextlib import contextmanager
import numpy as np
from collections import defaultdict
from fragmentedlinear import FragmentedLinear

def profile_detailed(
    in_features: int,
    out_features: int,
    num_fragments: int,
    batch_size: int,
    num_warmup: int = 10,
    num_iterations: int = 5000,
    selection_threshold: float = 0.75,
) -> Dict[str, Dict[str, float]]:
    """Run detailed profiling of the optimized model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FragmentedLinear(
        in_features=in_features,
        out_features=out_features,
        num_fragments=num_fragments,
        compressed_features=in_features//16,
        bias=False
    ).to(device)
    
    x = torch.randn(batch_size, in_features, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        _ = model(x)
    
    # Training mode profiling
    print("\nProfiling training mode...")
    model.train()
    for _ in range(num_iterations):
        _ = model(x)
    
    # Inference mode profiling
    print("\nProfiling inference mode...")
    model.eval()
    for _ in range(num_iterations):
        _ = model(x)
    
    # Process timing results
    results = {}
    for mode in ["Training", "Inference"]:
        mode_results = {}
        for name, times in model.timings.items():
            # Filter timings by mode
            if ((mode == "Training" and "Training" in name) or 
                (mode == "Inference" and any(x in name for x in ["Fragment Selection", "Inference"])) or
                name.startswith("1.") or name.startswith("2.")):
                mode_results[name] = {
                    'mean': np.mean(times[-num_iterations:]),
                    'std': np.std(times[-num_iterations:]),
                    'min': np.min(times[-num_iterations:]),
                    'max': np.max(times[-num_iterations:])
                }
        results[mode] = mode_results
    
    return results

def print_profiling_results(results: Dict[str, Dict[str, float]]):
    """Pretty print the profiling results"""
    for mode, timings in results.items():
        print(f"\n{'-'*20} {mode} Mode {'-'*20}")
        
        total_time = sum(timing['mean'] for timing in timings.values())
        
        for name in sorted(timings.keys()):
            stats = timings[name]
            percentage = (stats['mean'] / total_time) * 100
            print(f"{name:25} {stats['mean']:8.3f}ms ±{stats['std']:6.3f} ({percentage:5.1f}%)")

if __name__ == "__main__":
    # Test configurations
    configs = [
        (4096, 4096, 8, 8),
        (4096, 4096, 16, 8),
        (4096, 4096, 32, 8),
        (4096, 4096, 64, 8),
        (4096, 4096, 128, 8),
        (4096, 4096, 256, 8),
    ]
    
    for in_feat, out_feat, num_frags, batch_size in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {in_feat}→{out_feat}, {num_frags} fragments, batch_size={batch_size}")
        print(f"{'='*80}")
        
        results = profile_detailed(
            in_features=in_feat,
            out_features=out_feat,
            num_fragments=num_frags,
            batch_size=batch_size
        )
        
        print_profiling_results(results)