import torch
import torch.nn as nn
import time
from typing import Dict, List
from contextlib import contextmanager
import numpy as np
from collections import defaultdict

@contextmanager
def timer(timing_dict: Dict[str, List[float]], name: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    timing_dict[name].append((end - start) * 1000)

class FragmentedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_fragments, compressed_features, bias=False):
        super().__init__()
        self.timings = defaultdict(list)
        
        assert in_features % num_fragments == 0, "in_features must be divisible by num_fragments"
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_fragments = num_fragments
        self.compressed_features = compressed_features
        self.fragment_size = in_features // num_fragments
        self.features_per_fragment = self.fragment_size
        self.bias = bias
        
        with timer(self.timings, "0. Parameter Initialization"):
            # Selector network
            self.selector_weights = nn.Parameter(
                torch.randn(num_fragments, self.fragment_size) / np.sqrt(self.fragment_size)
            )
            
            # Expert weights for fragments
            self.expert_weights = nn.Parameter(
                torch.randn(num_fragments, self.fragment_size, out_features) / 
                np.sqrt(self.fragment_size)
            )
            
            # Simplified compression network
            self.compressor = nn.Linear(in_features, compressed_features, bias=False)
            self.compressed_net = nn.Linear(compressed_features, out_features, bias=False)
            
            if bias:
                self.expert_bias = nn.Parameter(torch.zeros(out_features))
                self.compressed_bias = nn.Parameter(torch.zeros(out_features))
            
            # Create and register fragment mapping buffer
            fragment_indices = torch.arange(num_fragments)
            fragment_mapping = fragment_indices.repeat_interleave(self.features_per_fragment)
            self.register_buffer('fragment_mapping', fragment_mapping)

    def create_fragment_mask(self, selector_probs, selected_idx=None):
        """Create mask for the selected fragment(s) mapped to full feature dimension"""
        batch_size = selector_probs.size(0)
        if selected_idx is not None:
            # Inference mode - hard selection
            # Create base mask of ones
            mask = torch.ones(batch_size, self.in_features, device=selector_probs.device)
            # Map selected indices to feature space
            selected_features = (self.fragment_mapping == selected_idx.unsqueeze(-1))
            # Zero out selected features
            mask[selected_features] = 0
            return mask
        else:
            # Training mode - soft selection
            # Map fragment probabilities to feature space
            expanded_probs = selector_probs[:, self.fragment_mapping]
            return 1 - expanded_probs

    def forward(self, x):
        batch_size = x.size(0)
        
        with timer(self.timings, "1. Input Reshaping"):
            x_fragments = x.view(batch_size, self.num_fragments, -1)
        
        with timer(self.timings, "2. Selector Computation"):
            selector_scores = torch.einsum('bfi,fi->bf', x_fragments, self.selector_weights)
            selector_probs = torch.softmax(selector_scores, dim=1)
        
        if self.training:
            with timer(self.timings, "3a. Expert Computation (Training)"):
                expert_outputs = torch.einsum('bfi,fio,bf->bo', 
                    x_fragments, self.expert_weights, selector_probs)
            
            with timer(self.timings, "4. Compression and Small Network"):
                # Create and apply mask
                mask = self.create_fragment_mask(selector_probs)
                masked_input = x * mask.to(x.dtype)
                
                # Process through simplified compression network
                compressed = self.compressor(masked_input)
                compressed_outputs = self.compressed_net(compressed)
        
        else:
            with timer(self.timings, "3a. Expert Selection (Inference)"):
                selected_idx = torch.argmax(selector_probs, dim=1)
                batch_idx = torch.arange(batch_size, device=x.device)
            
            with timer(self.timings, "3b. Expert Computation (Inference)"):
                expert_outputs = torch.einsum('bf,bfo->bo',
                    x_fragments[batch_idx, selected_idx],
                    self.expert_weights[selected_idx])
            
            with timer(self.timings, "4. Compression and Small Network"):
                # Create and apply mask
                mask = self.create_fragment_mask(selector_probs, selected_idx)
                masked_input = x * mask.to(x.dtype)
                
                # Process through simplified compression network
                compressed = self.compressor(masked_input)
                compressed_outputs = self.compressed_net(compressed)
        
        with timer(self.timings, "5. Output Combination"):
            output = expert_outputs + compressed_outputs
            if self.bias:
                output = output + self.expert_bias + self.compressed_bias
        
        return output

def test_optimized_fragmented_linear():
    """Test the OptimizedFragmentedLinear implementation with various input configurations"""
    test_configs = [
        (32, 512, 256, 8, 128),  # Standard case
        (1, 512, 256, 8, 128),   # Single batch
        (64, 1024, 512, 16, 256) # Larger dimensions
    ]
    
    for batch_size, in_features, out_features, num_fragments, compressed_features in test_configs:
        print(f"\nTesting configuration: batch={batch_size}, in={in_features}, out={out_features}, fragments={num_fragments}")
        
        layer = FragmentedLinear(in_features, out_features, num_fragments, compressed_features)
        x = torch.randn(batch_size, in_features)
        
        # Test training mode
        out_train = layer(x)
        assert out_train.shape == (batch_size, out_features)
        assert torch.isfinite(out_train).all()
        
        # Verify mask creation
        with torch.no_grad():
            selector_scores = torch.einsum('bfi,fi->bf', 
                x.view(batch_size, num_fragments, -1), 
                layer.selector_weights)
            selector_probs = torch.softmax(selector_scores, dim=1)
            mask = layer.create_fragment_mask(selector_probs)
            assert mask.shape == (batch_size, in_features)
            assert torch.all((mask >= 0) & (mask <= 1))
        
        # Test inference mode
        layer.eval()
        with torch.no_grad():
            out_eval = layer(x)
            assert out_eval.shape == (batch_size, out_features)
            assert torch.isfinite(out_eval).all()
    
    print("All tests passed!")

if __name__ == "__main__":
    test_optimized_fragmented_linear()