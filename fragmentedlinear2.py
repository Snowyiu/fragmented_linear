import torch
import torch.nn as nn
import numpy as np



class FragmentedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_fragments, compressed_features, bias=False):
        super().__init__()
        
        assert in_features % num_fragments == 0, "in_features must be divisible by num_fragments"
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_fragments = num_fragments
        self.compressed_features = compressed_features
        self.fragment_size = in_features // num_fragments
        self.features_per_fragment = self.fragment_size
        self.bias = bias
        
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

        # Precompute indices for each fragment
        indices_list = []
        for i in range(num_fragments):
            indices_list.append((fragment_mapping != i).nonzero().squeeze(-1))
        self.register_buffer('fragment_indices', torch.stack(indices_list))


    def create_fragment_mask(self, selector_probs, selected_idx=None):
        """Create mask for the selected fragment(s) mapped to full feature dimension"""
        batch_size = selector_probs.size(0)
        if selected_idx is not None:
            # Inference mode - hard selection
            mask = torch.ones(batch_size, self.in_features, device=selector_probs.device)
            selected_features = (self.fragment_mapping == selected_idx.unsqueeze(-1))
            mask[selected_features] = 0
            return mask
        else:
            # Training mode - soft selection
            expanded_probs = selector_probs[:, self.fragment_mapping]
            return 1 - expanded_probs

    def forward(self, x):
        batch_size = x.size(0)
        x_fragments = x.view(batch_size, self.num_fragments, -1)
        
        # Compute selector probabilities
        selector_scores = torch.einsum('bfi,fi->bf', x_fragments, self.selector_weights)
        selector_probs = torch.softmax(selector_scores, dim=1)
        
        if self.training:
            # Training mode - soft selection
            expert_outputs = torch.einsum('bfi,fio,bf->bo', 
                x_fragments, self.expert_weights, selector_probs)
            
            # Create and apply mask
            mask = self.create_fragment_mask(selector_probs)
            masked_input = x * mask.to(x.dtype)
            
        else:
            # Inference mode - hard selection
            selected_idx = torch.argmax(selector_probs, dim=1)
            batch_idx = torch.arange(batch_size, device=x.device)
            
            expert_outputs = torch.einsum('bf,bfo->bo',
                x_fragments[batch_idx, selected_idx],
                self.expert_weights[selected_idx])
            
            # Create and apply mask
            mask = self.create_fragment_mask(selector_probs, selected_idx)
            masked_input = x * mask.to(x.dtype)
        
        # Process through compression network
        compressed = self.compressor(masked_input)
        compressed_outputs = self.compressed_net(compressed)
        
        # Combine outputs
        output = expert_outputs + compressed_outputs
        if self.bias:
            output = output + self.expert_bias + self.compressed_bias
        
        return output