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
        
        # Simplified compression network - keep original input size
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
            mask = torch.ones(in_features, dtype=torch.bool)
            mask[fragment_mapping == i] = False
            indices_list.append(mask)
        self.register_buffer('fragment_masks', torch.stack(indices_list))

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
            expanded_probs = selector_probs[:, self.fragment_mapping]
            mask = 1 - expanded_probs
            masked_input = x * mask.to(x.dtype)
        else:
            # Inference mode - hard selection
            selected_idx = torch.argmax(selector_probs, dim=1)
            
            # Use index_select for potentially more efficient memory access
            expert_weights_selected = torch.index_select(self.expert_weights, 0, selected_idx)
            
            # Get corresponding input fragments
            batch_idx = torch.arange(batch_size, device=x.device)
            selected_fragments = x_fragments[batch_idx, selected_idx]
            
            # Matrix multiply
            expert_outputs = torch.bmm(
                selected_fragments.unsqueeze(1),
                expert_weights_selected
            ).squeeze(1)
            
            # Create masked input using precomputed masks
            mask = self.fragment_masks[selected_idx]
            masked_input = x.clone()
            masked_input[~mask] = 0
        
        # Process through compression network
        compressed = self.compressor(masked_input)
        compressed_outputs = self.compressed_net(compressed)
        
        # Combine outputs
        output = expert_outputs + compressed_outputs
        if self.bias:
            output = output + self.expert_bias + self.compressed_bias
        
        return output