import numpy as np
from collections.abc import Sequence
from typing import List

import torch
from .provenance import Provenance, Tag, TagBatch

class DAMP(Provenance):
    def zero(self, batch_shape, device="cpu"):
        return 0.0

    def one(self, batch_shape, device="cpu"):
        return 1.0
    
    def zeros(self, shape, device="cpu") -> TagBatch:
        return torch.zeros(shape, device=device)
    
    def add(self, a: Tag, b: Tag):
        return a + b
    
    def mul(self, a: Tag, b: Tag):
        return a * b
    
    def neg(self, a: Tag):
        return 1 - a

    def add_batch(self, a: TagBatch, b: TagBatch = None) -> TagBatch:
        if b is not None:
            assert a.shape == b.shape, "a and b must be of the same shape"
            return (a + b).clamp(min=0.0, max=1.0)
        
        elif isinstance(a, Sequence):
            return sum(a).clamp(min=0.0, max=1.0)
        else:
            assert isinstance(a, torch.Tensor), "Expected a tensor"
            return torch.sum(a, dim=-1).clamp(min=0.0, max=1.0)
        
    def mul_batch(self, a: TagBatch, b: TagBatch = None) -> TagBatch:
        if b is not None:
            assert a.shape == b.shape, "a and b must be of the same shape"
            return a * b
        else:
            if isinstance(a, torch.Tensor):
                return torch.prod(a, dim=-1)
            else:
                assert isinstance(a, Sequence), "Expected a sequence"
                return torch.prod(torch.stack(a), dim=0)
            
    def neg_batch(self, a: TagBatch) -> TagBatch:
        return 1 - a
        
    def reduce_symbols(self, prod: TagBatch, results: np.ndarray):
        # If results are not object arrays, just use np.unique directly
        if results.dtype != np.object_:
            symbols, idx = np.unique(results, return_inverse=True)
        else:
            sym = dict()
            symbols = []
            idx = []
            i = 0
            for r in results:
                if r not in sym:
                    sym[r] = i
                    i += 1
                    symbols.append(r)
                idx.append(sym[r])

        # Convert idx to a tensor
        idx_tensor = torch.tensor(idx, device=prod.device, dtype=torch.long)

        # Initialize  final probability tensor
        # prod.shape is (batch_size, num_results)
        batch_size, _ = prod.shape
        num_symbols = len(symbols)
        final_probs = torch.zeros(batch_size, num_symbols, device=prod.device)

        # Scatter-add probabilities from prod into final_probs based on idx_tensor
        # idx_tensor is shape [num_results], need to match batch_size rows
        # Unsqueeze to (1, num_results) and expand for batch_size rows
        final_probs.scatter_add_(1, idx_tensor.unsqueeze(0).expand(batch_size, -1), prod)

        # Clamp probabilities and return
        return final_probs.clamp(min=0.0, max=1.0), symbols
    
    def cartesian_prod(self, a: TagBatch, b: TagBatch) -> TagBatch:
        b_size_a, num_a = a.shape
        b_size_b, num_b = b.shape
        b_size = max(b_size_a, b_size_b)
        if b_size_a < b_size:
            a = a.expand(b_size, num_a)
        elif b_size_b < b_size:
            b = b.expand(b_size, num_b)
        return torch.bmm(a.view(-1, num_a, 1), b.view(-1, 1, num_b)).view(b_size, -1)
    
    
    #temporary solution; could be optimized
    def cartesian_prod_multi(self, tags_list: List[torch.Tensor]) -> torch.Tensor:
        # tags_list is a list of TagBatch, each shape: (batch_size, num_symbols_i)
        if len(tags_list) == 1:
            return tags_list[0]

        result = tags_list[0]
        for i in range(1, len(tags_list)):
            result = self.cartesian_prod(result, tags_list[i])
        return result