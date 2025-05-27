from dolphin import Distribution
from collections.abc import Sequence

import torch

from dolphin.provenances.provenance import Provenance

class DMMP(Provenance):
    def add(self, a, b: torch.Tensor = None) -> torch.Tensor:
        if b is not None:
            res = torch.max(a, b)
            return res
        
        assert isinstance(a, Sequence), "Expected a sequence"
        return torch.max(torch.stack(a), dim=0).values

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if b is not None:
            assert a.dim() == b.dim(), "Expected tensors to have the same dimension"
            if a.dim() >= 3:
                pass
            else:
                return torch.min(a, b)
        else:
            assert isinstance(a, Sequence), "Expected a sequence"
            return torch.min(torch.stack(a), dim=0).values