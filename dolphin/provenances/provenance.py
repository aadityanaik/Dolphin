import torch
import itertools
from typing import List, NewType

from ..distribution import Distribution

Tag = NewType('Tag', torch.tensor)

# b x n tensor of tags, where b is batch size and n is # of symbols
TagBatch = NewType('TagBatch', torch.tensor)

class Provenance:
    k = 1

    def zero(self, batch_shape, device="cpu") -> Tag:
        raise NotImplementedError

    def one(self, batch_shape, device="cpu") -> Tag:
        raise NotImplementedError
    
    def zeros(self, shape, device="cpu") -> TagBatch:
        tags = torch.stack([self.zero(shape, device=device) for _ in range(shape[0] * shape[1])])
        return tags.view(shape)

    def add(self, a: Tag, b: Tag):
        raise NotImplementedError

    def mul(self, a: Tag, b: Tag):
        raise NotImplementedError

    def neg(self, a: Tag):
        raise NotImplementedError

    def _binop_fold(self, a: List[Tag], op):
        assert len(a) > 0, "a must be non-empty"

        acc = a[0]
        for x in a[1:]:
            acc = op(acc, x)

        return acc
    
    def add_fold(self, a: List[Tag]):
        return self._binop_fold(a, self.add)
    
    def mul_fold(self, a: List[Tag]):
        return self._binop_fold(a, self.mul)

    def _binop_batch(self, a: TagBatch, b: TagBatch, op) -> TagBatch:
        assert a.shape == b.shape, "a and b must be of the same shape"

        c = self.zeros(self, a.shape, device=a.device)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                c[i, j] = op(a[i, j], b[i, j])

        return c

    def add_batch(self, a: TagBatch, b: TagBatch) -> TagBatch:
        return self._binop_batch(a, b, self.add)

    def mul_batch(self, a: TagBatch, b: TagBatch) -> TagBatch:
        return self._binop_batch(a, b, self.mul)

    def _unop_batch(self, a: TagBatch, op) -> TagBatch:
        c = self.zeros(self, a.shape, device=a.device)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                c[i, j] = op(a[i, j])
        return c
    
    def neg_batch(self, a: TagBatch) -> TagBatch:
        return self._unop_batch(a, self.neg)
    
    def reduce_symbols(self, prod: TagBatch, results):
        sym_idxs = dict()
        symbols = []
        for i in range(len(results)):
            r = results[i]
            if r not in sym_idxs:
                sym_idxs[r] = [i]
                symbols.append(r)
            else:
                sym_idxs[r].append(i)
        
        new_shape = list(prod.shape)
        new_shape[1] = len(symbols)
        tags = self.zeros(new_shape, device=prod.device)

        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                tags[i, j] = self.add_fold([prod[i, k] for k in sym_idxs[symbols[j]]])
        
        return tags, symbols
    
    def cartesian_prod(self, a: TagBatch, b: TagBatch) -> TagBatch:
        m, num_a, num_b = a.shape[0], a.shape[1], b.shape[1]
        new_shape = list(a.shape)
        new_shape[1] = num_a * num_b

        c = self.zeros(new_shape, device=a.device)
        for i in range(m):
            res = list(self.mul(a[i, ia], b[i, ib]) for ia, ib in itertools.product(range(num_a), range(num_b)))
            c[i] = torch.stack(res)

        return c


    def probs_from_tags(self, tags: TagBatch) -> torch.Tensor:
        return tags

    def tags_from_probs(self, probs: torch.Tensor, disjunctions: List) -> TagBatch:
        return probs
    
    def combine_tag_sources_multi(self, distributions: List[Distribution]):
        """
        Combine the tag sources for an arbitrary list of distributions
        (k >= 1). This mirrors the code you provided, but is named
        `combine_tag_sources_multi`.

        Returns:
            tags_list: List[torch.Tensor]
                A list of the .tags for each Distribution in `distributions`
            combined_src: List[Distribution]
                The combined list of source distributions (duplicates removed)
        """
        # Extract tags and src from each distribution
        tags_list = [dist.tags for dist in distributions]

        src_list = []
        for dist in distributions:
            src_list.extend(dist.src)
        combined_src = list(set(src_list))  # Remove duplicates if needed

        return tags_list, combined_src


    def combine_tag_sources(self, dist_a: Distribution, dist_b: Distribution):
        """
        Combine the tag sources for exactly two distributions, dist_a and dist_b.
        This returns a 2-element list of tag tensors, plus a combined src list.
        
        Returns:
            tags_list: List[torch.Tensor]
                Exactly [dist_a.tags, dist_b.tags]
            combined_src: List[Distribution]
                The combined list of source distributions (duplicates removed)
        """
        tags_list = [dist_a.tags, dist_b.tags]

        # Merge the .src sets from each distribution
        # (or list them, then deduplicate).
        src_list = list(set(dist_a.src + dist_b.src))

        return tags_list, src_list
