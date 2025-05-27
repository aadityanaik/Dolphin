import torch
from collections import Counter
import math

from .provenance import Provenance, Tag, TagBatch
from ..distribution import Distribution


class DTKP_AM(Provenance):
    """
    Provenance: difftopkproofs-addmult (dtkp-am)

    Tag T is a k x |F| tensor, where:
        - k is the max number of proofs being tracked.
        - F is the space of input facts derived from the tag's source distributions.

    Each proof T[i] of T is a DNF clause whose literals are input facts from F.
    If T has n < k proofs, then ZERO in T[i] iff. i >= n (ZERO indicates the absence of a proof).

    For each fact f_j in F:
        - If pos(f_j) in T[i], then T[i][j] = P(f_j).
        - If neg(f_j) in T[i], then T[i][j] = -(1 - P(f_j)).
        - Else, T[i][j] = NULL.

    Probabilities are calculated using the naive addmult assumption:
        - All facts in F are mutually independent: P(T[i]) = prod_j |T[i][j]|
        - All proofs are disjoint: P(T) = sum_i P(T[i])

    In PyTorch, we represent NULL with any value > 1, and ZERO with any value < -1.
    """

    _NULL = 1000.0
    _ZERO = -1000.0

    """
    Proof absorption:
    When performing add or mul, most efficient to naively concatenate tensors of proofs.
        - This could result in tensors with duplicate proofs or proofs that are supsets of others.
        - Logical conjunction is absorptive, so the probability of these proofs should not be added.

    Because absorbing proofs is expensive and oftentimes unnecessary, we have three tunable settings:
        _absorb_proofs_pre_topk:    absorb proofs before taking the top-k proofs;
                                    yields more accurate tags at any given point in computation;
                                    incurs large runtime and space penalty for every add / mul: O(# tags * |F| * k^4)

        _absorb_proofs_post_topk:   absorb proofs after taking the top-k proofs;
                                    yields less accurate tags than pre-topk, as top-k could return fewer than k proofs;
                                    incurs medium runtime and space penalty for every add / mul: O(# tags * |F| * k^2)

        _absorb_proofs_addmult:     absorb proofs before calculating probabilities;
                                    this does not improve the accuracy of the tag representations,
                                    but does avoid overapproximating the final probabilities in some cases;
                                    incurs negligible runtime and space penalty
    """
    _absorb_proofs_pre_topk = False
    _absorb_proofs_post_topk = False
    _absorb_proofs_addmult = True

    _mul_proofs_top_sqrtk = False

    def zero(self, batch_shape, device="cpu") -> Tag:
        return torch.full(batch_shape[2:], self._ZERO, device=device)

    def one(self, batch_shape, device="cpu") -> Tag:
        ones = torch.full(batch_shape[2:], self._NULL, device=device)
        ones[1:] = self._ZERO
        return ones

    def zeros(self, shape, device="cpu") -> TagBatch:
        return torch.full(shape, self._ZERO, device=device)

    def __nulls(self, shape, device="cpu") -> TagBatch:
        return torch.full(shape, self._NULL, device=device)

    def __norm(self, tags: Tag | TagBatch):
        result = torch.where(tags < -1, 0, tags)
        result = torch.where(tags > 1, 1, result)
        return result.abs()

    def __union_proofs(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        conflict = (a != b) & (a.abs() <= 1) & (b.abs() <= 1)
        return torch.where(conflict, self._ZERO, torch.min(a, b))

    def __absorb_proofs(self, proofs: torch.Tensor) -> torch.Tensor:
        k = proofs.shape[-2]
        missing_facts = (proofs.abs() > 1).sum(dim=-1)
        ordinal = k * missing_facts + torch.arange(k, device=proofs.device)
        proofs_ord = torch.concat([proofs, ordinal.unsqueeze(-1)], dim=-1)

        repeat_shape = [1] * proofs.dim()
        repeat_shape[-2] = k
        a = proofs_ord.repeat_interleave(k, dim=-2)
        b = proofs_ord.repeat(repeat_shape)

        c = self.__union_proofs(a, b)
        is_supset = (a == c).all(dim=-1).view(list(proofs.shape[:-2]) + [k, k])
        mask_supset = is_supset.sum(dim=-1) > 1

        proofs[mask_supset] = self._ZERO
        return proofs

    def __topk_proofs(self, proofs: torch.Tensor, k=None, batch=False) -> torch.Tensor:
        if k is None:
            k = self.k
        if self._absorb_proofs_pre_topk:
            proofs = self.__absorb_proofs(proofs)
        probs = torch.prod(self.__norm(proofs), dim=-1)
        idxs = torch.topk(probs, k).indices
        if batch:
            idxs = (torch.arange(proofs.shape[0]).unsqueeze(-1), idxs)
        if self._absorb_proofs_post_topk:
            return self.__absorb_proofs(proofs[idxs])
        return proofs[idxs]

    def add(self, a: Tag, b: Tag) -> Tag:
        return self.__topk_proofs(torch.cat([a, b]))

    def mul(self, a: Tag, b: Tag) -> Tag:
        k = self.k
        if self._mul_proofs_top_sqrtk:
            k = int(math.ceil(math.sqrt(self.k)))
            a = self.__topk_proofs(a, k=k)
            b = self.__topk_proofs(b, k=k)

        a = a.repeat_interleave(k, dim=0)
        b = b.repeat(k, 1)
        return self.__topk_proofs(self.__union_proofs(a, b))

    """
    Negation of tag T -> neg(T):
        1) Individually negate each literal T[i][j] to get neg'(T). (note neg(NULL) = ZERO and vice versa)
           neg'(T) is now in CNF form by DeMorgan's law, and needs to be converted to DNF to get neg(T).

        2) The top-k DNF proofs can only be formed by the top-k literals of each CNF clause neg'(T)[i].
           Define tags[i] to be the tag corresponding to the disjunction of the top-k literals of neg'(T)[i].
           There are some cases requiring special consideration:
                2a) If the proof T[i] is absent (i.e. T[i] has ZERO), then neg'(T)[i] must yield tags[i] = {{}}.
                2b) When T[i] has fewer than k literals, tags[i] should be padded with absent proofs.

        3) Compute: neg(T) = prod_i tags[i]
    """

    def neg(self, a: Tag) -> Tag:
        k = min(self.k, a.shape[1])
        idxs = torch.topk(1 - self.__norm(a), k, dim=1).indices
        vals = torch.gather(a, 1, idxs)
        signs = torch.where(vals.signbit(), 1, -1)
        neg_vals = torch.where(vals.abs() <= 1, (1 - vals.abs()) * signs, -vals)

        # Case 2a of negation
        neg_vals[(neg_vals > 1).any(dim=1), 1:] = self._ZERO

        tags = self.__nulls((self.k, self.k, a.shape[1]), device=a.device)
        tags[
            torch.arange(self.k).unsqueeze(-1), torch.arange(k).expand(idxs.shape), idxs
        ] = neg_vals

        # Case 2b of negation
        mask = torch.arange(self.k, device=tags.device).expand(self.k, self.k) >= k
        tags[mask] = self._ZERO

        return self.mul_fold([tags[i] for i in range(self.k)])

    def add_batch(self, a: TagBatch, b: TagBatch) -> TagBatch:
        assert a.shape == b.shape, "a and b must be of the same shape"
        a_flat = a.view(-1, a.shape[2], a.shape[3])
        b_flat = b.view(-1, b.shape[2], b.shape[3])
        c = torch.cat([a_flat, b_flat], dim=1)
        return self.__topk_proofs(c, batch=True).view(a.shape)

    def mul_batch(self, a: TagBatch, b: TagBatch) -> TagBatch:
        assert a.shape == b.shape, "a and b must be of the same shape"
        a_flat = a.view(-1, a.shape[2], a.shape[3])
        b_flat = b.view(-1, b.shape[2], b.shape[3])

        k = self.k
        if self._mul_proofs_top_sqrtk:
            k = int(math.ceil(math.sqrt(self.k)))
            a_flat = self.__topk_proofs(a_flat, k=k, batch=True)
            b_flat = self.__topk_proofs(b_flat, k=k, batch=True)

        a_flat = a_flat.repeat_interleave(k, dim=1).view(-1, a.shape[3])
        b_flat = b_flat.repeat(1, k, 1).view(-1, b.shape[3])

        c = self.__union_proofs(a_flat, b_flat).view(
            a.shape[0] * a.shape[1], -1, a.shape[3]
        )
        return self.__topk_proofs(c, batch=True).view(a.shape)

    def neg_batch(self, a: TagBatch) -> TagBatch:
        k = min(self.k, a.shape[3])
        idxs = torch.topk(1 - self.__norm(a), k, dim=3).indices
        vals = torch.gather(a, 3, idxs)
        signs = torch.where(vals.signbit(), 1, -1)
        neg_vals = torch.where(vals.abs() <= 1, (1 - vals.abs()) * signs, -vals)

        # Case 2a of negation
        mask_neg = (neg_vals > 1).any(dim=3).unsqueeze(-1).expand(idxs.shape)
        mask_neg = mask_neg & (
            torch.arange(k, device=mask_neg.device).expand(idxs.shape) >= 1
        )
        neg_vals[mask_neg] = self._ZERO

        idxs0 = torch.arange(self.k).expand(idxs.shape[:3]).unsqueeze(-1)
        idxs1 = (
            torch.arange(a.shape[0])
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(idxs.shape)
        )
        idxs2 = torch.arange(a.shape[1]).unsqueeze(-1).unsqueeze(-1).expand(idxs.shape)
        idxs3 = torch.arange(k).expand(idxs.shape)

        tags = self.__nulls([self.k] + list(a.shape), device=a.device)
        tags[idxs0, idxs1, idxs2, idxs3, idxs] = neg_vals

        # Case 2b of negation
        mask = torch.arange(self.k, device=tags.device).expand(tags.shape[:4]) >= k
        tags[mask] = self._ZERO

        return self._binop_fold([tags[i] for i in range(self.k)], self.mul_batch)

    def reduce_symbols(self, prod: TagBatch, results):
        col0 = self.zeros(
            (prod.shape[0], 1, prod.shape[2], prod.shape[3]), device=prod.device
        )
        prod0 = torch.cat([prod, col0], dim=1)

        if len(results) == 0:
            return prod[:, []], results

        n, width = len(set(results)), Counter(results).most_common(1)[0][1]
        reduce_shape = (prod.shape[0], n, prod.shape[2], prod.shape[3])

        sym = dict()
        idx_mat = [[-1] * n for _ in range(width)]
        symbols = []
        i = 0
        for idx, r in enumerate(results):
            if r not in sym:
                sym[r] = [0, i]
                i += 1
                symbols.append(r)
            idx_mat[sym[r][0]][sym[r][1]] = idx
            sym[r][0] += 1
        
        catlist = prod0[:, idx_mat[:]].permute(0, 2, 1, 3, 4).flatten(0, 1).reshape(-1, width * reduce_shape[2], reduce_shape[3])
        tags = self.__topk_proofs(catlist, batch=True).view(reduce_shape)

        return tags, symbols

    def cartesian_prod(self, a: TagBatch, b: TagBatch) -> TagBatch:
        na, nb = a.shape[1], b.shape[1]
        a = a.repeat_interleave(nb, dim=1)
        b = b.repeat(1, na, 1, 1)
        return self.mul_batch(a, b)

    def probs_from_tags(self, tags: TagBatch) -> torch.Tensor:
        if self._absorb_proofs_addmult:
            tags = self.__absorb_proofs(tags)
        return self.__norm(tags).prod(-1).sum(-1).clamp(min=0.0, max=1.0)

    def tags_from_probs(self, probs: torch.Tensor, disjunctions) -> TagBatch:
        m, n = probs.shape
        tags = self.__nulls((m, n, self.k, len(disjunctions)), device=probs.device)

        idx_map = {}
        for i, arr in enumerate(disjunctions):
            for sym in arr:
                idx_map[sym] = i

        idxs = torch.arange(n).unsqueeze(1)
        fact_idxs = [[idx_map[sym]] for sym in range(n)]

        tags[:, idxs, 0, fact_idxs] = probs.view(m, n, 1)
        tags[:, idxs, 1:] = self._ZERO

        return tags

    def combine_tag_sources(self, dist_a: Distribution, dist_b: Distribution):
        a_ids = set(d.id for d in dist_a.src)
        ab_src = [d for d in dist_a.src]
        for d in dist_b.src:
            if d.id not in a_ids:
                ab_src.append(d)

        ab_idx_map = {}
        ab_facts = 0
        for dist in ab_src:
            dist_facts = dist.tags.shape[-1]
            ab_idx_map[dist] = list(range(ab_facts, ab_facts + dist_facts))
            ab_facts += dist_facts

        a_tags = self.__nulls(
            (dist_a.tags.shape[0], dist_a.tags.shape[1], self.k, ab_facts),
            device=dist_a.tags.device,
        )
        b_tags = self.__nulls(
            (dist_b.tags.shape[0], dist_b.tags.shape[1], self.k, ab_facts),
            device=dist_b.tags.device,
        )

        def map_to_ab_idx(src):
            idxs = []
            for dist in src:
                idxs.extend(ab_idx_map[dist])
            return idxs

        a_idxs, b_idxs = map_to_ab_idx(dist_a.src), map_to_ab_idx(dist_b.src)
        a_tags[:, :, :, a_idxs] = dist_a.tags
        b_tags[:, :, :, b_idxs] = dist_b.tags

        # print("AB SOURCE", len(ab_src))

        return [a_tags, b_tags], ab_src
