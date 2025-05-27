from __future__ import annotations
from collections import defaultdict
from typing import Any, List, Tuple, Dict, Callable, Union

import torch
import numpy as np

class Distribution:
    _p = None
    _k = None
    __input_instances = []

    @property
    def provenance(self):
        return type(self)._p
    
    @provenance.setter
    def provenance(self, value: Any) -> None:
        type(self)._p = value

    @property
    def k(self) -> Any:
        return type(self)._k
    
    @k.setter
    def k(self, value: Any) -> None:
        type(self)._k = value

    @staticmethod
    def copy(d: Distribution) -> Distribution:
        assert isinstance(d, Distribution), "Input must be of type Distribution"
        return Distribution(d.tags, d.symbols, dist_as_probs=False, src=d.src)

    @staticmethod
    def stack(distributions: List) -> Distribution:
        from .utils import symbolic_collate_fn
        return symbolic_collate_fn(distributions)
    
    @staticmethod
    def l_and(a: Distribution, b: Distribution) -> Distribution:
        assert isinstance(a, Distribution) and isinstance(b, Distribution), "All inputs must be of type Distribution"
        # assert a.type == np.bool_ and b.type == np.bool_, "All inputs must have the type `np.bool_`"

        return a.__compute_possibilities(b, lambda s1, s2 : s1 and s2)

    @staticmethod
    def l_or(a: Distribution, b: Distribution) -> Distribution:
        assert isinstance(a, Distribution) and isinstance(b, Distribution), "All inputs must be of type Distribution"
        # assert a.type == np.bool_ and b.type == np.bool_, "All inputs must have the type `np.bool_`"

        return a.__compute_possibilities(b, lambda s1, s2 : s1 or s2)

    @staticmethod
    def l_not(a: Distribution) -> Distribution:
        assert isinstance(a, Distribution), "All inputs must be of type Distribution"
        # assert a.type == np.bool_, "All inputs must have the type `np.bool_`"

        return a.__compute_possibilities(a, lambda s1, s2 : not s1)
        
    def __get_symbols_from_array(self, symbol_list):
        if isinstance(symbol_list, np.ndarray) and len(symbol_list.shape) == 1:
            # print("Straight copy")
            return symbol_list
        try:
            symbols = np.array(symbol_list)
        except:
            symbols = np.array(symbol_list, dtype=np.object_)
        if symbols.shape == ():
            symbols = np.array([symbols])
        elif len(symbols.shape) > 1:
            symbols = np.empty(len(symbol_list), dtype=object)
            symbols[:] = symbol_list
        return symbols

    def __init__(self, distribution: torch.Tensor, symbols, dist_as_probs = True, disjunctions = None, src = None) -> None:
        """
        distribution:
            torch.Tensor of shape (N, M)  where N, M is the shape of symbols
        dist_as_probs:
            True if we want to treat distribution tensor as probabilities
            False if we want to treat distribution tensor as tags
        """
        assert self.provenance is not None, "Provenance not set"
        
        self.symbols = self.__get_symbols_from_array(symbols)

        if dist_as_probs:
            assert distribution.dim() <= 2, "Distribution must be 1D or 2D"
            assert (distribution.dim() == 0 and len(self.symbols.shape) == 0) or distribution.shape[-1] == len(self.symbols), f"Length of symbols must match number of columns of the distribution: {distribution.shape[-1]}, {len(self.symbols)}"
           
            if distribution.dim() > 2:
                probs = distribution.view(distribution.shape[0], -1)
            elif distribution.dim() == 1:
                probs = distribution.view(1, -1)
            else:
                probs = distribution

            if disjunctions is None:
                disjunctions = [list(range(len(self.symbols)))]

            self.tags = self.provenance.tags_from_probs(probs, disjunctions=disjunctions)
        else:
            # TODO: dimensionality checks for tag distributions
            if distribution.dim() == 1:
                self.tags = distribution.view(1, -1)
            else:
                self.tags = distribution

        self.id = f'D_{len(Distribution.__input_instances)}'
        self.inverted = False

        if src:
            self.src = src
        else:
            self.src = [self]
            Distribution.__input_instances.append(self)
            

    def sample_top_k(self, k, categorical=False) -> Distribution:
        if k is None or k > len(self.symbols):
            return self
        
        p = self.get_probabilities()
        # Avoid all-zero distributions
        p = p + (p.sum(dim=1).view(p.shape[0], -1) == 0).float()
        if categorical:
            categ = torch.distributions.Categorical(p)
            indices = categ.sample((k, )).T
        else:
            topk = torch.topk(p, k)
            indices = topk.indices
        flattened_indices = indices.unique()
        
        new_symbols = self.symbols[flattened_indices.cpu()]
        
        #harcdoed edge case for if new_symbols is a singleton and the type of the symbol is tuple
        if isinstance(new_symbols, tuple) and not isinstance(new_symbols, np.ndarray):
            symbols = np.empty(1, dtype=object) 
            symbols[0] = new_symbols
            new_symbols = symbols
        if not isinstance(new_symbols, np.ndarray):
            new_symbols = np.array([new_symbols, ])
        if len(flattened_indices) != len(new_symbols):
            assert len(flattened_indices) == 1, f"Indices: {flattened_indices}, Symbols: {new_symbols}"
            new_symbols = [new_symbols]
        
        if self.tags.dim() == 1:
            new_distribution = self.tags[flattened_indices]
        else:
            mask = self.provenance.zeros(self.tags.shape, device=self.tags.device)
            mask[torch.arange(self.tags.shape[0]).unsqueeze(1), indices] = self.provenance.one(self.tags.shape, device=self.tags.device)
            new_distribution = self.provenance.mul_batch(self.tags, mask)[:, flattened_indices]
        d = Distribution(new_distribution, new_symbols, dist_as_probs=False, src=self.src)

        return d

    def __calculate_possibilities(self, dists: List[Distribution], function: Callable, conditional=False) -> Distribution:
        if not isinstance(dists, List):
            dists = [dists]
        all_dists = [self] + dists
        for dist in all_dists:
            if len(dist.symbols) == 0:
                return Distribution.copy(dist)
        all_dists = [dist.sample_top_k(dist.k) for dist in all_dists]

        tags_list, combined_src = self.provenance.combine_tag_sources(all_dists[0], all_dists[1])
        num_lists = [len(dist.symbols) for dist in all_dists]
        symbol_lists = [dist.symbols for dist in all_dists]
        index_combinations = np.indices(num_lists).reshape(len(num_lists), -1).T
    
        args = [symbol_lists[i][index_combinations[:, i]] for i in range(len(all_dists))]
        res_list = [function(*combination) for combination in zip(*args)]
        results = self.__get_symbols_from_array(res_list)

        final_tags = self.provenance.cartesian_prod(tags_list[0], tags_list[1])
        prod_distribution = Distribution(final_tags, results, dist_as_probs=False, src=combined_src)

        if conditional:
            prod_distribution = prod_distribution.drop_symbol(None)

        final_tags, symbols = self.provenance.reduce_symbols(prod_distribution.tags, prod_distribution.symbols)
        d = Distribution(final_tags, symbols, dist_as_probs=False, src=combined_src)

        return d

    def __compute_possibilities(self, dist_b: Union[Distribution|np.ndarray|Any], function: Callable, conditional = False) -> Distribution:
        assert self.provenance is not None, "Provenance not set"
        if not isinstance(dist_b, Distribution):
            if isinstance(dist_b, np.ndarray):
                assert len(dist_b) == len(self.symbols), "Length of symbols must match"
                dist_b = Distribution(torch.ones(self.tags.shape[:2], device=self.tags.device), dist_b)
            else:
                if self.tags.dim() > 1:
                    dist_b = Distribution(torch.ones((self.tags.shape[0], 1), device=self.tags.device), [dist_b, ])
                else:
                    dist_b = Distribution(torch.ones(1, device=self.tags.device), [dist_b, ])

        res = self.__calculate_possibilities(dist_b, function, conditional)
        return res
    
    def apply(self, distributions: List[Distribution], function: Callable) -> Distribution:
        """
        self:
            symbols: T1
            distribution: torch.Tensor

        dist_b:
            symbols: T2
            distribution: torch.Tensor

        function:
            T1 x T2 -> T3 (caveat: T3 should be hashable)
        """
        return self.__compute_possibilities(distributions, function)
    
    def apply_if(self, distributions: List[Distribution], function: Callable, condition: Callable) -> Distribution:
        """
        Instead of passing just (a, b), we pass all symbols as (*args).
        For each combination of symbols from the input distributions, we only
        apply `function` if `condition` returns True, otherwise store None.
        """
        return self.__compute_possibilities(
            distributions,
            lambda *args: function(*args) if condition(*args) else None,
            conditional=True
        )

    
    def softmax(self) -> Distribution:
        probs = self.get_probabilities()
        d = Distribution(torch.nn.functional.softmax(probs, dim=-1), self.symbols, src=self.src)
        return d
    
    def map(self, function: Callable) -> Distribution:
        """
        function: something that applies to each symbol
        """
        # results = np.array(list(map(function, self.symbols)))
        results = self.__get_symbols_from_array(list(map(function, self.symbols)))
        final_tags, symbols = self.provenance.reduce_symbols(self.tags, results)
        
        return Distribution(final_tags, symbols, dist_as_probs=False, src=self.src)

    def __add__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 + s2)
    
    def __mul__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 * s2)
    
    def __sub__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 - s2)
    
    def __truediv__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 / s2)
    
    def __floordiv__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 // s2)
    
    def __mod__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 % s2)
    
    def __pow__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 ** s2)
    
    def __eq__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 == s2)
    
    def __ne__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 != s2)
    
    def __gt__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 > s2)
    
    def __ge__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 >= s2)
    
    def __lt__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 < s2)
    
    def __le__(self, dist_b: Union[Distribution|np.ndarray|Any]) -> Distribution:
        return self.__compute_possibilities(dist_b, lambda s1, s2 : s1 <= s2)

    def __invert__(self) -> Distribution:
        assert len(self.symbols) == 2, "Only binary distributions can be inverted"
        return Distribution(self.provenance.neg_batch(self.tags), self.symbols, dist_as_probs=False, src=self.src)
    
    def __repr__(self) -> str:
        return f"{{Symbols: {self.symbols}, Distribution: {self.tags}}}"
    
    def __getitem__(self, indices: int | slice | torch.Tensor | List | Tuple | None) -> Distribution:
        d = self.tags[indices]
        if torch.numel(d):
            d = d.view([-1] + list(self.tags.shape[1:]))
        else:
            d = d.view([1] + list(self.tags.shape[1:]))

        if isinstance(indices, tuple):
            s = self.symbols[indices[1]]
        else:
            if self.tags.dim() <= 1:
                s = self.symbols[indices]
            else:
                s = self.symbols

        f = Distribution(d, s, dist_as_probs=False, src=self.src)
        return f
    
    def __iter__(self):
        return (self[i] for i in range(len(self)))
    
    def __len__(self) -> int:
        return len(self.tags)
    
    def __hash__(self):
        return hash(self.id)
    
    def filter(self, filter_function) -> Distribution:
        filtered_indices = [ filter_function(s) for s in self.symbols ]
        true_symbols = self.symbols[filtered_indices]
        if self.tags.dim() == 1:
            d = Distribution(self.tags[filtered_indices], true_symbols, dist_as_probs=False, src=self.src)
        else:
            d = Distribution(self.tags[:, filtered_indices], true_symbols, dist_as_probs=False, src=self.src)

        return d
    
    def map_symbols(self, new_symbols: np.ndarray) -> Distribution:
        new_symbols = self.__get_symbols_from_array(new_symbols)

        if self.tags.dim() == 1:
            org_dist = self.tags.view(1, -1)
        else:
            org_dist = self.tags

        new_shape = list(org_dist.shape)
        new_shape[1] = len(new_symbols)

        new_dist = self.provenance.zeros(new_shape, device=self.tags.device)
        if self.inverted:
            new_dist = self.provenance.neg_batch(new_dist)

        if len(self.symbols) != 0:
            _, idx1, idx2 = np.intersect1d(self.symbols, new_symbols, return_indices=True)
            new_dist[:, idx2] = org_dist[:, idx1]

        d = Distribution(new_dist, new_symbols, dist_as_probs=False, src=self.src)
        return d
    
    def diff(self, dist_b: Distribution) -> Distribution:
        a = self.sample_top_k(self.k)
        b = dist_b.sample_top_k(self.k)

        new_symbols = np.setdiff1d(a.symbols, b.symbols, assume_unique=True)
        x = a.map_symbols(new_symbols)
        return x
    
    # probabilistic logic
    def __and__(self, dist_b: Distribution) -> Distribution:
        assert self.provenance is not None, "Provenance not set"
        symbols = np.union1d(self.symbols, dist_b.symbols)

        if not np.array_equal(self.symbols, symbols):
            a = self.map_symbols(symbols)
        else:
            a = self

        if not np.array_equal(dist_b.symbols, symbols):
            b = dist_b.map_symbols(symbols)
        else:
            b = dist_b

        tags_list, ab_src = self.provenance.combine_tag_sources(a, b)
        new_dist = self.provenance.mul_batch(tags_list[0], tags_list[1])
        d = Distribution(new_dist, symbols, dist_as_probs=False, src=ab_src)
        return d
    
    def __or__(self, dist_b: Distribution) -> Distribution:
        assert self.provenance is not None, "Provenance not set"
        symbols = np.union1d(self.symbols, dist_b.symbols)
        
        if not np.array_equal(self.symbols, symbols):
            a = self.map_symbols(symbols)
        else:
            a = self

        if not np.array_equal(dist_b.symbols, symbols):
            b = dist_b.map_symbols(symbols)
        else:
            b = dist_b

        tags_list, ab_src = self.provenance.combine_tag_sources(a, b)
        new_dist = self.provenance.add_batch(tags_list[0], tags_list[1])

        d = Distribution(new_dist, symbols, dist_as_probs=False, src=ab_src)
        return d
    
    def __invert__(self) -> Distribution:
        assert self.provenance is not None, "Provenance not set"
        new_dist = self.provenance.neg_batch(self.tags)
        d = Distribution(new_dist, self.symbols, dist_as_probs=False, src=self.src)
        d.inverted = not self.inverted

        return d

    def drop_symbol(self, symbol) -> Distribution:
        return self.filter(lambda s : s != symbol)
    
    def get_probabilities(self) -> torch.Tensor:
        return self.provenance.probs_from_tags(self.tags)
