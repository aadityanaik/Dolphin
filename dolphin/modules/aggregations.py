from functools import reduce
import numpy as np
import torch

from ..distribution import Distribution

class Collate(torch.nn.Module):
    """
    Module that takes a set of distributions and collates them into a single distribution where each symbol is a list of
    symbols from the input distributions.

    Example:
    >>> collate = Collate()
    >>> d1 = Distribution(p1, [1, 2, 3])
    >>> d2 = Distribution(p2, [4, 5])
    >>> collate(d1, d2)
    Symbols: [[1, 4], [1, 5], [2, 4], [2, 5], [3, 4], [3, 5]]
    Distribution: ...
    """
    def __init__(self):
        super().__init__()

    def forward(self, *distrs: Distribution, k=None):
        # each distr is a distribution
        assert all([isinstance(d, Distribution) for d in distrs]), "All inputs must be Distributions"

        res = distrs[0].map(lambda x: (x, ))
        for d in distrs[1:]:
            if k is not None:
                res = res.sample_top_k(k)
            res = res.apply(d, lambda x, y: x + (y, ))

        return res
    
class Sum(torch.nn.Module):
    """
    Module that takes a set of distributions and sums them up.

    Example:
    >>> sum = Sum()
    >>> d1 = Distribution(p1, [1, 2, 3])
    >>> d2 = Distribution(p2, [4, 5])
    >>> sum(d1, d2)
    Symbols: [5, 6, 7, 8]
    Distribution: ...
    """
    def __init__(self, summation_fn = lambda x : reduce(lambda a, b: a + b, x)):
        super().__init__()
        self.summation_fn = summation_fn
        self.collate = Collate()

    def forward(self, *distrs: Distribution):
        # each distr is a distribution
        assert all([isinstance(d, Distribution) for d in distrs]), "All inputs must be Distributions"

        collated_distr = self.collate(*distrs)
        return collated_distr.map(lambda x: self.summation_fn(x))
    
class Product(torch.nn.Module):
    """
    Module that takes a set of distributions and multiplies them.

    Example:
    >>> product = Product()
    >>> d1 = Distribution(p1, [1, 2, 3])
    >>> d2 = Distribution(p2, [4, 5])
    >>> product(d1, d2)
    Symbols: [4, 5, 8, 10, 12, 15]
    Distribution: ...
    """
    def __init__(self, product_fn = lambda x : reduce(lambda a, b: a * b, x)):
        super().__init__()
        self.product_fn = product_fn
        self.collate = Collate()

    def forward(self, *distrs: Distribution):
        # each distr is a distribution
        assert all([isinstance(d, Distribution) for d in distrs]), "All inputs must be Distributions"

        collated_distr = self.collate(*distrs)
        return collated_distr.map(lambda x: self.product_fn(x))
    
class Max(torch.nn.Module):
    """
    Module that takes a set of distributions and returns the maximum value for each symbol.

    Example:
    >>> max = Max()
    >>> d1 = Distribution(p1, [1, 2, 3])
    >>> d2 = Distribution(p2, [4, 5])
    >>> max(d1, d2)
    Symbols: [4, 5]
    Distribution: ...
    """
    def __init__(self, max_fn = lambda x : max(x)):
        super().__init__()
        self.collate = Collate()
        self.max_fn = max_fn

    def forward(self, *distrs: Distribution):
        # each distr is a distribution
        assert all([isinstance(d, Distribution) for d in distrs]), "All inputs must be Distributions"

        collated_distr = self.collate(*distrs)
        return collated_distr.map(lambda x: self.max_fn(x))
    
class Min(torch.nn.Module):
    """
    Module that takes a set of distributions and returns the minimum value for each symbol.

    Example:
    >>> min = Min()
    >>> d1 = Distribution(p1, [1, 2, 3])
    >>> d2 = Distribution(p2, [4, 5])
    >>> min(d1, d2)
    Symbols: [1, 2, 3]
    Distribution: ...
    """
    def __init__(self, min_fn = lambda x : min(x)):
        super().__init__()
        self.collate = Collate()
        self.min_fn = min_fn

    def forward(self, *distrs: Distribution):
        # each distr is a distribution
        assert all([isinstance(d, Distribution) for d in distrs]), "All inputs must be Distributions"

        collated_distr = self.collate(*distrs)
        return collated_distr.map(lambda x: self.min_fn(x))
    
class Mean(torch.nn.Module):
    """
    Module that takes a set of distributions and returns the mean value for each symbol.

    Example:
    >>> mean = Mean()
    >>> d1 = Distribution(p1, [1, 2, 3])
    >>> d2 = Distribution(p2, [4, 5])
    >>> mean(d1, d2)
    Symbols: [2.5, 3, 3.5, 4]
    Distribution: ...
    """
    def __init__(self, mean_fn = lambda x : np.mean(x)):
        super().__init__()
        self.collate = Collate()
        self.mean_fn = mean_fn

    def forward(self, *distrs: Distribution):
        # each distr is a distribution
        assert all([isinstance(d, Distribution) for d in distrs]), "All inputs must be Distributions"

        collated_distr = self.collate(*distrs)
        return collated_distr.map(lambda x: self.mean_fn(x))
    
class Count(torch.nn.Module):
    """
    Module that takes a condition and a set of distributions and returns the number of distributions that satisfy the condition.

    Example:
    >>> count = Count(lambda x : x % 2 == 0)
    >>> d1 = Distribution(p1, [1, 2, 3])
    >>> d2 = Distribution(p2, [4, 5])
    >>> count(d1, d2)
    Symbols: [0, 1, 2]
    Distribution: ...
    """
    def __init__(self, condition_fn):
        super().__init__()
        self.condition_fn = condition_fn
        self.collate = Collate()

    def forward(self, *distrs: Distribution):
        # each distr is a distribution
        assert all([isinstance(d, Distribution) for d in distrs]), "All inputs must be Distributions"

        collated_distr = self.collate(*distrs)
        return collated_distr.map(lambda x: sum([self.condition_fn(y) for y in x]))