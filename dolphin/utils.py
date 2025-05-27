from collections import defaultdict
from itertools import chain, combinations
from typing import Dict, Optional, Union, Type, Tuple, Callable
import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate_fn_map, collate

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def distribution_collate_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    from .distribution import Distribution

    def reshape_tags(distr, final_shape):
        provenance = distr.provenance
        tags = distr.tags
        if tags.shape[-1] < final_shape[-1]:
            tags = torch.cat([tags, torch.ones(tags.shape[:-1] + (final_shape[-1] - tags.shape[-1],), device=tags.device) * provenance._NULL], dim=-1)
        return tags.view(final_shape[1:])

    elem = batch[0]
    assert isinstance(elem, Distribution), f"Expected a Distribution, but got {type(elem)}"
    
    tag_shape = elem.tags.shape
    for e in batch[1:]:
        if e.tags.shape[-1] > tag_shape[-1]:
            tag_shape = e.tags.shape

    if all([np.array_equal(item.symbols, elem.symbols) for item in batch]):
        final_shape = [len(batch), len(elem.symbols)] + list(tag_shape)[2:]
        tags = torch.stack([reshape_tags(item, final_shape) for item in batch]).view(final_shape)
        d = Distribution(tags, elem.symbols, dist_as_probs=False)
    else:
        symbols = np.concatenate([item.symbols for item in batch], axis=None)
        symbols = np.unique(symbols)
        final_shape = [len(batch), len(symbols)] + list(tag_shape)[2:]

        new_distrs = [ reshape_tags(d.map_symbols(symbols), final_shape) for d in batch ]
        tags = torch.stack(new_distrs).view(final_shape)
        d = Distribution(tags, symbols, dist_as_probs=False)

    return d

def symbolic_collate_fn(batch):
    from .distribution import Distribution
    custom_collate_map = dict(default_collate_fn_map)
    custom_collate_map[Distribution] = distribution_collate_fn
    return collate(batch, collate_fn_map=custom_collate_map)
