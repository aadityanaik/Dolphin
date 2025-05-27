from typing import Any, Callable, Dict
import numpy as np

from torchql.table import Table
from functools import partial
import torch

class SQR:
    """
    Class representing a Statistical Quantile Rule.
    Initialize via a bounds table.
    """
    def __init__(self, rule: tuple, bounds: Table, batchwise=False) -> None:
        self.rule = rule
        self.bounds = bounds
        self.batchwise = batchwise
        func = locals()
        exec(self.to_function(), {"inf": np.inf, "torch": torch}, func)
        # print(func)
        self.func = func["func"]

        t = Table(self.bounds.rows, self.bounds.id_index)
        self.name = self.__to_str(self.rule, t)

    def run(self, *row) -> bool:
        """
        Run the rule on a row.
        """
        res = self.func(*row)
        return res
    
    def __call__(self, *row) -> bool:
        return self.run(*row)
    
    def __to_str(self, rule, t, indent=0, func=False) -> str:
        cond = ""
        # print(rule, t)
        var = rule[0] if (not func) else f"row[{indent}]"

        if len(t[0]) > 1:
            def grp_to_str(grp):
                if isinstance(grp, tuple):
                    assert not isinstance(grp[0], str) and not isinstance(grp[1], str), f"grp is not a tuple of numbers: {grp}"
                    return f"{grp[0]} <= {var} <= {grp[1]}" if not self.batchwise else f"torch.logical_and({grp[0]} <= {var}, {var} <= {grp[1]})"
                else:
                    if isinstance(grp, str):
                        return f"{var} == '{grp}'"
                    return f"{var} == {grp}"
                
            if self.batchwise:
                t = t.project(lambda *row: (grp_to_str(row[0]), self.__to_str(rule[1:], row[1], indent=indent+1, func=func))) \
                    .project(lambda *row: f"torch.logical_and({row[0]}, {row[1]})")
                cond = t.rows[0]
                for pred in t.rows[1:]:
                    cond = f"torch.logical_or({cond}, {pred})"
            else:
                t = t.project(lambda *row: (grp_to_str(row[0]), self.__to_str(rule[1:], row[1], indent=indent+1, func=func))) \
                    .project(lambda *row: f"({row[0]} and {row[1]})")
                cond = (" or " + "  " * indent).join(t.rows)

        else:
            # this is the case when the table's bounds are not grouped
            assert isinstance(t[0], tuple), f"t[0] is not a tuple: {t[0]}"
            bounds = t.rows
            cond = f"({bounds[0]} <= {var} <= {bounds[1]})" if not self.batchwise else f"torch.logical_and({bounds[0]} <= {var}, {var} <= {bounds[1]})"

        return cond
    
    def __str__(self) -> str:
        return self.name
    
    def to_function(self) -> str:
        """
        Construct a function that runs the rule.
        """
        func_str = self.__to_str(self.rule, Table(self.bounds.rows, self.bounds.id_index), func=True)
        return f"""
def func(*row):
    return {func_str}
"""
    
    def __repr__(self) -> str:
        return str(self)