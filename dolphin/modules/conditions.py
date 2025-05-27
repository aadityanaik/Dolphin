import numpy as np
import torch

from ..distribution import Distribution

# Conditional Module
class Condition(torch.nn.Module):
    def __init__(self, condition_module, then_module, else_module = None):
        super().__init__()
        self.condition = condition_module
        self.then_module = then_module
        self.else_module = else_module

    def forward(self, *x):
        cond = self.condition(*x)
        # preprocessing
        assert isinstance(cond, Distribution), "Condition must return a Distribution"
        
        if True not in cond.symbols:
            assert False in cond.symbols and len(cond.symbols) == 1, \
                "Condition must return a Distribution with the symbol True or False"
            cond = cond.map_symbols([False, True])
        elif False not in cond.symbols:
            assert True in cond.symbols and len(cond.symbols) == 1, \
                "Condition must return a Distribution with the symbol True or False"
            cond = cond.map_symbols([False, True])
        else:
            assert (True in cond.symbols or False in cond.symbols) and len(cond.symbols) == 2, \
                "Condition must return a Distribution with two symbols: True and False"

        then_result = self.then_module(*x)
        # print(then_result)
        assert isinstance(then_result, Distribution), "Condition must return a Distribution"
        # assert then_result.distribution.shape[0] == cond.distribution.shape[0], \
        #     "Condition and then result must have the same number of rows"

        then_result.tags = then_result.tags * cond.filter(lambda s : s == True).tags.view(-1, 1)

        else_result = None        
        if self.else_module is not None:
            else_result = self.else_module(*x)
            assert isinstance(else_result, Distribution), "Condition must return a Distribution"
            # assert else_result.distribution.shape[0] == cond.distribution.shape[0], \
            #     "Condition and else result must have the same number of rows"
        
            else_result.tags = else_result.tags * cond.filter(lambda s : s == False).tags.view(-1, 1)

        # print(then_result, else_result)

        return then_result, else_result


# Cases Module
class Case(torch.nn.Module):
    def __init__(self, *cases):
        super().__init__()
        self.cases = cases

    def forward(self, *x):
        pass