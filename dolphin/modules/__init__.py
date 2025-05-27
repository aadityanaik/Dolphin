from typing import Callable
from .conditions import Condition, Case
from .aggregations import Collate, Sum, Max, Min, Count, Mean, Product


# Syntactic sugar for aggregations

sum : Callable = Sum()
prod : Callable = Product()
max : Callable = Max()
min : Callable = Min()
count = lambda condition, *distrs: Count(condition)(*distrs)
mean : Callable = Mean()
collate : Callable = Collate()

