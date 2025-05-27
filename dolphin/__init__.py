from .distribution import Distribution
from .provenances import get_provenance
from .modules import *
from . import utils

# a decorator to convert d1.apply(f, d2, d3, ...) to f(d1, d2, ...)
def func(f):
    def wrapper(*args, **kwargs):
        d = args[0]
        args = args[1:]
        if not isinstance(d, Distribution):
            # just call the function as is
            return f(d, *args, **kwargs)
        elif len(args) == 0:
            return d.map(f)
        else:
            if len(args) == 1 and isinstance(args[0], Distribution):
                args = args[0]
            if 'if' in kwargs:
                return d.apply_if(args, f, kwargs['if'])
            else:
                return d.apply(args, f)
    return wrapper
