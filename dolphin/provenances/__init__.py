from .damp import DAMP
from .dmmp import DMMP
from .dtkp_am import DTKP_AM


def get_provenance(provenance: str):
    if provenance == "damp":
        return DAMP()
    elif provenance == "dmmp":
        return DMMP()
    elif provenance == "dtkp-am":
        return DTKP_AM()
    else:
        raise ValueError(f"Provenance {provenance} not supported.")