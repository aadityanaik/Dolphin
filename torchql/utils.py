from collections import namedtuple
from typing import TypeVar, Optional, Union, Iterable, Sequence, Callable, List, Any
from torch.utils.data import DataLoader, Dataset, Sampler
from torch import Tensor


Operation = namedtuple('Operation', ['op', 'arg', 'kwargs'])

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

def get_iterable(entry):
        if isinstance(entry, (list, tuple)):
            return entry
        elif isinstance(entry, Tensor):
            # if entry.dim() == 0:
            #     return (entry.item(), )
            return (entry, )
        elif isinstance(entry, dict):
             return list(entry.values())
        else:
            return (entry, )

class IndexedDataloader(DataLoader):
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: Optional[bool] = None, sampler: Union[Sampler, Iterable, None] = None, batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False, timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False, pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index) -> T_co:
        return self.dataset[index]
    