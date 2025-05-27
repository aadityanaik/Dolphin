from __future__ import annotations
from typing import Callable, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from io import BytesIO
import pickle
import tarfile
import os
import mmap
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .utils import get_iterable

def is_dataset(object):
    return (hasattr(object, "__getitem__") and callable(getattr(object, "__getitem__"))
            and hasattr(object, "__len__") and callable(getattr(object, "__len__")))

class Table(Dataset):
    rows: Union[dict, Dataset, list]

    def __init__(self, samples) -> None:
        super().__init__()
        if isinstance(samples, list) or is_dataset(samples):
            self.rows = samples
        else:
            assert isinstance(samples, dict), "Samples can either be in a dict or a list"
            self.rows = samples
        self.index_to_row = {i: i for i in range(len(self.rows))}

    def __len__(self) -> int:
        return len(self.index_to_row.keys())

    def __iter__(self):
        return (self.__getitem__(i) for i in range(len(self)))
    
    def __getitem__(self, index):
        row = self.rows[self.index_to_row[index]]
        if isinstance(row, dict):
            row = list(row.values())
        return row

    def get_rowid(self, index):
        return self.index_to_row[index]

    def has_cache(self):
        return False

    def join(self, table, key=None, fkey=None, fuzzy=False) -> Table:
        return JoinTable(self, table, key, fkey, fuzzy)

    def filter(self, cond: Callable[..., bool]) -> Table:
        return FilterTable(self, cond)

    def project(self, cond: Callable[..., list]) -> Table:
        return ProjectTable(self, cond)

    def flatten(self) -> Table:
        return FlattenTable(self)

    def group_by(self, group=None) -> Table:
        return GroupTable(self, group)

    def order_by(self, order=None, reverse=False) -> Table:
        return OrderTable(self, order, reverse)

    def cache(self) -> Table:
        return CacheTable(self)

    def head(self, k) -> Table:
        return HeadTable(self, k)

    def save(self, name: str) -> Table:
        return SavedTable(name, "", self)


class SavedTable(Table):
    def __init__(self, path: str, name: str, end_id=-1, table=None) -> None:
        self.table = table
        self.name = name

        exists = False
        if os.path.exists(f'{path}.tar'):
            with tarfile.open(f'{path}.tar', 'r') as tar:
                names = set([os.path.dirname(n) for n in tar.getnames()])
                print(names)
                exists = name in names

        if not exists:
            assert table is not None
            with tarfile.open(f'{path}.tar', 'a') as tar:
                for i, row in tqdm(enumerate(table)):
                    bytes_io = BytesIO()
                    pickle.dump(row, bytes_io)
                    bytes_io.seek(0)
                    fname = f"{name}/row_{i}.pkl"
                    tarinfo = tarfile.TarInfo(fname)
                    tarinfo.size = bytes_io.getbuffer().nbytes
                    tar.addfile(tarinfo, fileobj=bytes_io)

        with open(f'{path}.tar', 'rb') as f:
            self.archive = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            self.archive = tarfile.open(fileobj=self.archive, mode='r')
            self.all_files = [n for n in self.archive.getnames() if n.startswith(name)]
            if end_id > 0:
                self.all_files = [self.all_files[idx] for idx in range(end_id)]

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, index):
        fname = f"{self.name}/row_{index}.pkl" if self.name != "" else f"row_{index}.pkl"
        bytes_io = self.archive.extractfile(fname)
        return pickle.load(bytes_io)


class JoinTable(Table):
    def __init__(self, table1: Table, table2: Table, key=None, fkey=None, fuzzy=False) -> None:
        self.table1 = table1
        self.table2 = table2
        self.index_to_row = {}
        print(fuzzy)
        if fuzzy:
            table2_emb_idx = []
        else:
            table2_keymap = {}
        if fkey == None:
            table2_keymap = {i: [i] for i in range(len(self.table2))}
        else:
            def join_worker(x):
                row = self.table2[x]
                return fkey(*row), row

            results = map(join_worker, (i for i in range(len(table2))))
            for table2_idx, res in tqdm(enumerate(results), total=len(table2), desc="Join index"):
                if fuzzy:
                    assert type(res[0]) == np.ndarray
                    table2_emb_idx.append((res[0], table2_idx))
                else:
                    if res[0] in table2_keymap:
                        table2_keymap[res[0]].append(table2_idx)
                    else:
                        table2_keymap[res[0]] = [table2_idx]

        def main_join_worker(x):
            if key == None:
                return x
            return key(*self.table1[x])

        if fuzzy:
            neigh = KNeighborsClassifier(n_neighbors=2)
            table2_emb, table2_idx = list(zip(*table2_emb_idx))
            table2_emb = np.array(table2_emb).reshape((len(table2_emb), -1))
            table2_emb = table2_emb / np.linalg.norm(table2_emb, axis=1, keepdims=True)
            print(table2_emb.shape)
            neigh.fit(table2_emb, table2_idx)
        results = map(main_join_worker, (i for i in range(len(table1))))
        new_idx = 0
        for i, res in tqdm(enumerate(results), total=len(self.table1), desc="Join main"):
            if fuzzy:
                for table2_idx in neigh.predict(res):
                    self.index_to_row[new_idx] = (i, table2_idx)
                    new_idx += 1
            else:
                if res in table2_keymap:
                    for table2_idx in table2_keymap[res]:
                        self.index_to_row[new_idx] = (i, table2_idx)
                        new_idx += 1

    def __getitem__(self, index):
        table1_idx, table2_idx = self.index_to_row[index]
        t1 = get_iterable(self.table1[table1_idx])
        t2 = get_iterable(self.table2[table2_idx])
        return (*t1, *t2)

    def get_rowid(self, index):
        table1_idx, _ = self.index_to_row[index]
        return self.table1.get_rowid(table1_idx)

    def has_cache(self):
        return False

class FilterTable(Table):
    def __init__(self, table: Table, filter: Callable[..., bool]) -> None:
        self.table = table
        self.filter_fn = filter

        self.index_to_row = {}
        index = 0
        filter_worker = lambda x: self.filter_fn(*self.table[x])
        results = map(filter_worker, (i for i in range(len(table))))
        for i, res in tqdm(enumerate(results), total=len(table), desc="Filter"):
            if res:
                self.index_to_row[index] = i
                index += 1

    def __getitem__(self, index):
        return self.table[self.index_to_row[index]]

    def get_rowid(self, index):
        return self.table.get_rowid(self.index_to_row[index])

class ProjectTable(Table):
    def __init__(self, table, project: Callable[..., list]) -> None:
        self.table = table
        self.project_fn = project
        self.index_to_row = {i:i for i in range(len(self.table))}

    def __getitem__(self, index):
        row = self.project_fn(*self.table[index])
        return row

    def get_rowid(self, index):
        return self.table.get_rowid(self.index_to_row[index])

class FlattenTable(Table):
    def __init__(self, table) -> None:
        self.table = table
        self.index_to_row = {}

        def flatten_fn(x):
            row = table[x]
            return len(row[0]), row

        results = map(flatten_fn, (i for i in range(len(table))))
        idx = 0
        for i, res in tqdm(enumerate(results), total=len(table), desc="Flatten"):
            self.index_to_row.update({idx + j: (i, j) for j in range(res[0])})
            idx += res[0]

    def __getitem__(self, index):
        index, sub_index = self.index_to_row[index]
        return self.table[index][0][sub_index]

    def get_rowid(self, index):
        index, sub_index = self.index_to_row[index]
        return self.table.get_rowid(index)

class GroupTable(Table):
    def __init__(self, table, group) -> None:
        self.table = table
        self.index_to_row = {}

        if group == None:
            group_fn = lambda x: table.get_rowid(x)
        else:
            group_fn = lambda x: group(*table[x])
        results = map(group_fn, (i for i in range(len(table))))
        for i, res in tqdm(enumerate(results), total=len(table), desc="Group"):
            if res in self.index_to_row:
                self.index_to_row[res].append(i)
            else:
                self.index_to_row[res] = [i]

        self.index_to_row = {i: entry for i, entry in enumerate(self.index_to_row.items())}

    def __getitem__(self, index):
        key, idxs = self.index_to_row[index]
        rows = [self.table[i] for i in idxs]
        return [key, rows]

    def get_rowid(self, index):
        sub_index = self.index_to_row[index][0]
        return self.table.get_rowid(sub_index)

class OrderTable(Table):
    def __init__(self, table, order, reverse) -> None:
        self.table = table
        self.index_to_row = {i: i for i in range(len(table))}
        self.order_fn = order

        def order_worker(x):
            return self.order_fn(*self.table[x])

        results = map(order_worker, (i for i in range(len(table))))
        sorted_results = sorted(tqdm(enumerate(results), total=len(table), desc="Order"), reverse=reverse, key=lambda x: x[1])
        self.index_to_row = {i: entry[0] for i, entry in enumerate(sorted_results)}

    def __getitem__(self, index):
        return self.table[self.index_to_row[index]]

    def get_rowid(self, index):
        return self.table.get_rowid(self.index_to_row[index])

class HeadTable(Table):
    def __init__(self, table, k) -> None:
        self.table = table
        self.index_to_row = {i: i for i in range(min(k, len(table)))}

    def __getitem__(self, index):
        return self.table[index]

    def get_rowid(self, index):
        return self.table.get_rowid(index)


class CacheTable(Table):
    def __init__(self, table) -> None:
        self.table = table
        self.index_to_row = {i: i for i in range(len(table))}
        self.cache_table = {}

    def __getitem__(self, index):
        if index in self.cache_table:
            return self.cache_table[index]
        else:
            row = self.table[index]
            self.cache_table[index] = row
            return row

    def get_rowid(self, index):
        return self.table.get_rowid(index)

    def has_cache(self):
        return True