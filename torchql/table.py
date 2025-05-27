from __future__ import annotations
from collections import defaultdict
import logging
import random
from time import time
from typing import Any, Callable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .utils import get_iterable
from functools import partialmethod
import copy
from multiprocessing.pool import ThreadPool
import torch

from dolphin.utils import symbolic_collate_fn

logger = logging.getLogger("torchql.table")

logger.stats = defaultdict(float)
logger.reset_stats = lambda : logger.stats.clear()

# Uncomment to disable tqdm
# tqdm.__init__ = partialmethod(tqdm.__init__)

# def nop(it, *a, **k):
#     return it

# tqdm = nop

def is_dataset(object):
    return (hasattr(object, "__getitem__") and callable(getattr(object, "__getitem__"))
            and hasattr(object, "__len__") and callable(getattr(object, "__len__")))

class Table(Dataset):
    """
    A Table is a collection of samples.
    It is specifically designed to be queried from within a database.
    It is a subclass of torch.utils.data.Dataset, so it can be used as a dataset.
    """
    def __init__(self, samples, id_index: dict=None, transform=None, disable=True, **kwargs) -> None:
        super().__init__()
        self.transform = transform
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if id_index is None:
            self.id_index = {}
            build_idx = True
        else:
            self.id_index = id_index
            build_idx = False

        # building the index
        if isinstance(samples, (list, tuple)):
            self.rows = samples
            # self.rows = list(samples)
            if build_idx:
                for i in range(len(self.rows)):
                    self.id_index[i] = i
        elif isinstance(samples, dict):
            self.rows = []
            self.id_index = {}
            for key_idx, key in enumerate(samples):
                self.rows.append(samples[key])
                self.id_index[key] = key_idx
                assert self.rows[key_idx] == samples[key]
        elif is_dataset(samples):
            # print("HERE", samples[0])
            self.rows = []
            for row_idx, row in tqdm(enumerate(samples), total=len(samples), desc="Building table", disable=disable):
                # print("Appending row:", row)
                self.rows.append(row)
                if build_idx:
                    self.id_index[row_idx] = row_idx
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return len(self.rows)
    
    def __iter__(self):
        return iter(self.rows)
    
    def __getitem__(self, index):
        return get_iterable(self.transform(self.rows[index]) if self.transform is not None else self.rows[index])
    
    def __contains__(self, item):
        return item in self.rows
    
    def __eq__(self, t: Table) -> bool:
        return self.id_index == t.id_index and self.rows == t.rows
    
    def __repr__(self) -> str:
        repr_rows = self.rows[:10]
        s = "\n".join([str(row) for row in repr_rows])
        if len(self.rows) > 10:
            s += "\n... ({} more)".format(len(self.rows) - 10)
        return s

    def transform(self, transform) -> Table:
        """
        Register a PyTorch Transform to apply to this table.

        Args:
            transform (Callable): The PyTorch transform to apply to the rows of this table.

        Returns:
            A new table with the transformed rows.
        """
        return Table(self.rows, self.id_index, transform=transform)
    
    def join(self, table: Dataset, key=None, fkey=None, batch_size=0, disable=True, prov = False) -> Table:
        """
        Join this table with another table.
        Rows are joined if the key function returns the same value for both tables.
        If no key function is provided, the index is used.

        Args:
            table (Dataset): The table to join with.

            key (Callable, optional): The key function to use for this table. Defaults to None.
                Must take a set of columns from a row as input and return a hashable value that serves as a key.

            fkey (Callable, optional): The foreign key function to use for the other table. Defaults to None.
                Must take a set of columns from a row as input and return a hashable value that serves as a key.

            prov (bool, optional): Whether to track indices of joined rows. Defaults to False.

        Returns:
            A new table with the joined rows.
        """
        t = time()
        l_indices = {}
    
        for id, index in tqdm(self.id_index.items(), desc="Building index for left table", disable=disable):
            if key is None:
                l_indices[id] = {id, }
            else:
                row = self[index]
                if id not in l_indices:
                    l_indices[id] = set()
                l_indices[id].add(key(id, *row))
        
        r_indices = {}

        def table_iter():
            if isinstance(table, Table):
                for id, index in tqdm(table.id_index.items(), desc="Building index for right table", disable=disable):
                    yield id, table[index]
            else:
                for id, row in tqdm(enumerate(table), desc="Building index for right table", disable=disable):
                    yield id, get_iterable(row)

        def get_table_item(id):
            if isinstance(table, Table):
                return table[table.id_index[id]]
            else:
                return table[id]

        for id, row in table_iter():
            if fkey is None:
                r_indices[id] = {id, }
            else:
                fk_val = fkey(id, *row)
                if fk_val not in r_indices:
                    r_indices[fk_val] = set()
                r_indices[fk_val].add(id)

        joined_rows = []
        joined_indices = {}

        for lid, lvals in tqdm(l_indices.items(), desc="Joining", disable=disable):
            for lval in lvals:
                if lval in r_indices:
                    for rid in r_indices[lval]:
                        l_entry = self[self.id_index[lid]]
                        l_entry = get_iterable(l_entry)
                        r_entry = get_table_item(rid)
                        r_entry = get_iterable(r_entry)
                        joined_rows.append((*l_entry, *r_entry))
                        joined_indices[(lid, rid)] = len(joined_rows) - 1
        idx = None if not prov else joined_indices
        return Table(joined_rows, id_index=idx, transform=self.transform)
    
    def union(self, table: Dataset, batch_size=None, disable=True) -> Table:
        """
        Union this table with another table.
        Rows of the other table will be added to the bottom of the table.

        Args:
            table (Dataset): The table to union with.

        Returns:
            A new table with the combined rows.
        """
        unioned_rows = []
        unioned_indices = {}

        if batch_size is None:
            for id, index in tqdm(self.id_index.items(), desc="Getting rows of self table", disable=disable):
                row = self[index]
                row = get_iterable(row)
                
                unioned_rows.append(row)
                unioned_indices[id] = len(unioned_rows) - 1
        else:
            for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=symbolic_collate_fn), desc="Getting rows of self table", disable=disable):
                unioned_rows.extend(row_batch)

        def table_iter():
            if isinstance(table, Table):
                for id, index in tqdm(table.id_index.items(), desc="Getting rows of other table", disable=disable):
                    yield id, table[index]
            else:
                for id, row in tqdm(enumerate(table), desc="Getting rows of other table", disable=disable):
                    yield id, get_iterable(row)
        
        for id, row in table_iter():
            if isinstance(table, Table):
                row = get_iterable(row)

            unioned_rows.append(row)
            unioned_indices[id] = len(unioned_rows) - 1

        return Table(unioned_rows, self.transform)
    
    def intersect(self, table: Dataset, batch_size=None, disable=True) -> Table:
        """
        Intersect this table with another table.
        Common rows between this and other table will be used to create a new table.
        Common rows are identified by the id of the row.
        The columns of the other table will be used in the new table.

        Args:
            table (Dataset): The table to intersect with.

        Returns:
            A new table with the common rows.
        """
        self_rows = []
        self_indices = {}

        if batch_size is None:
            for id, index in tqdm(self.id_index.items(), desc="Getting rows of self table", disable=disable):
                row = self[index]
                row = get_iterable(row)
                
                self_rows.append(row)
                self_indices[id] = len(self_rows) - 1
        else:
            for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=symbolic_collate_fn), desc="Getting rows of self table", disable=disable):
                self_rows.extend(row_batch)

        def table_iter():
            if isinstance(table, Table):
                for id, index in tqdm(table.id_index.items(), desc="Getting rows of other table", disable=disable):
                    yield id, table[index]
            else:
                for id, row in tqdm(enumerate(table), desc="Getting rows of other table", disable=disable):
                    yield id, get_iterable(row)
        
        intersect_rows = []
        intersect_indices = {}

        for id, row in table_iter():
            if id in self_indices:

                if isinstance(table, Table):
                    row = get_iterable(row)

                intersect_rows.append(row)
                intersect_indices[id] = len(intersect_rows) - 1

        return Table(intersect_rows, self.transform)
    
    def filter(self, cond: Callable[..., bool], batch_size=None, disable=True) -> Table:
        """
        Filter this table by a condition.

        Args:
            cond (Callable): The condition to filter by.
        
        Returns:
            A new table with the filtered rows.
        """
        filtered_rows = []
        filtered_indices = {}
        if batch_size is None:
            for id, index in tqdm(self.id_index.items(), desc="Filtering", disable=disable):
                row = self[index]
                row = get_iterable(row)
                if cond(*row):
                    filtered_rows.append(row)
                    filtered_indices[id] = len(filtered_rows) - 1
        else:
            for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=symbolic_collate_fn), desc="Filtering", disable=disable):
                condArr = cond(*row_batch)
                for i in range(len(condArr)):
                    if condArr[i]:
                        filtered_rows.append(tuple(c[i] for c in row_batch))

        return Table(filtered_rows, self.transform)

    def project(self, cols: Callable[..., list], batch_size=None, disable=True, shuffle=False) -> Table:
        """
        Select or perform an operation on the columns of this table.

        Args:
            cols (Callable): A function that takes the columns of this table as arguments and returns a list of the
            projected columns.
        
        Returns:
            A new table with the projected columns.
        """
        projected_rows = []
        projected_indices = {}

        # print(f"Before project: {len(self.rows)}")
        # print(batch_size)
        # total_time = 0
        # total_collate = 0
        # t_begin = time()

        if batch_size is None:
            for id, index in tqdm(self.id_index.items(), desc="Projecting", disable=disable):
                row = self[index]
                row = get_iterable(row)
                projected_rows.append(cols(*row))
                projected_indices[id] = len(projected_rows) - 1
            return Table(projected_rows, projected_indices, transform=self.transform)
        else:
            if batch_size < 0:
                batch_size = max(len(self.rows), 1)
          
            # print(batch_size, len(self.rows))
            # t_d_init = time()
            d = DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=symbolic_collate_fn)
            # logger.stats["T_project_dataloader_init"] += time() - t_d_init
            # t_d_init = time()
            for row_batch in tqdm(d, desc="Projecting", disable=disable):
                # print("BEFORE", row_batch)
                # t = time()
                res = cols(*row_batch)

                # total_time += time() - t
                
                # t = time()
                if isinstance(res, (list, tuple)):
                    l = len(res[0])

                    # for c in res:
                    #     assert len(c) == l, f"All columns returned by the projection function must have the same length: {l} != {len(c)}"
                    # print(res)
                    try:
                        for i in range(l):
                            projected_rows.append(tuple(c[i] for c in res))
                    except Exception as e:
                        raise e
                        
                    # print(projected_rows)
                else:
                    # print("HERE")
                    # print(res)
                    projected_rows.extend(res)
                # total_collate += time() - t
                    
                # print(res)
                # print("AFTER",projected_rows)
            # print("Processed all batches")

            # print("Total time for project", time() - t_begin, "; Total time for calculation", total_time, "; Total time for collate", total_collate, "for batch size", batch_size, "and total rows", len(self.rows))
            # logger.stats["T_project"] += time() - t_d_init
            # logger.stats["T_project_calc"] += total_time
            # logger.stats["T_project_collate"] += total_collate
            return Table(projected_rows, transform=self.transform)
    
    def unique(self, batch_size=None, disable=True) -> Table:
        """
        Select the unique rows of this table.

        Returns:
            A new table with the unique rows.
        """
        return Table(list(set(self.rows)), transform=self.transform)
    
    def batch(self, size, shuffle, batch_size=None, disable=True) -> Table:
        """
        Batch this table.

        Args:
            size (int): The size of the batches.
            random (bool): Whether to shuffle the rows before batching.

        Returns:
            A new table with the batches.
        """
        if shuffle:
            rows = list(self.rows)
            random.shuffle(rows)
        else:
            rows = list(self.rows)
        batches = [rows[i:i+size] for i in range(0, len(rows), size)]
        return Table(batches, transform=self.transform)
    
    def flatten(self, batch_size=None, disable=True) -> Table:
        """
        Flatten this table. If the rows of this table are lists, the rows of the new table will be the elements of the
        lists.
        """
        flattened_rows = []
        flattened_indices = {}
        for id, index in tqdm(self.id_index.items(), desc="Flattening", disable=disable):
            rowlist = self.rows[index]
            for subidx, row in enumerate(rowlist):
                flattened_rows.append(row)
                flattened_indices[len(flattened_rows) - 1] = len(flattened_rows) - 1
        return Table(flattened_rows, transform=self.transform)

    def order_by(self, key: Callable[..., Any], reverse=False, batch_size=None, disable=True) -> Table:
        """
        Order this table by a key.

        Args:
            key (Callable): The key to order by.

            reverse (bool, optional): Whether to reverse the order. Defaults to False.

        Returns:
            A new table with the ordered rows.
        """
        ordered_rows = []
        ordered_indices = {}
        if batch_size is None :
            for id, index in tqdm(self.id_index.items(), desc="Ordering", disable=disable):
                row = self[index]
                ordered_rows.append((key(*row), id, row))
        # else:
        #     for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False), desc="Projecting", disable=disable):
        #         ordered_rows.extend(row_batch)

        ordered_rows.sort(key=lambda x: x[0], reverse=reverse)
        for idx, (key, id, row) in enumerate(ordered_rows):
            ordered_indices[id] = idx

        
        # return Table([row for _, _, row in ordered_rows], ordered_indices, transform=self.transform)
        return Table([row for _, _, row in ordered_rows], transform=self.transform)

    def group_by(self, key: Callable[..., Any], batch_size=None, disable=True) -> Table:
        """
        Group this table by a key.
        Rows are grouped by the key function. The key function should return a hashable value. The rows of the new table
        will be tuples of the key and a list of the rows that have that key.

        Args:
            key (Callable): The key to group by.
                Must return a hashable value.

        Returns:
            A new table with the grouped rows.
        """
        grouped_rows = []
        groups = {}
        if batch_size is None:
            for idx, index in tqdm(self.id_index.items(), desc="Grouping", disable=disable):
                row = self[index]
                grouping_key = key(*row)
                if grouping_key not in groups:
                    groups[grouping_key] = []
                groups[grouping_key].append(row)
        else:
           for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=symbolic_collate_fn), desc="Filtering", disable=disable):
                # key1:{idx_11,idx_12...},key2:..
                grouping_keys = key(*row_batch)
                for grouping_key,idx_arr in grouping_keys:
                    if grouping_key not in groups:
                        groups[grouping_key] = []
                    groups[grouping_key].extend(row_batch[idx_arr])                

        for grouping_key, group_rows in groups.items():
            grouped_rows.append((grouping_key, Table(group_rows)))

        return Table(grouped_rows, transform=self.transform)

    def group_by_with_index(self, key: Callable[..., Any], batch_size=None, disable=True) -> Table:
        """
        Group this table by a key.
        Rows are grouped by the key function. The key function should return a hashable value. The rows of the new table
        will be tuples of the key and a list of the rows that have that key.

        Args:
            key (Callable): The key to group by.
                Must return a hashable value.

        Returns:
            A new table with the grouped rows.
        """
        grouped_rows = []
        groups = {}
        if batch_size is None:
            for idx, index in tqdm(self.id_index.items(), desc="Grouping", disable=disable):
                row = self[index]
                grouping_key = key(idx, *row)
                if grouping_key not in groups:
                    groups[grouping_key] = []
                groups[grouping_key].append(row)
        else:
           for row_batch in tqdm(DataLoader(self, batch_size=batch_size, shuffle=False, collate_fn=symbolic_collate_fn), desc="Filtering", disable=disable):
                # key1:{idx_11,idx_12...},key2:..
                grouping_keys = key(idx, *row_batch)
                for grouping_key,idx_arr in grouping_keys:
                    if grouping_key not in groups:
                        groups[grouping_key] = []
                    groups[grouping_key].extend(row_batch[idx_arr])

        for grouping_key, group_rows in groups.items():
            grouped_rows.append((grouping_key, Table(group_rows)))

        return Table(grouped_rows, transform=self.transform)
    
    def reduce(self, reduction: Callable[..., Any], batch_size=None, disable=True) -> Table:
        """
        Reduce the rows of this table using a reduction function. This function operates over all the rows of the table
        as opposed to each row individually.

        Args:
            reduction (Callable): The reduction function that takes in the rows of the table.

        Returns:
            A new table with the reduced rows.
        """

        return reduction(Table(self.rows, self.id_index, transform=self.transform))
    
    def group_reduce(self, key: Callable[..., Any], reduction: Callable[..., Any], batch_size=None, disable=True) -> Table:
        """
        Group this table by a key and reduce the rows of each group using a reduction function.

        Args:
            key (Callable): The key to group by.
                Must return a hashable value.

            reduction (Callable): The reduction function that takes in the rows of each group.

        Returns:   
            A new table with the grouped and reduced rows.
        """

        return self.group_by(key, disable=disable).project(lambda key, group: (key, reduction(group)), disable=disable, batch_size=batch_size)

    
    def head(self, n=10, print_id=False):
        """
        Get the first n rows of this table.

        Args:
            n (int, optional): The number of rows to get. Defaults to 10.

            print_id (bool, optional): Whether to print the id of each row. Defaults to False.

        Returns:
            A table with the first n rows of this table.
        """
        i = 0
        l = []
        for id, index in self.id_index.items():
            row = self[index]
            if print_id:
                l.append((id, row))
            else:
                l.append(row)

            i += 1
            if i == n:
                break

        return l

    def sample_many(self, n=10, print_id=False):
        """
        Get n random rows of this table.

        Args:
            n (int, optional): The number of rows to get. Defaults to 10.

            print_id (bool, optional): Whether to print the id of each row. Defaults to False.

        Returns:
            A list of n random rows of this table.
        """
        l = []
        for id, index in random.sample(list(self.id_index.items()), n):
            row = self[index]
            if print_id:
                l.append((id, row))
            else:
                l.append(row)
        return l
    
    def sample(self, print_id=False):
        """
        Get a random row of this table.

        Args:
            print_id (bool, optional): Whether to print the id of the row. Defaults to False.

        Returns:
            A random row of this table.
        """
        id, index = random.choice(list(self.id_index.items()))
        row = self[index]
        if print_id:
            return (id, row)
        else:
            return row
