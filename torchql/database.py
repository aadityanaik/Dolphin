import copy
from time import time
from typing import Any, Callable, Dict, List
import pickle
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import product
from tqdm import tqdm
import torch

from .table import Table
from .utils import IndexedDataloader, Operation

class TableCollection(dict):
    __getattr__ = dict.get
    # __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def register_dataset(self, d, name: str, num_workers: int = None, batch_size=None, disable=True) -> None:
        """
        Register a dataset with the database. This will create a Table from the dataset and store it in the database.

        Args:
            d (Dataset): The dataset to register.
            
            name (str): The name of the dataset.

            num_workers (int): The number of workers to use for the dataloader. Defaults to 16.

            batch_size (int): The batch size to use for the dataloader. If it is None, the batch_size will be expected to be 1.

            disable (bool): Whether to disable the progress bar. Defaults to False.

        """
        if name in self:
            # print(f"A dataset with name '{name}' already exists. Overwriting it.")
            self.pop(name, f"Table {name} not found")

        if isinstance(d, Table):
            self[name] = d
        else:
            #TODO Check this and see why this is needed
            if num_workers is not None:
                d = DataLoader(d, shuffle=False, batch_size=batch_size, num_workers=num_workers)
            samples = [sample for sample in tqdm(d, desc=f"Loading data for {name}", disable=disable)]
            if batch_size is not None:
                self[name] = Table(samples).flatten()
            else:
                self[name] = Table(samples)

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.register_dataset(__value, __name)

class Database(object):
    """
    A database is a collection of Tables over which queries can be executed.
    Each Table is a dataset, or a collection of samples.
    """
    def __init__(self, name: str = "mldb") -> None:
        object.__setattr__(self, 'tables', TableCollection())
        object.__setattr__(self, 'T', self.tables)
        object.__setattr__(self, 'name', name)
        # self.tables = TableCollection()
        # self.T = self.tables
        # self.name = name

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.register_dataset(__value, __name)

    def __getattr__(self, __name: str) -> Any:
        return self.tables[__name]

    def register_dataset(self, d, name: str, num_workers: int = None, batch_size=None, disable=True) -> None:
        """
        Register a dataset with the database. This will create a Table from the dataset and store it in the database.

        Args:
            d (Dataset): The dataset to register.
            
            name (str): The name of the dataset.

            num_workers (int): The number of workers to use for the dataloader. Defaults to 16.

            batch_size (int): The batch size to use for the dataloader. If it is None, the batch_size will be expected to be 1.

            disable (bool): Whether to disable the progress bar. Defaults to False.
        """
        self.tables.register_dataset(d, name, num_workers, batch_size, disable)

    def store(self, path: str, which_tables: List[str] = None):
        """
        Store the database to disk.

        Args:
            path (str): The path to store the database to.
            
            which_tables (List[str]): The list of table names to store. If it is None, all tables in the database will be saved.
        """
        # outdir = os.path.dirname(path)
        # dbname = os.path.basename(path)
        if which_tables:
            save_items = []

            for name in which_tables:
                try:
                    save_items.append((name, self.tables[name]))
                except:
                    print(f"The table with name '{name}' does not exist.")
        else:
            save_items = self.tables.items()

        os.makedirs(path, exist_ok=True)
        for name, table in save_items:
            print(f"Storing table {name}")
            table_path = os.path.join(path, name + ".pt")
            # with open(table_path, 'wb') as dump:
            #     pickle.dump(table, dump)
            torch.save(table, table_path)

    def load(self, path: str, which_tables: List[str] = None):
        """
        Load a database from disk.

        Args:
            path (str): The path from which to load the database.

            which_tables (List[str]): The list of table names to load. If it is None, all files with the .pt extension will be loaded.
        """
        assert os.path.exists(path)
        for table_path in os.listdir(path):
            if table_path.endswith(".pt"):
                table_name = table_path[:-3]
                print(f"Loading table {table_name}")
                
                if (which_tables is None) or (table_name in which_tables):
                    table = torch.load(os.path.join(path, table_path))
                    print(f"Loaded table {table_name}")
                    self.tables[table_name] = table

    def execute_pipeline(self, pipeline: List[Operation], name: str = "default_pipeline", **kwargs) -> Table:
        """
        Execute a query pipeline over the database.
        The pipeline must be a list of operations, such as the pipeline from the torchql.Query class where the first
        operation must be a register operation.

        Args:
            pipeline (List[Operation]): The pipeline to execute.

            name (str): The name of the pipeline. Defaults to "default_pipeline".

            kwargs: Global options that override the options local to the operations in the query pipeline.

        Returns:
            A Table containing the result of the query.
        """
        # print(self.tables)
        # print(pipeline[0].arg)
        
        assert pipeline[0].op == "register", f"The first operation in any pipeline must register a table: {pipeline[0].op}"
        assert pipeline[0].arg in self.tables, f"Any registered table for the query must be first registered in the database: {pipeline[0].arg}"

        first_table = self.tables[pipeline[0].arg]
        result = first_table
        
        for operation in pipeline[1:]:
            t = time()
            op = operation.op
            arg = operation.arg
            op_kwargs = operation.kwargs if operation.kwargs is not None else {}

            for k in op_kwargs:
                if k in kwargs:
                    op_kwargs[k] = kwargs[k]

            if op == 'register' or op == 'union' or op == 'intersect':
                arg = self.tables[arg]
            elif op == 'join':
                assert arg is not None
                # Note: the fourth argument of join is disable instead of batch
                result = getattr(result, op)(self.tables[arg], **op_kwargs)
            else:
                if arg is None:
                    result = getattr(result, op)(**op_kwargs)
                elif type(arg) == tuple:
                    result = getattr(result, op)(*arg, **op_kwargs) 
                else:
                    result = getattr(result, op)(arg, **op_kwargs)
            # print("Operation", op, "took", time() - t, "seconds")
        
        if name in self.tables:
            self.tables.pop(name)
            
        self.tables[name] = result
        return self.tables[name]

        
