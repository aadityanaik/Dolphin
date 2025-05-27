import copy
from typing import Callable, Dict, List
import pickle
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from itertools import product
from tqdm import tqdm
from datasets import Dataset as HFDataset
import tarfile

# from .table import Table
from .table_lazy import Table, SavedTable
from .utils import IndexedDataloader, Operation

class Database:
    """
    A database is a collection of Tables over which queries can be executed.
    Each Table is a dataset, or a collection of samples.
    """
    def __init__(self, name: str = "mldb") -> None:
        # self.datasets : Dict[str, Dataset] = {}
        self.tables : Dict[str, Table] = {}
        self.name = name

    def register_dataset(self, d, name: str):
        """
        Register a dataset with the database. This will create a Table from the dataset and store it in the database.

        Args:
            d (Dataset): The dataset to register.
            name (str): The name of the dataset.
        """
        assert name not in self.tables, f"Table {name} already exists"
        # self.datasets[name] = d
        if isinstance(d, Table):
            self.tables[name] = copy.copy(d)
        else:
            self.tables[name] = Table(d)

    def store(self, path: str):
        """
        Store the database to disk.

        Args:
            path (str): The path to store the database to.
        """
        outdir = os.path.dirname(path)
        dbname = os.path.basename(path)
        os.makedirs(outdir, exist_ok=True)
        for name, table in self.tables.items():
            # Output the table to disk
            print(f"Storing table {name}")
            SavedTable(path, name, table=table)

    def load(self, path: str):
        """
        Load a database from disk.
        """
        assert os.path.exists(f"{path}.tar"), f"Database {path} does not exist"
        with tarfile.open(f"{path}.tar", "r") as tar:
            all_tables = set([os.path.dirname(n) for n in tar.getnames()])
            for name in all_tables:
                self.tables[name] = SavedTable(path, name)
        return self

    def execute_pipeline(self, pipeline: List[Operation], name: str = "default_pipeline") -> Table:
        """
        Execute a query pipeline over the database.
        The pipeline must be a list of operations, such as the pipeline from the torchql.Query class where the first
        operation must be a register operation.

        Args:
            pipeline (List[Operation]): The pipeline to execute.
            name (str): The name of the pipeline. Defaults to "default_pipeline".

        Returns:
            A Table containing the result of the query.
        """
        assert pipeline[0].op == "register", "The first operation in any pipeline must register a table"
        assert pipeline[0].arg in self.tables, "Any registered table for the query must be first registered in the database"

        first_table = self.tables[pipeline[0].arg]
        print("Initializing base")
        # assert(isinstance(first_table, Table))
        result = copy.copy(first_table)
        # else:
        #     assert isinstance(first_table, Dataset)
        #     print("Will need iterations")
        #     result = Table([ sample for sample in tqdm(first_table) ])
        print('Base initialized')
        
        for operation in pipeline[1:]:
            op = operation.op
            arg = operation.arg

            if op == 'register':
                arg = self.tables[arg]
            elif op == 'join':
                assert arg is not None
                result = getattr(result, op)(self.tables[arg[0]], key=arg[1], fkey=arg[2], fuzzy=arg[3])
            elif op == 'order_by':
                assert arg is not None
                result = getattr(result, op)(arg[0], reverse=arg[1])
            else:
                result = getattr(result, op)(arg) if arg is not None else getattr(result, op)()
        
        self.tables[name] = result
        return self.tables[name]

        
