from __future__ import annotations
import sys
from typing import Any, Callable, Dict, List, Tuple
import copy
from torch.nn import Module

from .database import Database, Table
from .utils import Operation

import importlib.metadata
from tqdm import tqdm

class Query(Module):
    """
    A query object that can be used to build a query pipeline and execute it over a database.
    """
    def __init__(self, name, base=None) -> None:
        super().__init__()
        self.name = name
        self.tables: List[str] = []
        self.conditions: List[Callable[..., bool]] = []
        self.columns: Callable[..., List] = []
        self.pipeline: List[Operation] = []
        self.cached = False

        if base is not None:
            self.tables.append(base)
            self.pipeline.clear()
            self.pipeline.append(Operation("register", base, None))
            self.cached = False
    
    def base(self, tablename: str) -> Query:
        """
        Set the base table over which the query pipeline will operate. This will simply reset the base, but maintain the
        rest of the operations in the pipeline. This is useful for when one wants to run the same query over different
        tables.

        Args:
            tablename (str): The name of the table to set as the base.

        Returns:
            A new query object with the base table set.
        """

        q = copy.deepcopy(self)
        if len(q.pipeline) > 0 and q.pipeline[0].op == "register":
            q.pipeline[0] = Operation("register", tablename, None)
            q.tables[0] = tablename
        else:
            q.pipeline.insert(0, Operation("register", tablename, None))
            q.tables.insert(0, tablename)

        q.cached = False

        return q
    
    def join(self, tablename: str, key=None, fkey=None, batch_size=None, disable=True, prov=False) -> Query:
        """
        Join a table. This will register a join operation to the pipeline.
        Rows from the query table and the table to join will be joined on the key and foreign key, respectively.
        If no key or foreign key is provided, the index is used.
        Otherwise, the key and foreign key must be functions that take a set of columns from a row as input and return
        hashable values that serve as a key.
        Rows are joined if the result of the key and foreign key functions are equal.
        
        Args:
            tablename (str): The name of the table to join

            key (Callable[..., Any]): The key to join on. Defaults to None, in which case the index is used.
                Must take a set of columns from a row as input and return a hashable value that serves as a key.

            fkey (Callable[..., Any]): The foreign key to join on. Defaults to None, in which case the index is used.
                Must take a set of columns from a row as input and return a hashable value that serves as a key.

        Returns:
            A new query object with the join operation registered.
        """
        q = copy.deepcopy(self)
        q.tables.append(tablename)
        q.pipeline.append(Operation("join", tablename, {'key': key, 'fkey': fkey, 'batch_size': batch_size, 'disable': disable, 'prov': prov}))

        return q
    
    def union(self, tablename: str, batch_size=None, disable=True) -> Query:
        """
        Union a table. This will register a union operation to the pipeline.
        Rows of the other table will be added to the bottom of the table.
        
        Args:
            tablename (str): The name of the table to union

        Returns:
            A new query object with the union operation registered.
        """
        q = copy.deepcopy(self)
        q.tables.append(tablename)
        q.pipeline.append(Operation("union", tablename, {'batch_size': batch_size, 'disable': disable}))
        # q.pipeline.append(Operation("union", (tablename,batch_size, disable)))

        return q
    
    def intersect(self, tablename: str, batch_size=None, disable=True) -> Query:
        """
        Intersect a table. This will register a intersect operation to the pipeline.
        Rows that are common to both tables will be the result.
        
        Args:
            tablename (str): The name of the table to intersect

        Returns:
            A new query object with the intersect operation registered.
        """
        q = copy.deepcopy(self)
        q.tables.append(tablename)
        q.cached = False
        q.pipeline.append(Operation("intersect", tablename, {'batch_size': batch_size, 'disable': disable}))
        # q.pipeline.append(Operation("intersect", tablename, {'batch_size': batch_size, 'disable': disable}))

        return q

    def filter(self, cond: Callable[..., bool],batch_size=None, disable=True) -> Query:
        """
        Filter rows based on a condition. This will register a filter operation to the pipeline.

        Args:
            cond (Callable[..., bool]): The condition to filter rows on.
                Must take a set of columns from a row as input and return a boolean.

        Returns:
            A new query object with the filter operation registered.
        """
        q = copy.deepcopy(self)
        q.conditions.append(cond)
        q.cached = False
        q.pipeline.append(Operation("filter", cond, {'batch_size': batch_size, 'disable': disable}))

        return q

    def project(self, cols: Callable[..., List], batch_size = None, disable=True) -> Query:
        """
        Select columns or perform a function on columns of the Table.
        This will register a projection operation to the pipeline.

        Args:
            project (Callable[..., List]): A function that takes a row as input and returns a list of columns to select.

        Returns:
            A new query object with the projection operation registered.
        """
        q = copy.deepcopy(self)
        q.columns = cols
        q.cached = False
        q.pipeline.append(Operation("project", cols, {'batch_size': batch_size, 'disable': disable}))

        return q
    
    def flatten(self, batch_size=None, disable=True) -> Query:
        """
        Flatten the result of the query. This will register a flatten operation to the pipeline.
        This applies when a Table's rows are iterables.
        This flattens each iterable into a set of rows for the table.

        Returns:
            A new query object with the flatten operation registered.
        """
        q = copy.deepcopy(self)
        q.cached = False
        q.pipeline.append(Operation("flatten", None, {'batch_size': batch_size, 'disable': disable}))

        return q
    
    def unique(self, batch_size=None, disable=True) -> Query:
        q = copy.deepcopy(self)
        q.cached = False
        q.pipeline.append(Operation("unique", None, {'batch_size': batch_size, 'disable': disable}))

        return q
    
    def order_by(self, key: Callable[..., Any], reverse: bool = False, batch_size=None, disable=True) -> Query:
        """
        Order the rows of the table by a key.

        Args:
            key (Callable[..., Any]): The key to order rows by.
            
            reverse (bool): Whether to reverse the order (order in the descending order). Defaults to False.

        Returns:
            A new query object with the order_by operation registered.
        """
        q = copy.deepcopy(self)
        q.cached = False
        q.pipeline.append(Operation("order_by", key, {'reverse': reverse, 'batch_size': batch_size, 'disable': disable}))

        return q
    
    def group_by(self, key: Callable[..., Any], batch_size=None, disable=True) -> Query:
        """
        Group the rows of the table by a key.
        Results in a Table with two columns: the key and a list of rows that share the key.

        Args:
            key (Callable[..., Any]): The key to group rows by.
                Must take a set of columns from a row as input and return a value that serves as a key.
                Rows are grouped if the result of the key function is equal.

        Returns:
            A new query object with the group_by operation registered.
        """
        q = copy.deepcopy(self)
        q.cached = False
        q.pipeline.append(Operation("group_by", key, {'batch_size': batch_size, 'disable': disable}))

        return q

    def group_by_with_index(self, key: Callable[..., Any], batch_size=None, disable=True) -> Query:
        """
        Group the rows of the table by a key.
        Results in a Table with two columns: the key and a list of rows that share the key.

        Args:
            key (Callable[..., Any]): The key to group rows by.
                Must take a set of columns from a row as input and return a value that serves as a key.
                Rows are grouped if the result of the key function is equal.

        Returns:
            A new query object with the group_by_with_index operation registered.
        """
        q = copy.deepcopy(self)
        q.cached = False
        q.pipeline.append(Operation("group_by_with_index", key, {'batch_size': batch_size, 'disable': disable}))

        return q
    
    def reduce(self, reduction: Callable[..., Any]) -> Query:
        """
        Reduce the rows of the table using a reduction function.

        Args:
            reduction (Callable[..., Any]): The reduction to apply to the rows.
                Must take a set of columns from a row as input and return a value.

        Returns:
            A new query object with the reduction operation registered.
        """
        q = copy.deepcopy(self)
        q.cached = False
        q.pipeline.append(Operation("reduce", reduction, None))

        return q
    
    def group_reduce(self, key: Callable[..., Any], reduction: Callable[..., Any], batch_size=None, disable=True) -> Query:
        """
        Group the rows of the table by a key and reduce each group using a reduction function.
        Results in a Table with two columns: the key and the result of the reduction function on the group.

        Args:
            key (Callable[..., Any]): The key to group rows by.
                Must take a set of columns from a row as input and return a value that serves as a key.
                Rows are grouped if the result of the key function is equal.

            reduction (Callable[..., Any]): The reduction to apply to the rows.
                Must take a set of columns from a row as input and return a value.

        Returns:
            A new query object with the group_reduce operation registered.
        """

        q = copy.deepcopy(self)
        q.cached = False
        q.pipeline.append(Operation("group_reduce", key, {'reduction': reduction, 'batch_size': batch_size, 'disable': disable}))

        return q

    def head(self, num_rows):
        q = copy.deepcopy(self)
        q.pipeline.append(Operation("head", num_rows, None))

        return q

    def cache(self):
        q = copy.deepcopy(self)
        q.pipeline.append(Operation("cache", None, None))

        return q

    def rename(self, name: str) -> Query:
        """
        Clone the query object.

        Args:
            name (str): The name of the new query object.

        Returns:
            A new query object with the same pipeline as the original.
        """
        q = copy.deepcopy(self)
        q.cached = False
        q.name = name if name is not None else self.name

        return q

    def run(self, database: Database, **kwargs) -> Table:
        """
        Execute the query pipeline over a database.

        Args:
            database (Database): The database to execute the query over.
            **kwargs: Key-word arguments to be passed to individual operations in the query.
                      These key-word arguments will override the options set while defining the query operations.

        Returns:
            The table object with the result of the query stored in the results attribute.
        """
        assert self.pipeline, "No operations registered. Register operations using the register, join, filter, cols, order_by, and group_by functions"
        assert self.tables, "At least one table should be registered as the base when running the query"
        t = database.execute_pipeline(self.pipeline, self.name, **kwargs)
        self.cached = True

        return t
    
    def __repr__(self):
        return f"Query({self.name})"
    
    def __str__(self):
        return f"Query({self.name})"
    
    def __len__(self):
        return len(self.pipeline)
    
    # if object is called as a function
    def forward(self, database: Database, cache=True, **kwargs) -> Table:
        """
        Execute the query pipeline over a database.
        Add argument `tensors={ "x": x, "y": y }` if intending to run a query over a tensor within a neural model.

        Args:
            database (Database): The database to execute the query over.
            **kwargs: Key-word arguments to be passed to individual operations in the query.
                      These key-word arguments will override the options set while defining the query operations.

        Returns:
            The table object with the result of the query stored in the results attribute.
        """
        if "tensors" in kwargs:
            for key, value in kwargs['tensors'].items():
                database.tables.pop(key, f"Table {key} not found")
                # print(f"Registering dataset {key} with {value}")
                database.register_dataset(value, key)

        if cache and self.cached and self.name in database.tables:
            return database.tables[self.name]
        t = self.run(database, **kwargs)

        if "tensors" in kwargs:
            # cleanup
            for key, value in kwargs['tensors'].items():
                database.tables.pop(key, f"Table {key} not found")
                database.register_dataset(value, key)
            database.tables.pop(self.name, f"Table {self.name} not found")

        return t


class QuerySuite:
    """
    This class allows for easier management of a collection of queries under a common category.
    One can add queries to the suite and run them all together.
    """

    def __init__(self, name, queries=[]) -> None:
        self.name = name
        self.queries: List[Query] = queries

    def __copy__(self):
        return type(self)(self.name, dict(self.queries))

    def add(self, *queries) -> QuerySuite:
        """
        Add queries to the suite.

        Args:
            queries (Query): The queries to add to the suite.
        """

        qs = self.__copy__()

        for q in queries:
            qs.queries[q.name] = q

        return qs
    
    def __getattr__(self, __name: str) -> QuerySuite:
        def wrapper(*args, **kwargs):
            qs = QuerySuite(self.name)
            for q in tqdm(self.queries, desc=f'Applying `{__name}` to queries'):
                qs.queries.append(getattr(q, __name)(*args, **kwargs))

            return qs

        return wrapper
            

    def run(self, database: Database) -> Table:
        """
        Run all queries in the suite over a database.

        Args:
            database (Database): The database to run the queries over.
        """
        tables = []
        for query in self.queries:
            table = query(database)
            tables.append((query.name, table))

        return Table(tables)
    
    def __call__(self, database: Database) -> Dict[str, Table]:
        return self.run(database)