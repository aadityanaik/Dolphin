from __future__ import annotations
import itertools
import time
from typing import Dict, Tuple
import numpy as np
from torchql.database import Database
from torchql.query import Query, QuerySuite
from torchql.table import Table
from .sqr import SQR

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import torch

def get_stat_table(rule: Tuple, stats) -> Table:
    t = stats[rule[0]]

    for r in rule[1:]:
        tp = stats[r]
        t = t.join(tp, key=lambda idx, *row: idx[0] if isinstance(idx, tuple) else idx)

    return t

def project_stat_table(rule: Tuple, stats, stat_table, batch_size=0) -> Table:
    stat_ids = [ stats[r] for r in rule ]
    if batch_size == 0:
        return stat_table.project(lambda *row: tuple(row[i] for i in stat_ids))
    else:
        def get_row_batch(*row_batch):
            return torch.stack(tuple((row_batch[i] for i in stat_ids)), 1).tolist()
        new_tab = stat_table.project(get_row_batch, batch_size=batch_size)
        # print(new_tab.head(1))
        return new_tab

def get_stat_table_single_arg(args) -> Table:
    return get_stat_table(*args)

def project_stat_table_single_arg(args) -> Table:
    return project_stat_table(*args)

def evaluate_sqr(args, batch_size=0):
    # sqr, ords, cats = args
    # stat_tab = get_stat_table(sqr.rule, ords, cats)
    sqr, stat_tab = args
    def run(*row):
        if batch_size <= 0:
            moved_row = row
            return (*row, sqr(*moved_row))
        else:
            moved_row = row
            return torch.stack((*row, sqr(*moved_row)), 1).tolist()

    return stat_tab.project(run, batch_size=batch_size)

def stat_union(stats):
    t = None
    i = 0
    stat_columns = {}
    for statname, stat in tqdm(stats.items(), desc='Joining stats'):
        if t is None:
            t = stat
        else:
            t = t.join(stat, key=lambda idx, *row: idx[-1] if isinstance(idx, tuple) else idx)
        stat_columns[statname] = i
        i += 1

    return t, stat_columns

class SQRL:
    def __init__(self, db: Database, ords={}, cats={}, t=99, grouping=2) -> None:
        self.db = db
        self.ord_fns = ords
        self.cat_fns = cats
        self.t = t
        self.grouping = grouping
        self.ords = {}
        self.cats = {}
        self.stats = {}
        self.sqrs = {}
        self.queries = []

    def generate(self, tablename: str, batch_size=0) -> SQRL:
        sqrl = SQRL(self.db, self.ord_fns, self.cat_fns, self.t, self.grouping)
        print(f"Generating SQRL rules over {tablename}...")

        sqrl.gen_table = sqrl.db.tables[tablename]
        sqrl._calculate_ords(sqrl.gen_table, batch_size=batch_size)
        sqrl._calculate_cats(sqrl.gen_table, batch_size=batch_size)
        sqrl.stats = {**sqrl.ords, **sqrl.cats}

        sqrl.sqrs = sqrl._generate_concrete_rules(batch_size=batch_size)

        return sqrl

    def eval(self, table, batch_size = 0) -> Table:
        if not isinstance(table, Table):
            table = self.db.tables[table]

        ords = {}
        cats = {}
        for ordname, ord in tqdm(self.ord_fns.items(), desc="Calculating ord stats"):
            ords[ordname] = table.project(ord, batch_size=batch_size)

        for catname, cat in tqdm(self.cat_fns.items(), desc="Calculating cat stats"):
            cats[catname] = table.project(cat, batch_size=batch_size)

        stats = {**ords, **cats}
        full_stat_table, stat_columns = stat_union(stats)

        results = {}
        for sqr in tqdm(self.sqrs.values()):
            stat_tab = project_stat_table(sqr.rule, stat_columns, full_stat_table, batch_size=batch_size)
            results[str(sqr)] = evaluate_sqr((sqr, stat_tab), batch_size=batch_size)
        
        
        return results

    def _calculate_ords(self, table: Table, batch_size=0) -> None:
        for ordname, ord in tqdm(self.ord_fns.items(), desc="Calculating ord stats"):
            self.ords[ordname] = table.project(ord, batch_size=batch_size)

    def _calculate_cats(self, table: Table, batch_size = 0) -> None:
        for catname, cat in tqdm(self.cat_fns.items(), desc="Calculating cat stats"):
            self.cats[catname] = table.project(cat, batch_size=batch_size)

    def _group_quartiles(self, val, quartiles):
        if val < quartiles[0]:
            return (-np.infty, quartiles[0])
        elif val < quartiles[1]:
            return (quartiles[0], quartiles[1])
        elif val < quartiles[2]:
            return (quartiles[1], quartiles[2])
        else:
            return (quartiles[2], np.infty)

    def _calculate_bounds(self, tab: Table, batch_size=0) -> None:
        threshold = 100 - ((100 - self.t) / 2)

        if len(tab) == 0:
            return ()
        
        # print(len(tab), tab.rows[0], tab.rows[0][0])
        tab = tab.order_by(lambda *row: row[0], batch_size=batch_size)

        # print(tab.head(), len(tab))
        if len(tab[0]) == 1:
            two_sided_lower_bound = np.percentile(tab, 100 - threshold)
            two_sided_upper_bound = np.percentile(tab, threshold)

            return Table((two_sided_lower_bound, two_sided_upper_bound))
        else:
            ord = tab.project(lambda *row: row[0], batch_size=batch_size) # .order_by(lambda *row: row)
            quartiles = np.percentile(ord, [25, 50, 75])

            if batch_size == 0:
                grouped = tab.group_reduce(
                    lambda *row: self._group_quartiles(row[0], quartiles),
                    lambda table: self._calculate_bounds(table.project(lambda *row: row[1:])),
                    disable=True
                )
            else:
                grouped = tab.group_reduce(
                    lambda *row: self._group_quartiles(row[0], quartiles),
                    lambda table: self._calculate_bounds(table.project(lambda *row_batch: torch.stack(row_batch[1:], 1).tolist(), batch_size=batch_size)),
                    disable=True
                )

            return grouped
    
    def _generate_abstract_rules(self):
        perms = []
        for catname in self.cats.keys():
            for i in range(1, self.grouping + 1):
                for rbody in list(itertools.permutations(list(self.ords.keys()), i)):
                    perms.append((catname, *rbody))

        return perms
    
    def _generate_concrete_rules(self, batch_size=0):
        print("Generating Abstract Rules...")
        abstract_rules = self._generate_abstract_rules()
        print(f"Generated {len(abstract_rules)} abstract rules")

        print("Generating Concrete Rules...")
        concrete_rules = {}

        print("Tracking time...")

        t_proj = 0
        t_gr = 0

        full_stat_table, stat_columns = stat_union(self.stats)

        # def gen_concrete_rule(idx_rule_pair):
        for idx, rule in tqdm(list(enumerate(abstract_rules)), desc="Generating concrete rules"):
            # global t_proj, t_gr
            # idx, rule = idx_rule_pair
            qname = f"rule_{idx} :: {rule}"
            # print(qname)
            
            # stat_table = get_stat_table(rule, self.stats)
            ts = time.time()
            stat_table = project_stat_table(rule, stat_columns, full_stat_table, batch_size=batch_size)
            t_proj += time.time() - ts
            # print(len(stat_table))

            ts = time.time()
            if batch_size <= 0:
                bound_table = stat_table.group_reduce(
                    lambda *row: row[0],
                    lambda table: self._calculate_bounds(table.project(lambda *row: row[1:], batch_size=batch_size)),
                    disable=True)
            else:
                bound_table = stat_table.group_reduce(
                    lambda *row: row[0],
                    lambda table: self._calculate_bounds(table.project(lambda *row_batch: torch.stack(row_batch[1:], 1).tolist(), batch_size=batch_size)),
                    disable=True)
            t_gr += time.time() - ts

            # return qname, SQR(rule, bound_table) # , self.ord_fns, self.cat_fns)
            concrete_rules[qname] = SQR(rule, bound_table, batchwise=True if batch_size > 0 else False)

        # idx_rule_pairs = list(enumerate(abstract_rules))

        # for idx, rule in tqdm(idx_rule_pairs, desc="Generating concrete rules"):
        #     qname, sqr = gen_concrete_rule((idx, rule))
        #     concrete_rules[qname] = sqr

        print("Finished generating concrete rules")
        print(f"Projection time: {t_proj}")
        print(f"Grouping time: {t_gr}")
        return concrete_rules
    
    

            

            