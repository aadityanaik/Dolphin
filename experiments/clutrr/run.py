import os
from argparse import ArgumentParser
from typing import Tuple
from tqdm import tqdm
import csv
import re
import random
import transformers
import wandb

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel
import math
import datetime
from time import time

from dolphin.provenances import get_provenance
from dolphin.distribution import Distribution
from torchql import Table

relation_id_map = {
    "daughter": 0,
    "sister": 1,
    "son": 2,
    "aunt": 3,
    "father": 4,
    "husband": 5,
    "granddaughter": 6,
    "brother": 7,
    "nephew": 8,
    "mother": 9,
    "uncle": 10,
    "grandfather": 11,
    "wife": 12,
    "grandmother": 13,
    "niece": 14,
    "grandson": 15,
    "son-in-law": 16,
    "father-in-law": 17,
    "daughter-in-law": 18,
    "mother-in-law": 19,
    "nothing": 20,
}

transitive_map = {
    (0, 0): 6,
    (0, 1): 0,
    (0, 2): 15,
    (0, 3): 1,
    (0, 4): 5,
    (0, 5): 16,
    (0, 7): 2,
    (0, 9): 12,
    (0, 10): 7,
    (0, 11): 17,
    (0, 13): 19,
    (1, 0): 14,
    (1, 1): 1,
    (1, 2): 8,
    (1, 3): 3,
    (1, 4): 4,
    (1, 7): 7,
    (1, 9): 9,
    (1, 10): 10,
    (1, 11): 11,
    (1, 13): 13,
    (2, 0): 6,
    (2, 1): 0,
    (2, 2): 15,
    (2, 3): 1,
    (2, 4): 5,
    (2, 7): 2,
    (2, 9): 12,
    (2, 10): 7,
    (2, 11): 4,
    (2, 12): 18,
    (2, 13): 9,
    (3, 1): 3,
    (3, 4): 11,
    (3, 7): 10,
    (3, 9): 13,
    (4, 0): 1,
    (4, 1): 3,
    (4, 2): 7,
    (4, 4): 11,
    (4, 7): 10,
    (4, 9): 13,
    (4, 12): 9,
    (5, 0): 0,
    (5, 2): 2,
    (5, 4): 17,
    (5, 6): 6,
    (5, 9): 19,
    (5, 15): 15,
    (6, 1): 6,
    (6, 7): 15,
    (7, 0): 14,
    (7, 1): 1,
    (7, 2): 8,
    (7, 3): 3,
    (7, 4): 4,
    (7, 7): 7,
    (7, 9): 9,
    (7, 10): 10,
    (7, 11): 11,
    (7, 13): 13,
    (8, 1): 14,
    (8, 7): 8,
    (9, 0): 1,
    (9, 1): 3,
    (9, 2): 7,
    (9, 4): 11,
    (9, 5): 4,
    (9, 7): 10,
    (9, 9): 13,
    (9, 17): 11,
    (9, 19): 13,
    (10, 1): 3,
    (10, 4): 11,
    (10, 7): 10,
    (10, 9): 13,
    (11, 12): 13,
    (12, 0): 0,
    (12, 2): 2,
    (12, 4): 17,
    (12, 6): 6,
    (12, 9): 19,
    (12, 15): 15,
    (12, 16): 16,
    (12, 17): 4,
    (12, 18): 18,
    (12, 19): 9,
    (13, 5): 11,
    (15, 1): 6,
    (15, 7): 15,
}


class CLUTRRDataset:
    def __init__(self, root, dataset, split, data_percentage):
        self.dataset_dir = os.path.join(root, f"clutrr/{dataset}/")
        self.file_names = [
            os.path.join(self.dataset_dir, d)
            for d in os.listdir(self.dataset_dir)
            if f"_{split}.csv" in d
        ]
        self.data = [
            row for f in self.file_names for row in list(csv.reader(open(f)))[1:]
        ]
        self.data_num = math.floor(len(self.data) * data_percentage / 100)
        self.data = self.data[: self.data_num]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # Context is a list of sentences
        context = [
            s.strip().lower() for s in self.data[i][2].split(".") if s.strip() != ""
        ]

        # Query is of type (sub, obj)
        query_sub_obj = eval(self.data[i][3])
        query = (query_sub_obj[0].lower(), query_sub_obj[1].lower())

        # Answer is one of 20 classes such as daughter, mother, ...
        answer = self.data[i][5]
        return ((context, query), answer)

    @staticmethod
    def collate_fn(batch):
        queries = [query for ((_, query), _) in batch]
        contexts = [fact for ((context, _), _) in batch for fact in context]
        context_lens = [len(context) for ((context, _), _) in batch]
        context_splits = [
            (sum(context_lens[:i]), sum(context_lens[: i + 1]))
            for i in range(len(context_lens))
        ]
        answers = torch.stack(
            [torch.tensor(relation_id_map[answer]) for (_, answer) in batch]
        )
        return ((contexts, queries, context_splits), answers)


def clutrr_loader(root, dataset, batch_size, training_data_percentage):
    train_dataset = CLUTRRDataset(root, dataset, "train", training_data_percentage)
    train_loader = DataLoader(
        train_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True
    )
    test_dataset = CLUTRRDataset(root, dataset, "test", 100)
    test_loader = DataLoader(
        test_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True
    )
    return (train_loader, test_loader)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        out_dim: int,
        num_layers: int = 0,
        softmax=False,
        normalize=False,
        sigmoid=False,
    ):
        super(MLP, self).__init__()
        layers = []
        layers += [nn.Linear(in_dim, embed_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
        layers += [nn.Linear(embed_dim, out_dim)]
        self.model = nn.Sequential(*layers)
        self.softmax = softmax
        self.normalize = normalize
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.model(x)
        if self.softmax:
            x = nn.functional.softmax(x, dim=1)
        if self.normalize:
            x = nn.functional.normalize(x)
        if self.sigmoid:
            x = torch.sigmoid(x)
        return x


class CLUTRRModel(nn.Module):
    def __init__(
        self,
        device="cpu",
        num_mlp_layers=1,
        debug=False,
        no_fine_tune_roberta=False,
        use_softmax=False,
        provenance="damp",
        sample_k=None,
        top_k=1,
    ):
        super(CLUTRRModel, self).__init__()

        # Options
        self.device = device
        self.debug = debug
        self.no_fine_tune_roberta = no_fine_tune_roberta

        # Roberta as embedding extraction model
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", local_files_only=False, add_prefix_space=True
        )
        self.roberta_model = RobertaModel.from_pretrained("roberta-base")

        # self.tokenizer = AutoTokenizer.from_pretrained('second-state/All-MiniLM-L6-v2-Embedding-GGUF')
        # self.roberta_model = AutoModel.from_pretrained('second-state/All-MiniLM-L6-v2-Embedding-GGUF')

        self.embed_dim = self.roberta_model.config.hidden_size

        # Entity embedding
        self.relation_extraction = MLP(
            self.embed_dim * 3,
            self.embed_dim,
            len(relation_id_map),
            num_layers=num_mlp_layers,
            sigmoid=not use_softmax,
            softmax=use_softmax,
        )

        Distribution.provenance = get_provenance(provenance)
        Distribution.provenance.k = top_k
        Distribution.k = sample_k

    def _preprocess_contexts(self, contexts, context_splits):
        clean_context_splits = []
        clean_contexts = []
        name_token_indices_maps = []
        for _, (start, end) in enumerate(context_splits):
            skip_next = False
            skip_until = 0
            curr_clean_contexts = []
            curr_name_token_indices_maps = []
            for j, sentence in zip(range(start, end), contexts[start:end]):
                # It is possible to skip a sentence because the previous one includes the current one.
                if skip_next:
                    if j >= skip_until:
                        skip_next = False
                    continue

                # Get all the names of the current sentence
                names = re.findall("\\[(\w+)\\]", sentence)

                # Check if we need to include the next sentence(s) as well
                num_sentences = 1
                union_sentence = f"{sentence}"
                for k in range(j + 1, end):
                    next_sentence = contexts[k]
                    next_sentence_names = re.findall("\\[(\w+)\\]", next_sentence)
                    if len(names) == 1 or len(next_sentence_names) == 1:
                        if len(next_sentence_names) > 0:
                            num_sentences += 1
                            union_sentence += f". {next_sentence}"
                            names += next_sentence_names
                        skip_next = True
                        if len(next_sentence_names) == 1:
                            skip_until = k - 1
                        else:
                            skip_until = k
                    else:
                        break

                # Deduplicate the names
                names = set(names)

                # Debug number of sentences
                if self.debug and num_sentences > 1:
                    print(
                        f"number of sentences: {num_sentences}, number of names: {len(names)}; {names}"
                    )
                    print("Sentence:", union_sentence)

                # Then split the context by `[` and `]` so that names are isolated in its own string
                splitted = [
                    u.strip()
                    for t in union_sentence.split("[")
                    for u in t.split("]")
                    if u.strip() != ""
                ]

                # Get the ids of the name in the `splitted` array
                is_name_ids = {
                    s: [j for (j, sp) in enumerate(splitted) if sp == s] for s in names
                }

                # Get the splitted input_ids
                splitted_input_ids_raw = self.tokenizer(splitted).input_ids
                splitted_input_ids = [
                    (
                        ids[:-1]
                        if j == 0
                        else (
                            ids[1:]
                            if j == len(splitted_input_ids_raw) - 1
                            else ids[1:-1]
                        )
                    )
                    for (j, ids) in enumerate(splitted_input_ids_raw)
                ]
                index_counter = 0
                splitted_input_indices = []
                for j, l in enumerate(splitted_input_ids):
                    begin_offset = 1 if j == 0 else 0
                    end_offset = 1 if j == len(splitted_input_ids) - 1 else 0
                    quote_s_offset = (
                        1 if "'s" in splitted[j] and splitted[j].index("'s") == 0 else 0
                    )
                    splitted_input_indices.append(
                        list(
                            range(
                                index_counter + begin_offset,
                                index_counter + len(l) - end_offset - quote_s_offset,
                            )
                        )
                    )
                    index_counter += len(l) - quote_s_offset

                # Get the token indices for each name
                name_token_indices = {
                    s: [
                        k
                        for phrase_id in is_name_ids[s]
                        for k in splitted_input_indices[phrase_id]
                    ]
                    for s in names
                }

                # Clean up the sentence and add it to the batch
                clean_sentence = union_sentence.replace("[", "").replace("]", "")

                # Preprocess the context
                curr_clean_contexts.append(clean_sentence)
                curr_name_token_indices_maps.append(name_token_indices)

            # Add this batch into the overall list; record the splits
            curr_size = len(curr_clean_contexts)
            clean_context_splits.append(
                (0, curr_size)
                if len(clean_context_splits) == 0
                else (
                    clean_context_splits[-1][1],
                    clean_context_splits[-1][1] + curr_size,
                )
            )
            clean_contexts += curr_clean_contexts
            name_token_indices_maps += curr_name_token_indices_maps

        # Return the preprocessed contexts and splits
        return (clean_contexts, clean_context_splits, name_token_indices_maps)

    def _extract_relations(
        self, clean_contexts, clean_context_splits, name_token_indices_maps
    ):
        # Use RoBERTa to encode the contexts into overall tensors
        context_tokenized_result = self.tokenizer(
            clean_contexts, padding=True, return_tensors="pt"
        )
        context_input_ids = context_tokenized_result.input_ids.to(self.device)
        context_attention_mask = context_tokenized_result.attention_mask.to(self.device)
        encoded_contexts = self.roberta_model(context_input_ids, context_attention_mask)
        # print(encoded_contexts)
        if self.no_fine_tune_roberta:
            roberta_embedding = encoded_contexts.last_hidden_state.detach()
        else:
            roberta_embedding = encoded_contexts.last_hidden_state

        # Extract features corresponding to the names for each context
        splits, name_pairs, name_pairs_features = [], [], []

        for begin, end in clean_context_splits:
            curr_datapoint_name_pairs = []
            curr_datapoint_name_pairs_features = []
            curr_sentence_rep = []

            for j, name_token_indices in zip(
                range(begin, end), name_token_indices_maps[begin:end]
            ):
                # Generate the feature_maps
                feature_maps = {}
                curr_sentence_rep.append(
                    torch.mean(
                        roberta_embedding[j, : sum(context_attention_mask[j]), :], dim=0
                    )
                )
                for name, token_indices in name_token_indices.items():
                    token_features = roberta_embedding[j, token_indices, :]

                    # Use max pooling to join the features
                    agg_token_feature = torch.max(token_features, dim=0).values
                    feature_maps[name] = agg_token_feature

                # Generate name pairs
                names = list(name_token_indices.keys())
                curr_sentence_name_pairs = [
                    (m, n) for m in names for n in names if m != n
                ]
                curr_datapoint_name_pairs += curr_sentence_name_pairs
                curr_datapoint_name_pairs_features += [
                    torch.cat((feature_maps[x], feature_maps[y]))
                    for (x, y) in curr_sentence_name_pairs
                ]

            global_rep = torch.mean(torch.stack(curr_sentence_rep), dim=0)

            # Generate the pairs for this datapoint
            num_name_pairs = len(curr_datapoint_name_pairs)
            splits.append(
                (0, num_name_pairs)
                if len(splits) == 0
                else (splits[-1][1], splits[-1][1] + num_name_pairs)
            )
            name_pairs += curr_datapoint_name_pairs
            name_pairs_features += curr_datapoint_name_pairs_features

        # Stack all the features into the same big tensor
        name_pairs_features = torch.cat(
            (
                torch.stack(name_pairs_features),
                global_rep.repeat(len(name_pairs_features), 1),
            ),
            dim=1,
        )

        # Use MLP to extract relations between names
        name_pair_relations = self.relation_extraction(name_pairs_features)

        # Return the extracted relations and their corresponding symbols
        return (splits, name_pairs, name_pair_relations)

    def _extract_facts(self, splits, name_pairs, name_pair_relations, queries):
        context_facts, context_disjunctions, question_facts = [], [], []
        num_pairs_processed = 0

        # Generate facts for each context
        for i, (begin, end) in enumerate(splits):
            # First combine the name_pair features if there are multiple of them, using max pooling
            name_pair_to_relations_map = {}
            for j, name_pair in zip(range(begin, end), name_pairs[begin:end]):
                name_pair_to_relations_map.setdefault(name_pair, []).append(
                    name_pair_relations[j]
                )
            name_pair_to_relations_map = {
                k: torch.max(torch.stack(v), dim=0).values
                for (k, v) in name_pair_to_relations_map.items()
            }

            # Generate facts and disjunctions
            curr_context_facts = []
            curr_context_disjunctions = []
            for (sub, obj), relations in name_pair_to_relations_map.items():
                curr_context_facts += [
                    (relations[k], (k, sub, obj)) for k in range(len(relation_id_map))
                ]
                curr_context_disjunctions.append(
                    list(range(len(curr_context_facts) - 20, len(curr_context_facts)))
                )
            context_facts.append(curr_context_facts)
            context_disjunctions.append(curr_context_disjunctions)
            question_facts.append([queries[i]])

            # Increment the num_pairs processed for the next datapoint
            num_pairs_processed += len(name_pair_to_relations_map)

        # Return the facts generated
        return (context_facts, context_disjunctions, question_facts)

    def forward(self, x, phase="train"):
        (contexts, queries, context_splits) = x

        # Debug prints
        if self.debug:
            print(contexts)
            print(queries)

        # Go though the preprocessing, RoBERTa model forwarding, and facts extraction steps
        (clean_contexts, clean_context_splits, name_token_indices_maps) = (
            self._preprocess_contexts(contexts, context_splits)
        )
        (splits, name_pairs, name_pair_relations) = self._extract_relations(
            clean_contexts, clean_context_splits, name_token_indices_maps
        )
        (context_facts, context_disjunctions, question_facts) = self._extract_facts(
            splits, name_pairs, name_pair_relations, queries
        )

        def transitive_rela(fact1, fact2):
            c1, rela1 = fact1
            c2, rela2 = fact2
            return (c1, (transitive_map.get((rela1[0], rela2[0])), rela1[1], rela2[2]))

        def transitive_rela_cond(fact1, fact2):
            c1, rela1 = fact1
            c2, rela2 = fact2
            return (
                c1 == c2
                and rela1[2] == rela2[1]
                and rela1[1] != rela2[2]
                and (rela1[0], rela2[0]) in transitive_map
            )

        def derive_rela(d: Distribution):
            new_d = d.apply_if(d, transitive_rela, transitive_rela_cond)
            merge_d = new_d | d
            if len(merge_d.symbols) == len(d.symbols):
                return merge_d
            return derive_rela(merge_d)

        def fast_derive_rela(old_d: Distribution, new_d: Distribution):
            new_d = new_d.apply_if(old_d, transitive_rela, transitive_rela_cond)
            merge_d = new_d | old_d
            if len(merge_d.symbols) == len(old_d.symbols):
                return merge_d
            return fast_derive_rela(merge_d, new_d)

        # new relation derivations
        def address_question(tab: Table, q: Tuple, max_i = 10):
            tab = tab.project(lambda i, a, b, probs : (a, b, probs), disable=True)
            tab_pairs = tab.project(lambda a, b, probs : (a, b), disable=True)
            while True:
                filtered = tab.filter(lambda a, b, probs: (a, b) == q, disable=True, batch_size=1024)
                if len(filtered) > 0:
                    break
                new_tab = tab.join(tab, key=lambda idx, a, b, probs : b, fkey=lambda idx, a, b, probs : a, disable=True) \
                            .filter(lambda a1, b1, p1, a2, b2, p2 : a1 != b2, disable=True)
                new_tab = new_tab.project(lambda a1, b1, p1, a2, b2, p2 : (a1, b2, p1.apply_if(p2, lambda rel1, rel2 : transitive_map.get((rel1, rel2), -1), lambda rel1, rel2: (rel1, rel2) in transitive_map)), disable=True, batch_size=1024)
                new_tab = tab.union(new_tab, disable=True)
                
                max_i -= 1
                
                if len(new_tab) == len(tab): # or max_i == 0 or len(new_tab) > 1000:
                    break
                tab = new_tab
                
            if len(filtered) == 0:
                f = Distribution(torch.tensor([0.0] * 21, device=self.device), list(range(21)))
            else:
                distrs = filtered.project(lambda a, b, probs : probs, disable=True).rows
                f = distrs[0]
                for i in range(1, len(distrs)):
                    f = f | distrs[i]
            return f
        
        def address_question_semi_naive_samplewise(tab: Table, q: Tuple, max_i = 10):
            tab = tab.project(lambda i, a, b, probs : (a, b, probs), disable=True)

            def merge_probs(tab: Table):
                probs = tab.project(lambda a, b, probs : probs, disable=True).rows
                f = probs[0]
                for i in range(1, len(probs)):
                    f = f | probs[i]
                return f
            
            new_tab = tab
            while True:
                # tab_pairs = tab.project(lambda a, b, probs : (a, b), disable=True)
                filtered = tab.filter(lambda a, b, probs: (a, b) == q, disable=True)
                if len(filtered) > 0:
                    break
                new_tab = new_tab.join(tab, key=lambda idx, a, b, probs : b, fkey=lambda idx, a, b, probs : a, disable=True) \
                            .filter(lambda a1, b1, p1, a2, b2, p2 : a1 != b2, disable=True)
                
                new_tab = new_tab.project(lambda a1, b1, p1, a2, b2, p2 : (a1, b2, p1.apply_if(p2, 
                            lambda rel1, rel2 : transitive_map.get((rel1, rel2), -1),
                            lambda rel1, rel2: (rel1, rel2) in transitive_map)), disable=True, batch_size=1024)
                
                # merged_tab = tab.union(new_tab.filter(lambda a, b, probs : (a, b) not in tab_pairs, disable=True), disable=True)


                merged_tab = tab.union(new_tab, disable=True)
                # grouping tabs
                merged_tab = merged_tab.group_by(lambda a, b, probs : (a, b), disable=True) \
                                .project(lambda pair, group : (pair[0], pair[1], group.reduce(lambda tab: merge_probs(tab), disable=True)), disable=True)
                
                max_i -= 1
                
                if len(merged_tab) == len(tab): # or max_i == 0: # or len(new_tab) > 1000:
                    break
                tab = merged_tab
                
            if len(filtered) == 0:
                f = Distribution(torch.tensor([0.0] * 21, device=self.device), list(range(21)))
                # if phase == "test":
                #     print("NO SOLUTION")
            else:
                distrs = filtered.project(lambda a, b, probs : probs, disable=True).rows
                f = distrs[0]
                # for i in range(1, len(distrs)):
                #     f = f | distrs[i]
            return f

        def address_question_semi_naive(tab: Table, recent_tab: Table, answers: Table = None, max_i = 10):
            filtered = tab.filter(lambda i, a, b, probs, *q: (a, b) == (q[0], q[1]), disable=True)
            filtered_ids = filtered.project(lambda i, a, b, probs, *q : i, disable=True).rows
            tab = tab.filter(lambda i, a, b, probs, *q: i not in filtered_ids, disable=True)
            tab_pairs = tab.project(lambda i, a, b, probs, *q : (i, a, b), disable=True)

            if answers is not None:
                answers = answers.union(filtered.project(lambda i, a, b, probs, *q : (i, q, probs), disable=True), disable=True)
            else:
                answers = filtered.project(lambda i, a, b, probs, *q : (i, q, probs), disable=True)

            new_tab = recent_tab.join(tab, key=lambda idx, i, a, b, probs, *q : (i, b), fkey=lambda idx, i, a, b, probs, *q : (i, a), disable=True) \
                        .filter(lambda i1, a1, b1, p1, q1a, q1b, i2, a2, b2, p2, q2a, q2b : a1 != b2, disable=True) \
                        .project(lambda i1, a1, b1, p1, q1a, q1b, i2, a2, b2, p2, q2a, q2b : (i1, a1, b2, p1.apply_if(p2, 
                            lambda rel1, rel2 : transitive_map.get((rel1, rel2), -1),
                            lambda rel1, rel2: (rel1, rel2) in transitive_map), q1a, q2b), disable=True, batch_size=1024) \
                            .project(lambda i, a, c, p, *q : (int(i), a, c, p, *q), disable=True)
            # print(new_tab)
            # new_tab = new_tab.project(lambda i, a, b, c, p1, p2, *q : (i, a, c, p1.apply_if(p2, 
            #                 lambda rel1, rel2 : transitive_map.get((rel1, rel2), -1),
            #                 lambda rel1, rel2: (rel1, rel2) in transitive_map), *q), disable=True, batch_size=1024) \
            #                 .project(lambda i, a, c, p, *q : (int(i), a, c, p, *q), disable=True)
            
            merged_tab = tab.union(new_tab.filter(lambda i, a, b, probs, *q : (int(i), a, b) not in tab_pairs, disable=True), disable=True)
            # merged_tab = tab.union(new_tab, disable=True)

            if len(merged_tab) == len(tab): # or len(new_tab) > 20000: # or max_i == 0:
                return answers
            
            return address_question_semi_naive(merged_tab, new_tab, answers, max_i - 1)
        
        def get_next_derivation_grouped(prob_tab: Table, query_tab: Table):
            """
            Prob Tab:
                    | i | p1 | p2 | Distr:[0, 1, ... 20] [p0, p1, ..., p20] |
            
            Query Tab:
                    | i | (qp1, qp2) |
            """
            t = time()
            grouped = prob_tab.group_by(lambda i, a, b, probs : i, disable=True).join(query_tab, key=lambda idx, i, _ : i, fkey=lambda idx, i, _ : i, disable=True) \
                        .project(lambda i, p_tab, _, q : (i, p_tab, q), disable=True) \
                        .project(lambda i, p_tab, q : (address_question_semi_naive_samplewise(p_tab, q), ), disable=True)
            return grouped

        def get_next_derivation(prob_tab: Table, query_tab: Table):
            """
            Prob Tab:
                    | i | p1 | p2 | Distr:[0, 1, ... 20] [p0, p1, ..., p20] |
            
            Query Tab:
                    | i | (qp1, qp2) |
            """
            t = time()
            joined = prob_tab.join(query_tab, key=lambda idx, i, _, __, ___ : i, fkey=lambda idx, i, _ : i, disable=True) \
                        .project(lambda i, a, b, probs, _, q : (i, a, b, probs, *q), disable=True)

            id_list = query_tab.project(lambda i, *q : i, disable=True)
            # print(id_list.rows)

            answers = address_question_semi_naive(joined, joined)
            answered_ids = answers.project(lambda i, q, probs : i, disable=True)

            unanswered = joined.project(lambda i, a, b, probs, *q : (i, q), disable=True).filter(lambda i, q : i not in answered_ids, disable=True)

            def merge_probs(tab: Table):
                probs = tab.project(lambda i, q, probs : probs, disable=True).rows
                f = probs[0]
                for i in range(1, len(probs)):
                    f = f | probs[i]
                return f

            answers = answers.union(Table([(i, q, Distribution(torch.tensor([0.0] * 21, device=self.device), list(range(21))) ) for i, q in unanswered]), disable=True) \
                            .group_by(lambda i, q, probs : i, disable=True) \
                            .project(lambda i, group: (i, group.reduce(lambda tab: merge_probs(tab), disable=True)), disable=True)
            
            # print(answers.project(lambda i, probs : i, disable=True).rows)
            
            return id_list.join(answers, key=lambda idx, i : i, fkey=lambda idx, i, probs : i, disable=True).project(lambda i, i1, probs : probs, disable=True)


            
            

        probability_dict = {}
        question_tab = []
        for i in range(len(context_facts)):
            facts, question = context_facts[i], question_facts[i][0]
            for p, s in facts:
                key = (i, s[1], s[2])
                if key in probability_dict:
                    probability_dict[key][s[0]] = p
                else:
                    probability_dict[key] = {s[0]: p}
            question_tab.append((i, question))

        probs_tab = Table(list(probability_dict.items())).project(lambda key, prob_dict: (*key, Distribution(torch.stack(list(prob_dict.values())), list(prob_dict.keys()))), disable=True)
        """
        (i, p1, p2) | Distr:[0, 1, ... 20] [p0, p1, ..., p20]
        
        """
        question_tab = Table(question_tab)

        # derived = get_next_derivation_grouped(probs_tab, question_tab).project(lambda d : d.map_symbols(list(range(21))).get_probabilities(), disable=True)
        derived = get_next_derivation(probs_tab, question_tab).project(lambda d : d.map_symbols(list(range(21))).get_probabilities(), disable=True)
        final_probs = torch.stack([ x[0] for x in derived.rows ])
        return final_probs
        


class Trainer:
    def __init__(
        self,
        train_loader,
        test_loader,
        device,
        model_dir,
        model_name,
        learning_rate,
        **args,
    ):
        self.device = device
        load_model = args.pop("load_model")
        if load_model:
            new_model = CLUTRRModel(device=device, **args).to(device)
            loaded_model = torch.load(
                os.path.join(model_dir, model_name + ".best.model")
            )
            new_model.tokenizer = loaded_model.tokenizer
            new_model.roberta_model = loaded_model.roberta_model
            self.model = new_model
        else:
            self.model = CLUTRRModel(device=device, **args).to(device)
        self.model_dir = model_dir
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.min_test_loss = 10000000000.0
        self.max_accu = 0

    def loss(self, y_pred, y):
        # print(y_pred)
        result = y_pred
        (_, dim) = result.shape
        gt = torch.stack(
            [torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y]
        ).to(self.device)
        result_loss = nn.functional.binary_cross_entropy(result, gt)
        return result_loss

    def accuracy(self, y_pred, y):
        batch_size = len(y)
        result = y_pred.detach()
        # print(result)
        pred = torch.argmax(result, dim=1)
        # print(list(zip(pred, y)))
        num_correct = len([() for i, j in zip(pred, y) if i == j])
        return (num_correct, batch_size)

    def train(self, num_epochs):
        # self.test_epoch(0)
        total_training_time = 0
        for i in range(1, num_epochs + 1):
            t = time()
            self.train_epoch(i)
            total_training_time += time() - t
            self.test_epoch(i)
        print(f"Total training time: {total_training_time}")
        print(f"Max accuracy: {self.max_accu}")

    def train_epoch(self, epoch):
        self.model.train()
        total_count = 0
        total_correct = 0
        total_loss = 0
        iterator = tqdm(self.train_loader)
        t_begin_epoch = time()
        print("LEN ITERATON", len(iterator))
        # torch.cuda.memory._record_memory_history(max_entries=100000)

        # try:
        for i, (x, y) in enumerate(iterator):
            self.optimizer.zero_grad()
            y_pred = self.model(x, "train")
            loss = self.loss(y_pred, y)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            (num_correct, batch_size) = self.accuracy(y_pred, y)
            total_count += batch_size
            total_correct += num_correct
            correct_perc = 100.0 * total_correct / total_count
            avg_loss = total_loss / (i + 1)

            iterator.set_description(
                f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)"
            )
            wandb.log({"epoch": epoch, "train/loss": loss})
            wandb.log({"epoch": epoch, "train/avg_loss": avg_loss})

        t_epoch = time() - t_begin_epoch
        wandb.log({"epoch": epoch, "train/it_time": t_epoch / len(iterator)})
        
        
        # except Exception as e:
        #     torch.cuda.memory._dump_snapshot("OOM RECORD")
        #     raise e
        # finally:
        #     torch.cuda.memory._record_memory_history(enabled=None)

        return avg_loss, correct_perc

    def test_epoch(self, epoch):
        self.model.eval()
        total_count = 0
        total_correct = 0
        total_loss = 0
        with torch.no_grad():
            iterator = tqdm(self.test_loader)
            for i, (x, y) in enumerate(iterator):
                y_pred = self.model(x, "test")
                loss = self.loss(y_pred, y)
                total_loss += loss.item()

                (num_correct, batch_size) = self.accuracy(y_pred, y)
                total_count += batch_size
                total_correct += num_correct
                correct_perc = 100.0 * total_correct / total_count
                avg_loss = total_loss / (i + 1)

                iterator.set_description(
                    f"[Test Epoch {epoch}] Avg Loss: {avg_loss}, Accuracy: {total_correct}/{total_count} ({correct_perc:.2f}%)"
                )

        # Save model
        if total_correct / total_count > self.max_accu:
            self.max_accu = total_correct / total_count
            torch.save(
                self.model,
                os.path.join(self.model_dir, f"{self.model_name}.best.model"),
            )
        torch.save(
            self.model, os.path.join(self.model_dir, f"{self.model_name}.latest.model")
        )

        wandb.log(
            {
                "epoch": epoch,
                "test/loss": total_loss,
                "test/acc": total_correct / total_count,
            }
        )

        return avg_loss, correct_perc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data_7685d0cd")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--training_data_percentage", type=int, default=100)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1831)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-mlp-layers", type=int, default=2)
    parser.add_argument(
        "--provenance", type=str, default="damp", choices=["damp", "dmmp", "dtkp-am"]
    )
    parser.add_argument("--sample-k", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--constraint-weight", type=float, default=0.2)

    parser.add_argument("--no-fine-tune-roberta", type=bool, default=False)
    parser.add_argument("--use-softmax", type=bool, default=True)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--use-last-hidden-state", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    args.cuda = True
    print(args)

    name = f"training_perc_{args.training_data_percentage}_seed_{args.seed}_clutrr"

    if args.model_name is None:
        args.model_name = name

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        transformers.set_seed(args.seed)

    if args.cuda:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    # Setting up data and model directories
    # data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    data_root = "../data/"
    model_dir = os.path.abspath(
        os.path.join(os.path.abspath(__file__), "../../model/clutrr")
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Setup wandb
    config = {
        "device": device,
        "provenance": args.provenance,
        "top_k": args.top_k,
        "sample_k": args.sample_k,
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "training_data_perc": args.training_data_percentage,
        "experiment_type": "torchql-alt",
    }

    timestamp = datetime.datetime.now()
    id = f'torchql_clutrr_{args.provenance}({args.top_k})_{args.seed}_{timestamp.strftime("%Y-%m-%d %H-%M-%S")}'

    wandb.login()
    wandb.init(project="clutrr", config=config, id=id)
    wandb.define_metric("epoch")
    wandb.define_metric("train/it_time", step_metric="epoch", summary="mean")
    wandb.define_metric("test/loss", step_metric="epoch", summary="min")
    wandb.define_metric("test/acc", step_metric="epoch", summary="max")

    # Load the dataset
    (train_loader, test_loader) = clutrr_loader(
        data_root, args.dataset, args.batch_size, args.training_data_percentage
    )

    # Train
    trainer = Trainer(
        train_loader,
        test_loader,
        device,
        model_dir,
        args.model_name,
        args.learning_rate,
        num_mlp_layers=args.num_mlp_layers,
        debug=args.debug,
        provenance=args.provenance,
        sample_k=args.sample_k,
        top_k=args.top_k,
        use_softmax=args.use_softmax,
        no_fine_tune_roberta=args.no_fine_tune_roberta,
        load_model=args.load_model,
    )
    trainer.train(args.n_epochs)
