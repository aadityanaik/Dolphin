import enum
import logging
from re import L, S
from select import select
from time import time
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import os

from dolphin.provenances import get_provenance
from dolphin.distribution import Distribution
from torchql import Table

from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from modules import VideoEncoder, ProjectionHead, Projection, MLPClassifier


# logger = logging.getLogger("torchql.alignmodule")

# logger.stats = defaultdict(float)
# logger.reset_stats = lambda : logger.stats.clear()

action_list = ["crouch",  "stand", "walk", "climb", "jump", "collect", "die", "kill"]
horizontal_directions = ['left', 'right', 'none']
vertical_directions = ['up', 'down', 'none']
mugen_is_dead = [True, False]

monsters = ['gear', 'barnacle', 'face', 'slime', 'mouse', 'snail', 'ladybug', 'worm', 'frog', "bee", "none"]
collectables = ['coin', 'gem', 'none']

actions = {'text_mugen_action': action_list,
           'text_mugen_horizontal_dir': horizontal_directions,
           'text_mugen_vertical_dir': vertical_directions,
           'text_mugen_kill_monster': monsters,
           'text_mugen_kill_by_monster': monsters,
           'text_mugen_collect_item': collectables,
           'video_mugen_action': action_list,
           'video_mugen_horizontal_dir': horizontal_directions,
           'video_mugen_vertical_dir': vertical_directions,
           'video_mugen_kill_monster': monsters,
           'video_mugen_kill_by_monster': monsters,
           'video_mugen_collect_item': collectables,
           }


def split_n_per_list(l, n):
    for idx in range(0, len(l), n):
        yield l[idx: idx+n]

class CLIPModel(nn.Module):
    def __init__(self, video_enc=False, pretrained=False, trainable=False,):
        super().__init__()
        self.video_enc = video_enc

        if self.video_enc:
            self.visual_encoder = VideoEncoder(pretrained=pretrained, trainable=trainable)
            self.image_projection = Projection(self.visual_encoder.embedding_dim)

    def get_video_embedding(self, batch):
        image_features = self.visual_encoder(batch["video"])
        image_embed = self.image_projection(image_features)
        image_embed = F.normalize(image_embed, dim=-1)
        return image_embed

    def get_audio_embedding(self, batch):
        audio_features = self.audial_encoder(batch["audio"])
        audio_embed = self.audio_projection(audio_features)
        audio_embed = F.normalize(audio_embed, dim=-1)
        return audio_embed

    def get_text_embedding(self, batch):
        text_features = self.text_encoder(batch['text'])
        # Getting Image and Text Embeddings (with same dimension)
        caption_embed = self.text_projection(text_features)
        caption_embed = F.normalize(caption_embed, dim=-1)
        return caption_embed

def get_video_scl_tuples(prediction, action_list, idxes):
    batched_scl_actions = []
    for data_id, (start_idx, end_idx) in enumerate(idxes):
        scl_actions = []
        for frame_id, seq_pred in enumerate(prediction[start_idx:end_idx]):
            for action_idx, prob in enumerate(seq_pred):
                scl_actions.append((prob, (frame_id, action_list[action_idx])))
        batched_scl_actions.append(scl_actions)
    return batched_scl_actions

def get_text_scl_tuples(prediction, action_list, idxes, multi_text):
    all_scl_actions = []
    batch_size = len(idxes)
    for data_id, (start_idx, end_idx) in enumerate(idxes):
        scl_actions = []
        for frame_id, seq_pred in enumerate(prediction[start_idx:end_idx]):
            for action_idx, prob in enumerate(seq_pred):
                if multi_text:
                    scl_actions.append((prob, (data_id, frame_id, action_list[action_idx])))
                else:
                    scl_actions.append((prob, (frame_id, action_list[action_idx])))
        all_scl_actions.append(scl_actions)
    if multi_text:
        all_scl_actions = [[j for i in all_scl_actions for j in i ]] * batch_size
    return all_scl_actions

def get_gt_text_scl_tuples(text_gt, idxes, multi_text):
    all_scl_actions = {}
    batch_size = len(idxes)
    assert not multi_text
    for data_id, (start_idx, end_idx) in enumerate(idxes):
        scl_actions = {}
        gt_result_ls = []

        for i in range(start_idx, end_idx):
            gt_result_ls += text_gt[i]

        for frame_id, gt_result in enumerate(gt_result_ls):
            for k, gt_action in gt_result.items():
                if not k in scl_actions:
                    scl_actions[k] = []
                if multi_text:
                    scl_actions[k].append((data_id, frame_id, gt_action))
                else:
                    scl_actions[k].append((frame_id, gt_action))

        for k, v in scl_actions.items():
            if not k in all_scl_actions:
                all_scl_actions[k] = []
            all_scl_actions[k].append(v)

    return all_scl_actions

def to_scl_string(result):
    scl_strings = []
    for rel_name, batched_tuples in result.items():
        tuples = batched_tuples[3]
        if isinstance(tuples[0][0], torch.Tensor):
            current_rel_string = 'rel ' + rel_name + '={' + ', '.join([str(prob.item()) + '::' + str(tp).replace("'", '"') for prob, tp in tuples]) + '}'
        else:
            current_rel_string = 'rel ' + rel_name + '={' + ', '.join([str(tp).replace("'", '"') for tp in tuples]) + '}'

        scl_strings.append(current_rel_string)
    return '\n'.join(scl_strings)

def obtain_prediction(result, text_idxes):
    predictions = {}
    batch_size = len(text_idxes)
    video_counts = int(result["video_mugen_action"].shape[0] / batch_size)

    for rel_name, preds in result.items():
        selected_values, selected_indices = torch.topk(preds, k=3, dim = 1)
        k_selected = [[(i, prob.item(), actions[rel_name][index])  for prob, index in zip(k_probs, k_indexes)] for i, (k_probs, k_indexes) in enumerate(zip(selected_values, selected_indices))]

        if not rel_name in predictions:
            predictions[rel_name] = []

        if "text" in rel_name:
            for start_idx, end_idx in text_idxes:
                predictions[rel_name].append(k_selected[start_idx: end_idx])
        else:
            predictions[rel_name] = [k_selected[video_counts * i: video_counts * (i+1) ] for i in range(batch_size)]

    return predictions

def combine_text_and_video(text_results, video_results):
    batch_size = len(list(text_results.values())[0])
    all_pos = []
    combined_scl_queries = {}

    for k in text_results.keys():
        combined_scl_queries[k] = []
    for k in video_results.keys():
        combined_scl_queries[k] = []

    for vid in range(batch_size):
        for tid in range(batch_size):
            pos = (vid, tid)
            all_pos.append(pos)

    for vid, tid in all_pos:
        for text_rel_name, text_batched_rels in text_results.items():
            combined_scl_queries[text_rel_name].append(text_batched_rels[tid])
        for video_rel_name, video_batched_rels in video_results.items():
            combined_scl_queries[video_rel_name].append(video_batched_rels[vid])

    return all_pos, combined_scl_queries

class AlignModule(nn.Module):
    def __init__(self, batch_size, video_enc=False, audio_enc=False, text_enc=False, pretrained=False, trainable=False,
                 text_embedding=768, video_decoder_layers=2, text_decoder_layers=2, dropout_rate=0.3, constraint_weight=0.1,
                 provenance="damp", device="cpu", top_k=5, debug=True, multi_text=True,
                 alternative_train_freq=10, load_path=None, gt_text=False, pred_save_dir=None):

        super().__init__()

        if not load_path is None:
            self.load(load_path)
        else:
            self.clip_model = CLIPModel(video_enc=video_enc, pretrained=pretrained,
                      trainable=trainable)

            self.video_action_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(action_list), n_layers=video_decoder_layers, dropout_rate=dropout_rate)
            self.video_direction_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(horizontal_directions), n_layers=video_decoder_layers, dropout_rate=dropout_rate)
            self.video_jump_direction_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(vertical_directions), n_layers=video_decoder_layers, dropout_rate=dropout_rate)
            self.video_is_dead_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(mugen_is_dead), n_layers=video_decoder_layers, dropout_rate=dropout_rate)
            self.video_killed_monster_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(monsters), n_layers=video_decoder_layers, dropout_rate=dropout_rate)
            self.video_killed_by_monster_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(monsters), n_layers=video_decoder_layers, dropout_rate=dropout_rate)
            self.video_collects_item_decoder =  MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(collectables), n_layers=video_decoder_layers, dropout_rate=dropout_rate)

            self.text_action_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(action_list), n_layers=text_decoder_layers, dropout_rate=dropout_rate)
            self.text_direction_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(horizontal_directions), n_layers=text_decoder_layers, dropout_rate=dropout_rate)
            self.text_jump_direction_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(vertical_directions), n_layers=text_decoder_layers, dropout_rate=dropout_rate)
            self.text_is_dead_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(mugen_is_dead), n_layers=text_decoder_layers, dropout_rate=dropout_rate)
            self.text_killed_monster_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(monsters), n_layers=text_decoder_layers, dropout_rate=dropout_rate)
            self.text_killed_by_monster_decoder = MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(monsters), n_layers=text_decoder_layers, dropout_rate=dropout_rate)
            self.text_collects_item_decoder =  MLPClassifier(input_dim=256, latent_dim=64, output_dim=len(collectables), n_layers=text_decoder_layers, dropout_rate=dropout_rate)

        self.debug = debug
        self.pred_save_dir = pred_save_dir
        self.multi_text = multi_text
        self.alternative_train_freq=alternative_train_freq
        self.constraint_weight = constraint_weight

        Distribution.provenance = get_provenance(provenance)
        Distribution.provenance.k = top_k
        # Distribution.k = 7
        self.device = device

        self.processed_batch = 0
        self.current_training_id = 0

        self.modal2models = {
            "video":
            [self.clip_model.visual_encoder,
             self.video_action_decoder,
             self.video_direction_decoder,
             self.video_jump_direction_decoder,
             self.video_is_dead_decoder,
             self.video_killed_monster_decoder,
             self.video_killed_by_monster_decoder,
             self.video_collects_item_decoder],
            }

    def save(self, save_path):
        nn_info = {"clip_model": self.clip_model,
        "video_action_decoder": self.video_action_decoder,
        "video_direction_decoder": self.video_direction_decoder,
        "video_jump_direction_decoder": self.video_jump_direction_decoder,
        "video_is_dead_decoder": self.video_is_dead_decoder,
        "video_killed_monster_decoder": self.video_killed_monster_decoder,
        "video_killed_by_monster_decoder": self.video_killed_by_monster_decoder,
        "video_collects_item_decoder": self.video_collects_item_decoder,
        }

        torch.save(nn_info, save_path)

    def load(self, save_path):
        nn_info = torch.load(save_path)
        self.clip_model = nn_info["clip_model"]
        self.video_action_decoder = nn_info["video_action_decoder"]
        self.video_direction_decoder = nn_info["video_direction_decoder"]
        self.video_jump_direction_decoder = nn_info["video_jump_direction_decoder"]
        self.video_is_dead_decoder = nn_info["video_is_dead_decoder"]
        self.video_killed_monster_decoder = nn_info["video_killed_monster_decoder"]
        self.video_killed_by_monster_decoder = nn_info["video_killed_by_monster_decoder"]
        self.video_collects_item_decoder = nn_info["video_collects_item_decoder"]

    def set_train(self, model_ls, is_train):
        for model in model_ls:
            if is_train:
                model.train()
            else:
                model.eval()

    def toggle_training_model(self):
        training_models = list(self.modal2models.keys())
        current_train_model = training_models[self.current_training_id]

        for modal in self.modal2models.keys():
            if not modal == current_train_model:
                self.set_train(self.modal2models[modal], False)
            else:
                self.set_train(self.modal2models[modal], True)

    def devide_batch(self, batch):
        batch_size = len(batch['text_idx'])
        single_dps = [{} for _ in range(batch_size)]
        video_split = int(batch['video'].shape[0] / batch_size)

        for dp_ct, single_text_idx in enumerate(batch['text_idx']):
            single_dps[dp_ct]['text_idx'] = []
            single_dps[dp_ct]['text'] = []
            start_idx = single_text_idx[0]
            end_idx = single_text_idx[1]
            single_dps[dp_ct]['text'].append(batch['text'][start_idx: end_idx])
            from_id, to_id = single_text_idx
            single_dps[dp_ct]['text_idx'].append((from_id - start_idx, to_id - start_idx))

        for k, v in batch.items():
            if k == 'text' or k == 'text_idx':
                continue
            if k == 'video':
                for dp_ct, single_video in enumerate(list(split_n_per_list(v, video_split))):
                    single_dps[dp_ct][k] = single_video
                continue
            for dp_ct, small_batch_values in enumerate(v):
                single_dps[dp_ct][k] = small_batch_values

        return single_dps


    # Just calculate the probability, no need to backprob
    def predict(self, batch, n=1):
        raise Exception("unimplemented")
    
        # text_gt = self.get_gt_text(batch)
        # text_results = get_gt_text_scl_tuples(text_gt, batch['text_idx'], self.multi_text)
        # single_dps = self.devide_batch(batch)
        # single_dps_pred_scls = {'video_mugen_action': [],
        #                         'video_mugen_horizontal_dir': [],
        #                         'video_mugen_vertical_dir': [],
        #                         'video_mugen_kill_monster': [],
        #                         'video_mugen_kill_by_monster': [],
        #                         'video_mugen_collect_item': []}

        # batch_size = len(batch['text_idx'])
        # video_split = int(batch['video'].shape[0] / batch_size)
        # video_idxes = [(0,  video_split)]



        # video_embedding = self.clip_model.get_video_embedding(batch)

        # pred_video_actions = self.video_action_decoder(video_embedding)
        # pred_video_horizontal_directions = self.video_direction_decoder(video_embedding)
        # pred_video_vertical_directions = self.video_jump_direction_decoder(video_embedding)
        # pred_video_killed_monster = self.video_killed_monster_decoder(video_embedding)
        # pred_video_killed_by_monster = self.video_killed_by_monster_decoder(video_embedding)
        # pred_video_collects_item = self.video_collects_item_decoder(video_embedding)

        # batch_size = len(batch['text_idx'])
        # video_split = int(batch['video'].shape[0] / batch_size)
        # video_idxes = [(video_split * i,  video_split * (i + 1)) for i in range(batch_size)]

        # text_starts = []
        # text_ends = []
        # video_starts = []
        # video_ends = []

        # if self.multi_text:
        #     for data_id, ((text_start, text_end)) in enumerate(batch['text_idx']):
        #         text_starts.append((data_id, 0))
        #         text_ends.append((data_id, text_end - text_start - 1))
        #         video_starts.append([(0,)])
        #         video_ends.append([(video_split-1,)])
        #     text_starts = [text_starts] * batch_size
        #     text_ends = [text_ends] * batch_size

        # else:
        #     for data_id, ((text_start, text_end)) in enumerate(batch['text_idx']):
        #         text_starts.append([(0,)])
        #         text_ends.append([(text_end - text_start - 1,)])
        #         video_starts.append([(0,)])
        #         video_ends.append([(video_split-1,)])

        # text_results['text_start'] = text_starts
        # text_results['text_end'] = text_ends
        # video_mugen_action = get_video_scl_tuples(pred_video_actions, action_list, video_idxes)
        # video_mugen_horizontal_dir = get_video_scl_tuples(pred_video_horizontal_directions, horizontal_directions, video_idxes)
        # video_mugen_vertical_dir = get_video_scl_tuples(pred_video_vertical_directions, vertical_directions, video_idxes)
        # video_mugen_killed_monster = get_video_scl_tuples(pred_video_killed_monster, monsters, video_idxes)
        # video_mugen_killed_by_monster = get_video_scl_tuples(pred_video_killed_by_monster, monsters, video_idxes)
        # video_mugen_collects_item = get_video_scl_tuples(pred_video_collects_item, collectables, video_idxes)

        # video_results = {
        #     'video_mugen_action': video_mugen_action,
        #     'video_mugen_horizontal_dir': video_mugen_horizontal_dir,
        #     'video_mugen_vertical_dir': video_mugen_vertical_dir,
        #     'video_mugen_kill_monster': video_mugen_killed_monster,
        #     'video_mugen_kill_by_monster': video_mugen_killed_by_monster,
        #     'video_mugen_collect_item': video_mugen_collects_item,
        #     'video_start': video_starts,
        #     'video_end': video_ends
        # }

        # results = {
        #     'video_mugen_action': pred_video_actions,
        #     'video_mugen_horizontal_dir': pred_video_horizontal_directions,
        #     'video_mugen_vertical_dir': pred_video_vertical_directions,
        #     'video_mugen_kill_monster': pred_video_killed_monster,
        #     'video_mugen_kill_by_monster': pred_video_killed_by_monster,
        #     'video_mugen_collect_item': pred_video_collects_item,
        # }

        # text_pred = obtain_prediction(results, batch['text_idx'])
        # pos, queries = combine_text_and_video(text_results, video_results)

        # pred = self.reason(**queries)

        # # pred dim: video x text
        # pred_match = pred['text_video_match'].reshape(batch_size, batch_size)
        # pred_constraint_violation = pred['too_many_consecutive_text'].reshape(batch_size, batch_size)

        # self.processed_batch = (self.processed_batch + 1) % self.alternative_train_freq
        # if self.processed_batch == 0:
        #     self.current_training_id = (self.current_training_id + 1) % len(self.modal2models)

        # return pred_match, pred_constraint_violation

    def get_one_aspect(self, text, action_ls, default="none"):

        processed_text = text.replace("killed by", "die")
        mentioned_elements = []

        for word in text.split(' '):
            for action in action_ls:
                if action in word:
                    mentioned_elements.append(action)

        # for action in action_ls:
        #     if action in text:
        #        mentioned_elements.append(action)

        if len(mentioned_elements) == 0:
            mentioned_elements = [default]
        return mentioned_elements

    def get_gt_text(self, batch):
        batched_texts = {}
        for text_id, text in enumerate(batch['text']):

            action = self.get_one_aspect(text, action_list)
            assert(not action == ["none"] and len(action) == 1)
            hori_dir = self.get_one_aspect(text, horizontal_directions)
            verti_dir = self.get_one_aspect(text, vertical_directions)
            collectable_ls = self.get_one_aspect(text, collectables)
            monster = self.get_one_aspect(text, monsters)

            # Only collectable can be more than 1
            assert(len(hori_dir) == 1 and len(verti_dir) == 1)
            # if action[0] == "die" or action[0] == "kill":
            #     assert (len(monster) == 1) // There maybe more than two monsters being killed

            monster = monster[0]
            if "killed_by" in text:
                kill_by = monster
            elif "killed" in text:
                kill_by = "none"
                kill =  monster
            else:
                # assert monster == "none"
                kill_by = "none"
                kill =  "none"

            # if len(collectable_ls) > 1:
            #     print("here")

            for collectable in collectable_ls:

                text_description = {}
                text_description['text_mugen_action'] = action[0]
                text_description['text_mugen_horizontal_dir'] = hori_dir[0]
                text_description['text_mugen_vertical_dir'] = verti_dir[0]
                text_description['text_mugen_collect_item'] = collectable
                text_description['text_mugen_kill_by_monster'] = kill_by
                text_description['text_mugen_kill_monster'] = kill

                if not text_id in batched_texts:
                    batched_texts[text_id] = []
                batched_texts[text_id].append(text_description)
        return batched_texts

    # Forward train video only
    def forward(self, batch):
        text_gt = self.get_gt_text(batch)
        text_results = get_gt_text_scl_tuples(text_gt, batch['text_idx'], self.multi_text)

        # TODO: maybe use a window rather than hard split?
        video_embedding = self.clip_model.get_video_embedding(batch)

        pred_video_actions = self.video_action_decoder(video_embedding)
        pred_video_horizontal_directions = self.video_direction_decoder(video_embedding)
        pred_video_vertical_directions = self.video_jump_direction_decoder(video_embedding)
        pred_video_killed_monster = self.video_killed_monster_decoder(video_embedding)
        pred_video_killed_by_monster = self.video_killed_by_monster_decoder(video_embedding)
        pred_video_collects_item = self.video_collects_item_decoder(video_embedding)

        batch_size = len(batch['text_idx'])
        video_split = int(batch['video'].shape[0] / batch_size)
        video_idxes = [(video_split * i,  video_split * (i + 1)) for i in range(batch_size)]
        # print(batch['video'].shape, video_split, video_idxes)

        text_starts = []
        text_ends = []
        video_starts = []
        video_ends = []

        if self.multi_text:
            for data_id, ((text_start, text_end)) in enumerate(batch['text_idx']):
                text_starts.append((data_id, 0))
                text_ends.append((data_id, text_end - text_start - 1))
                video_starts.append([(0,)])
                video_ends.append([(video_split-1,)])
            text_starts = [text_starts] * batch_size
            text_ends = [text_ends] * batch_size

        else:
            for data_id, ((text_start, text_end)) in enumerate(batch['text_idx']):
                text_starts.append([(0,)])
                text_ends.append([(text_end - text_start - 1,)])
                video_starts.append([(0,)])
                video_ends.append([(video_split-1,)])

        text_results['text_start'] = text_starts
        text_results['text_end'] = text_ends
        video_mugen_action = get_video_scl_tuples(pred_video_actions, action_list, video_idxes)
        video_mugen_horizontal_dir = get_video_scl_tuples(pred_video_horizontal_directions, horizontal_directions, video_idxes)
        video_mugen_vertical_dir = get_video_scl_tuples(pred_video_vertical_directions, vertical_directions, video_idxes)
        video_mugen_killed_monster = get_video_scl_tuples(pred_video_killed_monster, monsters, video_idxes)
        video_mugen_killed_by_monster = get_video_scl_tuples(pred_video_killed_by_monster, monsters, video_idxes)
        video_mugen_collects_item = get_video_scl_tuples(pred_video_collects_item, collectables, video_idxes)

        video_results = {
            'video_mugen_action': video_mugen_action,
            'video_mugen_horizontal_dir': video_mugen_horizontal_dir,
            'video_mugen_vertical_dir': video_mugen_vertical_dir,
            'video_mugen_kill_monster': video_mugen_killed_monster,
            'video_mugen_kill_by_monster': video_mugen_killed_by_monster,
            'video_mugen_collect_item': video_mugen_collects_item,
            'video_start': video_starts,
            'video_end': video_ends
        }

        # print(str(video_results))

        # results = {
        #     'video_mugen_action': pred_video_actions,
        #     'video_mugen_horizontal_dir': pred_video_horizontal_directions,
        #     'video_mugen_vertical_dir': pred_video_vertical_directions,
        #     'video_mugen_kill_monster': pred_video_killed_monster,
        #     'video_mugen_kill_by_monster': pred_video_killed_by_monster,
        #     'video_mugen_collect_item': pred_video_collects_item,
        # }

        # text_pred = obtain_prediction(results, batch['text_idx'])

        # pos, queries = combine_text_and_video(text_results, video_results)

        def unary_fact_table(batched_facts):
            flatten_facts = []
            for i, facts in enumerate(batched_facts):
                flatten_facts.append((i, facts[0][0]))
            return Table(flatten_facts)

        def non_probabilistic_fact_table(batched_facts):
            # flatten_facts = []
            # for i, facts in enumerate(batched_facts):
            #     # TODO: think about what the disjunction should be for non-probabilistic distributions
            #     flatten_facts.append((i, Distribution(torch.ones(len(facts), device=self.device), facts)))
            # return Table(flatten_facts).group_by(lambda i, D : i)
            facts = []
            for fact in batched_facts:
                row = [fact[0][1]]
                for i, f in fact[1:]:
                    if row[-1] != f:
                        row.append(f)
                facts.append(tuple(row))

            return facts

            
        def probabilistic_fact_table(batched_facts, text):
            # flatten_facts = []
            # for i, facts in enumerate(batched_facts):
            #     prob_dict = defaultdict(dict)
            #     for p, f in facts:
            #         prob_dict[f[0]][f[1:]] = p
            #     for id in prob_dict:
            #         flatten_facts.append((i, id, Distribution(torch.stack(list(prob_dict[id].values())), list(prob_dict[id].keys()))))
            # return Table(flatten_facts).group_by(lambda i, id, probs : i)
            flatten_facts = []
            for i, facts in enumerate(batched_facts):
                row = [i, ]
                prob_dict = defaultdict(dict)
                for p, f in facts:
                    prob_dict[f[0]][f[1:]] = p
                for id in prob_dict:
                    row.append(Distribution(torch.stack(list(prob_dict[id].values())), list(prob_dict[id].keys())))
                
                # analyze one text at a time
                # for t in text:
                #     # distrs = [ row[i].filter(lambda x : x[0] in t[:i]) for i in range(1, len(row)) ]
                #     distrs = row[1:]
                #     flatten_facts.append((row[0], t, *distrs))

                # analyze all texts in one go
                flatten_facts.append((row[0], *row[1:]))

            return Table(flatten_facts)
        
        text_start = unary_fact_table(text_results["text_start"])
        text_end = unary_fact_table(text_results["text_end"])
        video_start = unary_fact_table(video_results["video_start"])
        video_end = unary_fact_table(video_results["video_end"])

        text_mugen_action = non_probabilistic_fact_table(text_results["text_mugen_action"])
        video_mugen_action = probabilistic_fact_table(video_results["video_mugen_action"], text_mugen_action)

        # e1, e2, e3
        # e1, e1, e1, e2, e2, e3

        def concat_action(act_list, action):
            if act_list[-1] != action[0]:
                return act_list + action
            return act_list

        def cond(text, act_list, action):
            c = concat_action(act_list, action)

            # analyze all texts in one go
            for t in text:
                if c == t[:len(c)]:
                    return True
            return False
            
            # analyze one text at a time
            # return c == text[:len(c)]


        def get_action_list(idx, *actions):

            action_list = actions[0]
            # i=0
            for action_d in actions[1:]:
                # print("AL", action_list)
                # print("ACTION", action_d)
                # print("TEXT", text)
                action_list = action_list.apply_if(action_d, concat_action , lambda a1, a2: cond(text_mugen_action, a1, a2))
                # action_list = action_list | action_list.apply_if(action_d, concat_action, lambda a1, a2: cond(text, a1, a2))
                # print(text)
                # print(action_list)
                # print("\n\n\n")
                # print(i)
                # if i == 3:
                #     exit()
                # i += 1
            # print("AL", action_list, text)
            # exit()
            # old_action_list = action_list

            # analyze one text at a time
            # probs = action_list.map_symbols((text,)).get_probabilities()[0]

            # analyze all texts in one go
            probs = action_list.map_symbols(text_mugen_action).get_probabilities()
            # if probs.sum() == 0:
            #     print(probs)
            #     print(old_action_list, text)
            #     exit()
            return probs

        # print(video_mugen_action)
        pred_match = torch.stack(video_mugen_action.project(get_action_list, batch_size=-1).rows).reshape(batch_size, batch_size)
        # print(rezs)
        # exit()
        # print(text_mugen_action, video_mugen_action)

        def map_action_to_tid(video_T: Table, text_T: Table):
            # For every (vid, D) in video_T, map to (vid, D'), where the symbols of D' are the tids from text_T corresponding to single actions in D
            # print("Length of video_T is", len(video_T), "Length of text_T is", len(text_T))
            return video_T.join(text_T, lambda idx, *args: 0, fkey=lambda idx, *args: 0, batch_size=-1) \
                .project(lambda i, vid, vD, _, tD:
                         (vid, vD.apply_if(tD,
                                           lambda v_action, t_action: (t_action[0], t_action[0]),
                                           lambda v_action, t_action: v_action[0] == t_action[1])), batch_size=-1)
        
        def text_video_action_match_batched(T: Table, n_batches):
            # T :: [batch_idx, vid, D, vid_start, vid_end, text_start, text_end]
            # print("Length of T is", len(T), "video_start", video_start, "video_end", video_end, "text_start", text_start, "text_end", text_end)
            T = T.project(lambda b_idx, vid, D, *args: (b_idx, vid, vid + 1, D, *args))
            t = time()
            # print(T)
            single_frame = T
            num_ops = 0
            satisfied_batches = []
            # print(T)
            while True:
                if len(T) == 0:
                    break
                # TODO: check whether original scallop program is off by one w.r.t. vid_end?
                # filtered = T.filter(lambda b_idx, vid_start, vid_end, D, fvid_start, fvid_end, ft_start, ft_end: (int(b_idx), int(vid_start)) == (int(b_idx), int(fvid_start)) and (int(b_idx), int(vid_end)) == (int(b_idx), int(fvid_end))) \
                #     .project(lambda b_idx, vs, ve, D, vss, vee, ts, te: (b_idx, D, ts, te))
                
                # if len(filtered) >= n_batches or len(T) <= 0:
                #     b = filtered.project(lambda b_idx, vid_start, vid_end, D, *args: b_idx).rows

                #     T = T.filter(lambda b_idx, *args: b_idx not in b)
                #     # print("Length of T on termination is", len(T))
                #     # assert len(T) == 0, "Length of T is not 0 but " + str(len(T))
                #     break
                if len(T) == n_batches:
                    break
                # t = time()
                T = T.join(single_frame, key=lambda idx, b_idx, vid_start, vid_mid, D, *args: (int(b_idx), int(vid_mid)), fkey=lambda idx, b_idx, vid_mid, vid_end, D, *args: (int(b_idx), int(vid_mid)))
                T = T.project(lambda b_idx, vid_start, vid_mid1, D1, vs1, ve1, ts1, te1, _, vid_mid2, vid_end, D2, vs2, ve2, tv1, tv2:
                             (b_idx, vid_start, vid_end, D1.apply_if(D2,
                                                              lambda tid1, tid2: (tid1[0], tid2[1]),
                                                              lambda tid1, tid2: tid2[0] - 1 <= tid1[1] <= tid2[0]), vs1, ve1, ts1, te1), batch_size=-1)
                # T = T.project(lambda vid_start, vid_mid1, D1, vid_mid2, vid_end, D2:
                #              (vid_start, vid_end, D1.apply(D2, lambda tid1, tid2: (tid1[0], tid2[1]))), batch_size=256)
                # logger.stats["T_Reasoning_One_Loop"] += (time() - t)
                # if len(T) == 0:
                #     exit(1)
                # print("Length of T is", len(T))
            # print("OUTPUTTT")
            # print(d)
            # print("TIME")
            # print(time() - t)
            # exit()
            d = T.project(lambda b_idx, vs, ve, D, vss, vee, ts, te: D.map_symbols([(ts, te)]).get_probabilities())
            # print(d.rows)
            # exit()
            # print("Number of operations", num_ops)
            # print("Time taken", time() - t)
            return d.rows
        
        def text_video_action_match_recurse(T: Table, video_start, video_end, text_start, text_end):
            D = Distribution.stack(T.rows)

            # D : sfid, efid, t1, t2

            D_res = Distribution(D.tags, D.symbols)

            while True:
                D_res = D_res.apply_if(D, lambda S_res, S: (S_res[0], S[1], S_res[2], S[3]), lambda S_res, S: S_res[1] == S[0] and S[2] - 1 <= S_res[3] <= S[2])
                # print(torch.sum(D_res.tags))
                if (video_start, video_end, text_start, text_end) in list(D_res.symbols) or len(D_res.symbols) <= 1:
                    break

            final_probability = D_res.map_symbols([(video_start, video_end, text_start, text_end)]).get_probabilities()[0]
        
            return final_probability
        
        def text_video_action_match(T: Table, video_start, video_end, text_start, text_end):
            # print("Length of T is", len(T), "video_start", video_start, "video_end", video_end, "text_start", text_start, "text_end", text_end)
            # t = time()
            T = T.project(lambda vid, D: (vid, vid + 1, D))

            # T :: sfid, efid, D

            single_frame = T
            while True:
                # TODO: check whether original scallop program is off by one w.r.t. vid_end?
                # print(len(T))
                # filtered = T.filter(lambda vid_start, vid_end, D: int(vid_start) == int(video_start) and int(vid_end) == int(video_end))
                # # filtered = T.filter(lambda vid_start, vid_end, D: (vid_start == video_start) * (vid_end == video_end), batch_size=256)
                # if len(filtered) > 0:
                #     print(T)
                #     break
                if len(T) == 1:
                    break

                T = T.join(single_frame, key=lambda idx, vid_start, vid_mid, D: int(vid_mid), fkey=lambda idx, vid_mid, vid_end, D: int(vid_mid), batch_size=-1)
                # sfid, efid, D, efid, efid2, D2
                T = T.project(lambda vid_start, vid_mid1, D1, vid_mid2, vid_end, D2:
                             (vid_start, vid_end, D1.apply_if(D2,
                                                              lambda tid1, tid2: (tid1[0], tid2[1]),
                                                              lambda tid1, tid2: tid2[0] - 1 <= tid1[1] <= tid2[0])), batch_size=-1)
                # print(T)
                # print("\n\n\n")
            # exit()
            # print(T)
            # exit()
            d = T.project(lambda vid_start, vid_end, D: D).rows[0]
            # print("Number of operations", num_ops)
            # print("Time taken", time() - t)
            return d.map_symbols([(text_start, text_end)]).get_probabilities()[0]

        # t = time()
        # pred = video_mugen_action.join(text_mugen_action, key=lambda idx, *args: 0, fkey=lambda idx, *args: 0) \
        #     .project(lambda video_idx, video_T, text_idx, text_T: (video_idx, text_idx, map_action_to_tid(video_T, text_T))) \
        #     .join(video_start, key=lambda idx, i, *args: i, fkey=lambda idx, i, s: i) \
        #     .join(video_end, key=lambda idx, i, *args: i, fkey=lambda idx, i, e: i) \
        #     .join(text_start, key=lambda idx, _, i, *args: i, fkey=lambda idx, i, s: i) \
        #     .join(text_end, key=lambda idx, _, i, *args: i, fkey=lambda idx, i, e: i)
        
        # pred = pred.join(range(len(pred)))
        # # NUM_SAMPLES = len(pred)
        # # logger.stats["T_Join"] += (time() - t)
        # # t = time()
        # pred = pred.project(lambda vi, ti, T, vi2, video_start, vi3, video_end, ti2, text_start, ti3, text_end, idx:
        #                     (T, video_start, video_end, text_start, text_end))
        # pred_tvam = pred.project(text_video_action_match).rows

        # print(pred.rows[0][2])

        # pred = pred.project(lambda vi, ti, T, vi2, video_start, vi3, video_end, ti2, text_start, ti3, text_end, idx:
        #                     (T.project(lambda vid, D: D.apply(vid, lambda s, v: (int(v), int(v)+1, s[0], s[1]))), video_start, video_end, text_start, text_end))
        # pred_tvam = pred.project(text_video_action_match_recurse).rows

        # pred = pred.project(lambda vi, ti, T, vi2, video_start, vi3, video_end, ti2, text_start, ti3, text_end, idx:
        #             T.project(lambda *args: (idx, ) + args + ( video_start, video_end, text_start, text_end))).flatten()
                    
        # pred_tvam = text_video_action_match_batched(pred, NUM_SAMPLES)

        # logger.stats["T_Reason"] += (time() - t)
        
        # pred_match = torch.stack(pred_tvam).reshape(batch_size, batch_size)
        # print(pred_match)
        # exit()
        # TODO: handle constraint violation
        pred_constraint_violation = torch.zeros(batch_size, batch_size, device=self.device)

        self.processed_batch = (self.processed_batch + 1) % self.alternative_train_freq
        if self.processed_batch == 0:
            self.current_training_id = (self.current_training_id + 1) % len(self.modal2models)
        # print(pred_match)
        # exit()

        return pred_match, pred_constraint_violation