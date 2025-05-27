import os
import torch
import re
import json
from torchvision.io import write_video, read_video
import torch.utils.data as data
import random

def split_text(text):
    regex_expr = "(, and|and)(?! a | gems| coins)"
    regex_expr1 = ", and |and"
    subsentences = [t.strip() for t in re.split(regex_expr, text) if not ('and' == t or ', and' == t)]

    return subsentences

class SVMugenDataset(data.Dataset):
    def __init__(
            self,
            args,
            split='train',
    ):
        self.args = args
        self.train = split == 'train'
        self.max_label = 21

        dataset_json_file = os.path.join(self.args.data_dir, f"{split}_mini.json")
        self.data = json.load(open(dataset_json_file, 'r'))['data']
        if split == 'train' and args.train_size:
            splitRandom = random.Random(1234)
            idxs = list(range(len(self.data)))
            splitRandom.shuffle(idxs)
            idxs = idxs[:args.train_size]
            self.data = [self.data[i] for i in idxs]
        self.video_save_dir = args.video_save_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # self.load_json_file(idx)
        game_name = self.data[idx]["video"]["json_file"].split('/')[-1][:-5]

        result_dict = {}
        result_dict['game_name'] = game_name

        save_video_file = os.path.join(self.video_save_dir, f"{game_name}.mp4")
        game_video, _, _ = read_video(save_video_file, pts_unit="sec")
        result_dict['video'] = game_video

        text_desc = self.data[idx]["annotations"][0]["text"]
        result_dict['text'] = split_text(text_desc)

        return result_dict

    def collate_fn(self, result_dict_ls):
        combined_result_dict = {}
        result_keys = list(result_dict_ls[0].keys())
        for key in result_keys:
            combined_result_dict[key] = []
        combined_result_dict['text_idx'] = []
        current_text_id = 0

        for result_dict in result_dict_ls:
            for key in result_keys:
                if key == "game_name":
                    combined_result_dict[key].append(result_dict[key])
                else:
                    combined_result_dict[key] += result_dict[key]
                if key == 'audio' or key == 'video_smap':
                    raise NotImplementedError
                if key == 'text':
                    next_text_id = current_text_id + len(result_dict[key])
                    combined_result_dict['text_idx'].append((current_text_id, next_text_id))
                    current_text_id = next_text_id

        if 'video' in combined_result_dict:
            # Group the video
            combined_result_dict['video'] = torch.stack(combined_result_dict['video'])
            # combined_result_dict['video'] = combined_result_dict['video'].unfold(0, 4, 2).reshape(-1, 4, 256, 256, 3) // learns slower
            combined_result_dict['video'] = combined_result_dict['video'].reshape(-1, 2, 256, 256, 3)

        return combined_result_dict
