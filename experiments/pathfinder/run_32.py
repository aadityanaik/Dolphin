from typing import Optional, Callable, Tuple
import os
import random
import itertools
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from datetime import datetime
from dolphin.provenances import get_provenance
from dolphin.distribution import Distribution
import torch.nn.functional as F
from time import time
import wandb 
import sys
import logging
import traceback

def exception_handler(exc_type, exc_value, exc_traceback):
    error_msg = f"An uncaught {exc_type.__name__} exception occurred:\n"
    error_msg += f"{exc_value}\n"
    error_msg += "Traceback:\n"
    error_msg += ''.join(traceback.format_tb(exc_traceback))

    logging.error(error_msg)

    print(error_msg, file=sys.stderr)

sys.excepthook = exception_handler

class PathFinder32Dataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  :param data_dir, the directory to the pathfinder dataset under the root folder
  :param difficulty, can be picked from "easy", "normal", "hard", and "all"
  """

  pathfinder_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

  def __init__(self, data_root: str, data_dir: str = "pathfinder32", difficulty: str = "all"):
    # Store
    self.transform = self.pathfinder_img_transform

    # Get subdirectories
    if difficulty == "all":
      sub_dirs = ["curv_baseline", "curv_contour_length_9", "curv_contour_length_14"]
    elif difficulty == "easy":
      sub_dirs = ["curv_baseline"]
    elif difficulty == "normal":
      sub_dirs = ["curv_contour_length_9"]
    elif difficulty == "hard":
      sub_dirs = ["curv_contour_length_14"]
    else:
      raise Exception(f"Unrecognized difficulty {difficulty}")

    # Get all image paths and their labels
    self.samples = []
    for sub_dir in sub_dirs:
      metadata_dir = os.path.join(data_root, data_dir, sub_dir, "metadata")
      for sample_group_file in os.listdir(metadata_dir):
        sample_group_dir = os.path.join(metadata_dir, sample_group_file)
        sample_group_file = open(sample_group_dir, "r")
        sample_group_lines = sample_group_file.readlines()[:-1]
        for sample_line in sample_group_lines:
          sample_tokens = sample_line[:-1].split(" ")
          sample_img_path = os.path.join(data_root, data_dir, sub_dir, sample_tokens[0], sample_tokens[1])
          sample_label = int(sample_tokens[3])
          self.samples.append((sample_img_path, sample_label))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, label) = self.samples[idx]
    img = Image.open(open(img_path, "rb"))
    if self.transform is not None:
      img = self.transform(img)
    return (img, label)

  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)


def pathfinder_32_loader(data_root, difficulty, batch_size, train_percentage):
  dataset = PathFinder32Dataset(data_root, difficulty=difficulty)
  num_train = int(len(dataset) * train_percentage)
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=PathFinder32Dataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=PathFinder32Dataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class CNNPathFinder32Net(nn.Module):
  def __init__(self, provenance="damp", k=3, num_block_x=6, num_block_y=6):
    super(CNNPathFinder32Net, self).__init__()
    Distribution.provenance = get_provenance(provenance)
    Distribution.provenance.k = k
    # block
    self.num_block_x = num_block_x
    self.num_block_y = num_block_y
    self.num_blocks = num_block_x * num_block_y
    self.block_coord_to_block_id = lambda x, y: y * num_block_x + x

    # Adjacency
    self.adjacency = self.build_adj(num_block_x, num_block_y)
    self.max_path_length = num_block_x * num_block_y

    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5),
      nn.Conv2d(32, 32, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.Conv2d(64, 64, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Flatten())

    # Fully connected for `is_endpoint`
    self.is_endpoint_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, self.num_blocks),
      nn.Sigmoid())

    # Fully connected for `connectivity`
    self.is_connected_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, len(self.adjacency)),
      nn.Sigmoid())
 
  def build_adj(self, num_block_x, num_block_y):
    adjacency = []
    block_coord_to_block_id = lambda x, y: y * num_block_x + x
    for i, j in itertools.product(range(num_block_x), range(num_block_y)):
      for (dx, dy) in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        x, y = i + dx, j + dy
        if x >= 0 and x < num_block_x and y >= 0 and y < num_block_y:
          source_id = block_coord_to_block_id(i, j)
          target_id = block_coord_to_block_id(x, y)
          adjacency.append((source_id, target_id))
    return adjacency 

  def extend_path(self, path, edge):
    if not path or not isinstance(path, tuple):  
        return (edge[1],) if edge else ()
    return (path[1], edge[1]) #if path[-1] == path[0] else ()
  def extend_path_cond(self, path, edge):
     return path[-1] == edge[0]
  def is_endpoint_cond(self, path, endpoint):
    return len(path) > 1 and path[0] != path[-1]
  
  def first_index_check(self, path, endpoint):
    return path[0] == endpoint
  
  def last_index_check(self, path, endpoint):
    return path[-1] == endpoint
  
  def is_endpoint(self, path, endpoint):
    if len(path) > 1 and path[0] != path[-1]:
      first = path.apply_if(endpoint, self.first_index_check, self.is_endpoint_cond)
      last = path.apply_if(endpoint, self.last_index_check, self.is_endpoint_cond)
      if first and last:
        return path
      else:
        return ()

  def build_paths_recursive(self, connected_dist, endpoint_dist):
    paths = self.build_path_recursive(connected_dist, connected_dist, connected_dist, endpoint_dist) 
    return paths
  
  def build_path_recursive(self, old_path, recent_path, edge, endpoint):
    new_path = recent_path.apply_if(edge, self.extend_path, self.extend_path_cond).drop_symbol(())
    merged_path = old_path | new_path
    if len(merged_path.symbols) == len(old_path.symbols):
        old_path = self.is_endpoint(old_path, endpoint)
        return old_path
    diffs = new_path.diff(old_path)
    return self.build_path_recursive(merged_path, diffs, edge, endpoint)

  def forward(self, image):
    embedding = self.cnn(image)
    is_connected = self.is_connected_fc(embedding)
    is_endpoint = self.is_endpoint_fc(embedding)
    connected_dist = Distribution(is_connected, self.adjacency)
    endpoint_dist = Distribution(is_endpoint, list(range(self.num_blocks)))
    t = time()
    paths = self.build_paths_recursive(connected_dist, endpoint_dist).drop_symbol(())
    wandb.log({"build_paths_time": time() - t})
    return paths.get_probabilities()[:,0]

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, provenance, k, gpu, save_model=False):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    self.network = CNNPathFinder32Net(provenance, k).to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

    # Aggregated loss (initialized to be a huge number)
    self.save_model = save_model
    self.min_test_loss = 100000000.0

  def loss(self, output, expected_output):
    return torch.mean(torch.square(output - expected_output)) 
  
  def accuracy(self, output, expected_output) -> Tuple[int, int]:
    diff = torch.abs(output - expected_output)
    num_correct = len([() for d in diff if d.item() < 0.4999])
    return (len(output), num_correct)

  def train(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    num_items = 0
    total_train_correct = 0
    t_begin_epoch = time()
    for (input, expected_output) in iter:
      self.optimizer.zero_grad()
      input = input.to(self.device)
      expected_output = expected_output.to(self.device)
      paths = self.network(input)
      loss = self.loss(paths, expected_output)
      loss.backward()
      self.optimizer.step()
      batch_size, num_correct = self.accuracy(paths, expected_output)
      num_items += batch_size
      total_train_correct += num_correct
      correct_perc = 100. * total_train_correct / num_items
      iter.set_description(f"[Train Epoch {epoch}] Batch Loss: {loss.item():.4f}, Overall Accuracy: {correct_perc:.4f}%")

    t_epoch = time() - t_begin_epoch
    wandb.log({"epoch": epoch, "train_time": t_epoch})

  def test(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (i, (input, expected_output)) in enumerate(iter):
        input = input.to(self.device)
        expected_output = expected_output.to(self.device)
        output = self.network(input)
        test_loss += self.loss(output, expected_output).item()
        batch_size, num_correct_in_batch = self.accuracy(output, expected_output)
        num_items += batch_size
        total_correct += num_correct_in_batch
        perc = 100. * total_correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {test_loss / (i + 1):.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save the model
    if self.save_model and test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      torch.save(self.network, "../model/pathfinder_32/pathfinder_32_net.pkl")
    
    wandb.log({
      "epoch": epoch,
      "test_accuracy": total_correct/num_items,
      "test_loss": test_loss,
    })

  def run(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train(epoch)
      self.test(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("pathfinder_32")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--train-percentage", type=float, default=0.9)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--difficulty", type=str, default="all")
  parser.add_argument("--provenance", type=str, default="dtkp-am")
  parser.add_argument("--top-k", type=int, default=1)
  args = parser.parse_args()

  # Setup parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  config = {
    "pathfinder_n": 32,
    "n_epochs": args.n_epochs,
    "batch_size": args.batch_size, 
    "provenance": args.provenance,
    "seed": args.seed,
    "learning_rate": args.learning_rate,
    "pathfinder_difficulty": args.difficulty,
    "experiment_type": "torch",
    "k": args.top_k,
  }

  timestamp = datetime.now()
  id = f'torch_pathfinder_32_applyk_{args.seed}_{args.provenance}_{timestamp.strftime("%Y-%m-%d %H-%M-%S")}'


  wandb.init(
    project="Pathfinder Apply-K", config=config, id=id
  )
  wandb.define_metric("epoch")
  wandb.define_metric("train_time", step_metric="epoch", summary="mean")
  wandb.define_metric("test_accuracy", step_metric="epoch", summary="max")
  wandb.define_metric("test_loss", step_metric="epoch", summary="min")
  wandb.define_metric("build_paths_time", step_metric="epoch", summary="mean")


  # Load data
  data_root = os.path.abspath("../data/")
  (train_loader, test_loader) = pathfinder_32_loader(data_root, args.difficulty, args.batch_size, args.train_percentage)
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.provenance, args.top_k, args.gpu)

  # Run!
  trainer.run(args.n_epochs)