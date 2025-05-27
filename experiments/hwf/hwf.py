from collections import defaultdict
import logging
import os
import json
import random
from argparse import ArgumentParser
from time import time
import numpy as np
from tqdm import tqdm
import math
from datetime import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from PIL import Image

import math

from torchql import Database, Query
from dolphin import get_provenance
from dolphin import Distribution
import wandb
import sys
import traceback

def exception_handler(exc_type, exc_value, exc_traceback):
    error_msg = f"An uncaught {exc_type.__name__} exception occurred:\n"
    error_msg += f"{exc_value}\n"
    error_msg += "Traceback:\n"
    error_msg += ''.join(traceback.format_tb(exc_traceback))

    logging.error(error_msg)

    print(error_msg, file=sys.stderr)

sys.excepthook = exception_handler

class HWFDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, prefix: str, split: str, l):
    super(HWFDataset, self).__init__()
    self.root = root
    self.split = split
    md = json.load(open(os.path.join(root, f"HWF/hwf_{l}_{split}.json")))
    # finding only the metadata with length == 1
    if l > 0:
      self.metadata = [m for m in md if len(m['img_paths']) <= l]
    else:
      self.metadata = md

    self.img_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (1,))
    ])

  def __getitem__(self, index):
    sample = self.metadata[index]

    # Input is a sequence of images
    img_seq = []
    for img_path in sample["img_paths"]:
      img_full_path = os.path.join(self.root, "HWF/Handwritten_Math_Symbols", img_path)
      img = Image.open(img_full_path).convert("L")
      img = self.img_transform(img)
      img_seq.append(img)
    img_seq_len = len(img_seq)

    # Output is the "res" in the sample of metadata
    res = sample["res"]

    # Return (input, output) pair
    return (img_seq, img_seq_len, res)

  def __len__(self):
    return len(self.metadata)

  @staticmethod
  def collate_fn(batch):
    max_len = max([img_seq_len for (_, img_seq_len, _) in batch])
    zero_img = torch.zeros_like(batch[0][0][0])
    pad_zero = lambda img_seq: img_seq + [zero_img] * (max_len - len(img_seq))
    img_seqs = torch.stack([torch.stack(pad_zero(img_seq)) for (img_seq, _, _) in batch])
    img_seq_len = torch.stack([torch.tensor(img_seq_len).long() for (_, img_seq_len, _) in batch])
    results = torch.stack([torch.tensor(res) for (_, _, res) in batch])
    return (img_seqs, img_seq_len, results)


def hwf_loader(data_dir, batch_size, prefix, l):
  train_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "train", l), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "test", l), collate_fn=HWFDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class SymbolNet(nn.Module):
  def __init__(self):
    super(SymbolNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
    self.fc1 = nn.Linear(30976, 128)
    self.fc1_bn = nn.BatchNorm1d(128)
    self.fc2 = nn.Linear(128, 14)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.max_pool2d(x, 2)
    x = F.dropout(x, p=0.25, training=self.training)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.fc1_bn(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)

class HWFNet(nn.Module):
  def __init__(self, provenance, k, debug=False):
    super(HWFNet, self).__init__()

    # Symbol embedding
    self.symbol_cnn = SymbolNet()
    self.operators = [("+", ), ("-", ), ("*", ), ("/", )]
    self.symbols = [ (str(i),) for i in range(10)] + self.operators

    Distribution.provenance = get_provenance(provenance)
    Distribution.k = k

  def forward(self, img_seq, img_seq_len, db):
    batch_size, formula_length, _, _, _ = img_seq.shape
    length = [l.item() for l in img_seq_len]

    inp = img_seq.flatten(start_dim=0, end_dim=1)
    
    t = time()
    symbol = self.symbol_cnn(inp).view(batch_size, -1, 14)
    wandb.log({
      "hwfnet_symbol_cnn_time": time()-t,
    })

    def eval_formula(s):
      try:
        return eval("".join(s))
      except:
        return math.nan
      
    def concat_symbol(formula, symbol):
      if formula[-1] == "":
        return formula
      else:
        # print(f'symbol is {symbol} formula is {formula}')
        if not isinstance(symbol, tuple):
          symbol = (symbol,)
        formula += symbol
        if len(formula) % 2 == 1 and len(formula) > 1:
          # formula has at least 1 expression:
          # a <op> b ...
          # we can evaluate the last 3 symbols only if the last operator is a multiplication or division

          if formula[-2] in ["*", "/"]:
            eval_result = str(eval_formula(formula[-3:]))
            formula = formula[:-3] + (eval_result,)
        return formula

    def infer_expression(length, *symbols):
      t = time()
      res = symbols[0]
      for i in range(1, len(symbols)):
     #   print("BEFORE:", res.symbols, symbols[i].symbols)
        # res += symbols[i]
        # print(f'res {res} symbols[i] {symbols[i]}')
        res = res.apply(symbols[i], concat_symbol)
     #   print("AFTER", res.symbols)
        # input()
      # exit()
      x = (res.map(eval_formula), )
      wandb.log({
        "hwfnet_infer_expression_time": time()-t,
      })
      return x

    def reorg(symbols, lengths):
      t = time()
      distrs = []
      for i in range(symbol.shape[1]):
        if i < lengths:
          distrs.append(Distribution(symbols[i, :].view(-1, 14), self.symbols))
          if i % 2 == 0:
            distrs[-1] = distrs[-1].filter(lambda s : s not in self.operators)
          else:
            distrs[-1] = distrs[-1].filter(lambda s : s in self.operators)
        else:
          distrs.append(Distribution(torch.ones(1, device=device), [("",), ]))

      res = (lengths, *distrs)
      wandb.log({
        "hwfnet_reorg_time": time()-t,
      })
      return res
    

    q = Query("hwf", base="symbols").join("lengths").project(lambda symbols, lengths: reorg(symbols, lengths)) \
      .project(infer_expression, batch_size=batch_size)

    t = time()
    res = q(db, tensors={"symbols": symbol, "lengths": length}, disable=True)
    
    stacked = Distribution.stack(res.rows)
    wandb.log({
      "hwfnet_query_time": time()-t,
    })
    return stacked

class Trainer():
  def __init__(self, train_loader, test_loader, device, model_root, model_name, learning_rate, provenance, k, step_size=10, gamma=0.1):
    self.network = HWFNet(provenance, k).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate) #, weight_decay=0.01)
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    self.db = Database()
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.device = device
    self.loss_fn = F.binary_cross_entropy
    self.model_root = model_root
    self.model_name = model_name
    self.min_test_loss = 100000000.0
    self.best_accuracy = 0.0

  def eval_result_eq(self, a, b, threshold=0.01):
    if a is None or b is None:
      return False
    result = abs(a - b) < threshold
    return result

  def retrieve_y(self, label, s, threshold=0.01):
    num_labels = len(label)
    num_symbols = len(s)

    label_2d = label.view(-1, 1).expand(-1, num_symbols)
    symbols_2d = s.view(1, -1).expand(num_labels, -1)

    return (torch.abs(label_2d - symbols_2d) < threshold).float()

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    t_begin_total_epoch = time()
    
    for (i, (img_seq, img_seq_len, label)) in enumerate(iter):
      t_begin = time()
      self.optimizer.zero_grad()

      t = time()
      img_seq, img_seq_len, label = img_seq.to(device), img_seq_len.to(device), label.to(device)

      distributions = self.network(img_seq, img_seq_len, self.db)
      d = distributions[0]
      
      s = d.symbols
      y_pred = d.get_probabilities()

      if len(y_pred.shape) == 1:
        y_pred = y_pred.view(1, -1)
      batch_size, num_outputs = y_pred.shape
      
      t = time()
      y = self.retrieve_y(label, torch.tensor(s.astype(float), device=device))

      # Compute loss
      t = time()
      loss = self.loss_fn(y_pred, y)
      loss.backward()
      self.optimizer.step()
      if not math.isnan(loss.item()):
        train_loss += loss.item()

      # Collect index and compute accuracy
      t = time()
      if num_outputs > 0:
        y_index = torch.argmax(y, dim=1)
        y_pred_index = torch.argmax(y_pred, dim=1)
        correct_count = torch.sum(torch.where(torch.sum(y, dim=1) > 0, y_index == y_pred_index, torch.zeros(batch_size, device=device).bool())).item()
      else:
        correct_count = 0

      # Stats
      num_items += batch_size
      total_correct += correct_count
      perc = 100. * total_correct / num_items
      avg_loss = train_loss / (i + 1)

      # Prints
      iter.set_description(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, LR: {self.scheduler.get_lr()}, Acc: {total_correct}/{num_items} ({perc:.2f}%)")
    total_epoch_time = time() - t_begin_total_epoch
    wandb.log(
      {
        "epoch": epoch,
        "total_epoch_time": total_epoch_time,
      }
    )
    print(f"Total Epoch Time: {total_epoch_time}")
    self.scheduler.step()

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    total_correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for i, (img_seq, img_seq_len, label) in enumerate(iter):
        distributions = self.network(img_seq.to(device), img_seq_len.to(device), self.db)

        d = distributions[0]

        # Normalize label format
        s = d.symbols
        y_pred = d.get_probabilities()
        if len(y_pred.shape) == 1:
          y_pred = y_pred.view(1, -1)
        batch_size, num_outputs = y_pred.shape
        y = torch.tensor([1.0 if self.eval_result_eq(l.item(), m) else 0.0 for l in label for m in s.astype(float)], device=device).view(batch_size, -1)
        
        # Compute loss
        loss = self.loss_fn(y_pred, y)
        if not math.isnan(loss.item()):
          test_loss += loss.item()

        # Collect index and compute accuracy
        if num_outputs > 0:
          y_index = torch.argmax(y, dim=1)
          y_pred_index = torch.argmax(y_pred, dim=1)
          correct_count = torch.sum(torch.where(torch.sum(y, dim=1) > 0, y_index == y_pred_index, torch.zeros(batch_size, device=device).bool())).item()
        else:
          correct_count = 0

        # Stats
        num_items += batch_size
        total_correct += correct_count
        perc = 100. * total_correct / num_items
        avg_loss = test_loss / (i + 1)

        # Prints
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {total_correct}/{num_items} ({perc:.2f}%)")

    # Save model
    if test_loss < self.min_test_loss:
      self.min_test_loss = test_loss
      torch.save(self.network, os.path.join(self.model_root, self.model_name))

    if perc > self.best_accuracy:
      self.best_accuracy = perc
    wandb.log(
          {
            "epoch": epoch,
            "test_accuracy": perc,
            "test_loss": test_loss
          }
        )

  def train(self, n_epochs):
    def compare_weights(w1, w2):
      for p1, p2 in zip(w1, w2):
        if not torch.equal(p1, p2):
          return True
      return False
    
    def get_weights(model):
      weights = []

      for param in model.parameters():
          weights.append(param.clone())

      return weights
    
    # params_init = get_weights(self.network)
    # self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)
      # logging.debug(f"Did the weights change? {compare_weights(params_init, get_weights(self.network))}")


if __name__ == "__main__":
  # Command line arguments
  parser = ArgumentParser("hwf")
  parser.add_argument("--model-name", type=str, default="hwf.pkl")
  parser.add_argument("--n-epochs", type=int, default=20)
  parser.add_argument("--no-sample-k", action="store_true")
  parser.add_argument("--sample-k", type=int, default=7)
  parser.add_argument("--l", type=int, default=7)
  parser.add_argument("--dataset-prefix", type=str, default="expr")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--step-size", type=int, default=10)
  parser.add_argument("--gamma", type=float, default=0.1)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--do-not-use-hash", action="store_true")
  parser.add_argument("--provenance", type=str, default="dtkp-am")
  parser.add_argument("--top-k", default=3)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--recompile", action="store_true")
  parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--log-level", type=str, default="INFO", choices=["INFO", "DEBUG", "WARNING"])
  parser.add_argument("--log-file", type=str, default=None)
  args = parser.parse_args()

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  
  if args.device == "cuda" and torch.cuda.is_available():
    device_name = f"cuda:{args.gpu}"
  elif args.device == "mps" and torch.backends.mps.is_available():
    device_name = "mps"
  else:
    device_name = "cpu"

  device = torch.device(device_name)

  config = {
    "hwf_n": args.l,
    "n_epochs": args.n_epochs,
    "batch_size": args.batch_size, 
    "provenance": args.provenance,
    "seed": args.seed,
    "experiment_type": "torch symbolic gpu", 
  }

  timestamp = datetime.now()
  id = f'torch_hwf{args.l}_{args.seed}_{args.provenance}_{timestamp.strftime("%Y-%m-%d %H-%M-%S")}'


  wandb.init(
    project="HWF-N", config=config, id=id
  )
  wandb.define_metric("epoch")
  wandb.define_metric("total_epoch_time")
  wandb.define_metric("train_time_per_epoch", step_metric="epoch", summary="mean")
  wandb.define_metric("test_accuracy", step_metric="epoch", summary="max")
  wandb.define_metric("test_loss", step_metric="epoch", summary="min")
  wandb.define_metric("hwfnet_symbol_cnn_time", summary="mean")
  wandb.define_metric("hwfnet_infer_expression_time", summary="mean")
  wandb.define_metric("hwfnet_reorg_time", summary="mean")
  wandb.define_metric("hwfnet_query_time", summary="mean")
 
  handler = [logging.StreamHandler()]
  if args.log_file:
    handler = [logging.FileHandler(args.log_file, mode="w")]

  logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=args.log_level, handlers=handler)

  # Data
  data_dir = os.path.abspath(os.path.join("../data/"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/hwf"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)
  train_loader, test_loader = hwf_loader(data_dir, batch_size=args.batch_size, prefix=args.dataset_prefix, l=args.l)

  k = int(args.top_k) if args.top_k is not None else None

  # Training
  trainer = Trainer(train_loader, test_loader, device, model_dir, args.model_name, args.learning_rate, args.provenance, k, step_size=args.step_size, gamma=args.gamma)
  trainer.train(args.n_epochs)
  print(f"Best accuracy: {trainer.best_accuracy:.2f}%")
