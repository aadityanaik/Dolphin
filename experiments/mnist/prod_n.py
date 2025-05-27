import wandb
from datetime import datetime
import os
import time
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

from torchql import Query, Table, Database

from dolphin import Distribution
from dolphin.provenances import get_provenance

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTProdNDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    prod_n: int,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.prod_n = prod_n
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.mnist_dataset) / self.prod_n)

  def __getitem__(self, idx):
     # Get n data points
    imgs = ()
    prod = 1
    for i in range(self.prod_n):
      img, digit = self.mnist_dataset[self.index_map[idx*self.prod_n + i]]
      imgs = imgs + (img,)
      prod *= digit 
    # Each data has two images and the GT is the prod of n digits
    return (*imgs, prod)

  @staticmethod
  def collate_fn(batch):
    imgs = ()
    for i in range(len(batch[0])-1):
      a = torch.stack([item[i] for item in batch])
      imgs = imgs + (a,)
    digits = torch.stack([torch.tensor(item[len(batch[0])-1]).long() for item in batch])
    return ((imgs), digits)


def mnist_prod_n_loader(data_dir, prod_n, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTProdNDataset(
      data_dir,
      prod_n,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTProdNDataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTProdNDataset(
      data_dir,
      prod_n,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTProdNDataset.collate_fn,
    batch_size=batch_size_test,
    shuffle=True
  )

  return train_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTprodNNet(nn.Module):
  def __init__(self, db=None):
    super(MNISTprodNNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor], db: Database = None):
    for i in range(len(x)):
      if i == 0:
        a = Distribution(self.mnist_net(x[i]), range(10))
      else:
        a = a * Distribution(self.mnist_net(x[i]), range(10)) 

    return a.map_symbols(list(range(9**prod_n+1))).get_probabilities()


def bce_loss(output, ground_truth):
  gt = torch.nn.functional.one_hot(ground_truth, num_classes=(9**prod_n)+1).float()
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, provenance, device, k, prod_n):
    self.device = device
    self.model_dir = model_dir
    Distribution.provenance = get_provenance(provenance)
    # if k > 0:
    Distribution.provenance.k = k
    self.db = Database()
    self.network = MNISTprodNNet().to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.provenance = provenance
    self.prod_n = prod_n
    self.best_acc = 0
    if loss == "nll":
      self.loss = nll_loss
    elif loss == "bce":
      self.loss = bce_loss
    else:
      raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    t_begin_epoch = time.time()
    for (data, target) in iter:
      imgs = ()
      for x in range(self.prod_n):
        imgs = imgs + (data[x].to(self.device),)
      target = target.to(self.device)
      self.optimizer.zero_grad()
      
      output = self.network(imgs, self.db)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    total_epoch_time = time.time() - t_begin_epoch
    wandb.log(
      {
        "epoch": epoch,
        "total_epoch_time": total_epoch_time,
      }
    )
    print(f"Total Epoch Time: {total_epoch_time}")
  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        imgs = ()
        for x in range(prod_n):
          imgs = imgs + (data[x].to(self.device),)
        target = target.to(self.device)

        output = self.network(imgs, self.db)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        if perc > 97.00:
          # record prod_n + epoch number combination when accuracy is high
          file_path = f'torch_mnist_prod_n_{self.provenance}_epoch_count.log'
          current_timestamp = datetime.now()
          formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
          if os.path.exists(file_path):
            with open(file_path, 'a') as file:
              file.write(f'prod n={self.prod_n}, epoch num={epoch}, {formatted_timestamp}\n')
          else:
             with open(file_path, 'w') as file:
              file.write(f'prod n={self.prod_n}, epoch num={epoch}, {formatted_timestamp}\n')
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
        wandb.log(
          {
            "epoch": epoch,
            "accuracy": perc,
          }
        )
      if test_loss < self.best_loss:
        self.best_loss = test_loss
        # torch.save(self.network, os.path.join(model_dir, "prod_2_best.pt"))
      if perc > self.best_acc:
        self.best_acc = perc
      print(f"Best loss: {self.best_loss:.4f}")
      print(f"Best acc: {self.best_acc:.2f}%")

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser()
  parser.add_argument("--prod-n", type=int, default=4)
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="damp", choices=['damp', 'dmmp', 'dtkp-am'])
  parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
  parser.add_argument("--topk", type=int, default=1)
  args = parser.parse_args()

  print(args)

  # Parameters
  prod_n = args.prod_n
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  provenance = args.provenance
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  config = {
    "prod_n": prod_n,
    "n_epochs": n_epochs,
    "batch_size_train": batch_size_train, 
    "batch_size_test": batch_size_test,
    "provenance": provenance,
    "seed": args.seed,
    "experiment_type": "torchql", 
  }

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f'../../model/mnist_prod_{prod_n}'))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_prod_n_loader(data_dir, prod_n, batch_size_train, batch_size_test)

  if args.device == "cuda" and torch.cuda.is_available():
    device_name = "cuda"
  elif args.device == "mps" and torch.backends.mps.is_available():
    device_name = "mps"
  else:
    device_name = "cpu"

  device = torch.device(device_name)
  timestamp = datetime.now()
  id = f'torchql_prod{prod_n}_{args.seed}_{provenance}_{timestamp.strftime("%Y-%m-%d %H-%M-%S")}'

  wandb.init(
    project="WIP", config=config, id=id
  )
  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, provenance, device, args.topk, prod_n)
  trainer.train(n_epochs)