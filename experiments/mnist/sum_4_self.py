import os
import random
import datetime
import time
import wandb
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

from dolphin import Distribution
from dolphin.provenances import get_provenance

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSum2Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
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
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.mnist_dataset)

  def __getitem__(self, idx):
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx]]

    # Each data has two images and the GT is the sum of two digits
    return (a_img, 4 * a_digit)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    digits = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return ((a_imgs), digits)


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test):
  train_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
    batch_size=batch_size_train,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSum2Dataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum2Dataset.collate_fn,
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


class MNISTSum2Net(nn.Module):
  def __init__(self, db=None):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    a_imgs = x

    a = Distribution(self.mnist_net(a_imgs), range(10))
    res = a + a + a + a
    
    return res.get_probabilities() # Tensor b x 37


def bce_loss(output, ground_truth):
  gt = torch.nn.functional.one_hot(ground_truth, num_classes=37).float()
  return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)


class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, provenance, device, k):
    self.device = device
    self.model_dir = model_dir
    Distribution.provenance = get_provenance(provenance)
    Distribution.provenance.k = k
    # if k > 0:
    #   Distribution.k = k
    self.network = MNISTSum2Net().to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
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
      a_imgs = data.to(self.device)
      target = target.to(self.device)
      self.optimizer.zero_grad()
      
      output = self.network((a_imgs), self.db)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
      wandb.log({"epoch": epoch, "train/loss": loss})
    t_epoch = time.time() - t_begin_epoch
    wandb.log({"epoch": epoch, "train/it_time": t_epoch / len(iter)})

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        a_imgs = data.to(self.device)
        target = target.to(self.device)

        output = self.network((a_imgs), self.db)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
      wandb.log({"epoch": epoch, "test/loss": test_loss, "test/acc": perc})

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser()
  parser.add_argument("--n-epochs", type=int, default=3)
  parser.add_argument("--batch-size-train", type=int, default=64)
  parser.add_argument("--batch-size-test", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="dtkp-am", choices=['damp', 'dmmp', 'dtkp-am'])
  parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
  parser.add_argument("--topk", type=int, default=1)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size_train = args.batch_size_train
  batch_size_test = args.batch_size_test
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  provenance = args.provenance
  k = args.topk
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_sum_2"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test)

  if args.device == "cuda" and torch.cuda.is_available():
    device_name = "cuda"
  elif args.device == "mps" and torch.backends.mps.is_available():
    device_name = "mps"
  else:
    device_name = "cpu"

  device = torch.device(device_name)

  config = {
    "sum_n": "sum_4_self_src",
    "device": device,
    "provenance": provenance,
    "k": k,
    "seed": args.seed,
    "n_epochs": n_epochs,
    "batch_size_train": batch_size_train, 
    "batch_size_test": batch_size_test,
    "learning_rate": learning_rate,
    "experiment_type": "torchql",
  }

  timestamp = datetime.datetime.now()
  id = f'torchql_sum_4_self_{provenance}({k})_{args.seed}_{timestamp.strftime("%Y-%m-%d %H-%M-%S")}'

  wandb.login()
  wandb.init(project="MNIST-dtkp", config=config, id=id)
  wandb.define_metric("epoch")
  wandb.define_metric("train/it_time", step_metric="epoch", summary="mean")
  wandb.define_metric("test/loss", step_metric="epoch", summary="min")
  wandb.define_metric("test/acc", step_metric="epoch", summary="max")

  print(args)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, provenance, device, k)
  trainer.train(n_epochs)
