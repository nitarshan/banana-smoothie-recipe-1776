from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from .experiment_config import EConfig, DatasetType


def get_dataloaders(cfg: EConfig, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
  if cfg.dataset_type == DatasetType.MNIST:
    dataset = MNIST
  elif cfg.dataset_type == DatasetType.CIFAR10:
    dataset = CIFAR10
  else:
    raise KeyError

  train = dataset(cfg, device, train=True, download=True)
  test = dataset(cfg, device, train=False, download=True)

  train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
  train_eval_loader = DataLoader(train, batch_size=5000, shuffle=False, num_workers=0)
  test_loader = DataLoader(test, batch_size=5000, shuffle=False, num_workers=0)
  return train_loader, train_eval_loader, test_loader


def process_data(cfg: EConfig, data: torch.Tensor, targets: torch.Tensor, device: torch.device, train: bool):
  # Resize training dataset
  if train and (cfg.train_dataset_size is not None):
    rng = np.random.RandomState(cfg.data_seed) if (cfg.data_seed is not None) else np.random
    indices = rng.random.choice(len(data), cfg.train_dataset_size, replace=False)
    indices = torch.from_numpy(indices)
    data = torch.index_select(data, 0, indices)
    targets = torch.index_select(targets, 0, indices)

  # Label noise
  if cfg.label_noise is not None:
    mask = torch.rand_like(targets) <= cfg.label_noise
    noise = torch.randint_like(targets, cfg.dataset_type.K)
    targets[mask] = noise[mask]

  # Put both data and targets on GPU in advance
  return data.to(device), targets.to(device)


# https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
# We need to keep the class name the same as base class methods rely on it
class MNIST(tv.datasets.MNIST):
  def __init__(self, cfg: EConfig, device: torch.device, *args, **kwargs):
    super().__init__(cfg.data_dir, *args, **kwargs)

    # Scale data to [0,1]
    self.data = self.data.unsqueeze(1).float().div(255)

    # Normalize it with the usual MNIST mean and std
    self.data = self.data.sub_(0.1307).div_(0.3081)

    self.data, self.targets = process_data(cfg, self.data, self.targets, device, self.train)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]


class CIFAR10(tv.datasets.CIFAR10):
  def __init__(self, cfg: EConfig, device: torch.device, *args, **kwargs):
    super().__init__(cfg.data_dir, *args, **kwargs)

    # Scale data to [0,1] floats
    self.data = self.data / 255

    # Normalize data
    self.data = (self.data - self.data.mean(axis=(0,1,2))) / self.data.std(axis=(0,1,2))

    # NHWC -> NCHW
    self.data = self.data.transpose((0, 3, 1, 2))

    # Numpy -> Torch
    self.data = torch.tensor(self.data, dtype=torch.float32)
    self.targets = torch.tensor(self.targets, dtype=torch.long)

    self.data, self.targets = process_data(cfg, self.data, self.targets, device, self.train)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]
