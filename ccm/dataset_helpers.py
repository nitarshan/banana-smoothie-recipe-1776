from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from .experiment_config import EConfig, DatasetType


def get_dataloaders(cfg: EConfig, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
  if cfg.dataset_type == DatasetType.MNIST:
    dataset = MNIST
  elif cfg.dataset_type == DatasetType.CIFAR10:
    dataset = CIFAR10
  else:
    raise KeyError

  train = dataset(cfg, device, train=True, download=True)
  test = dataset(cfg, device, train=False, download=True)

  val_split = 0.15
  validation_size = round(len(train) * val_split)
  train, val = torch.utils.data.random_split(train, (len(train) - validation_size, validation_size))

  train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
  train_eval_loader = DataLoader(train, batch_size=5000, shuffle=False, num_workers=0)
  val_loader = DataLoader(val, batch_size=5000, shuffle=False, num_workers=0)
  test_loader = DataLoader(test, batch_size=5000, shuffle=False, num_workers=0)
  return train_loader, train_eval_loader, val_loader, test_loader

def bootstrap_indices(seed: int, length: int, count: Optional[int]) -> torch.Tensor:
  rng = np.random.RandomState(seed)
  indices = torch.from_numpy(rng.randint(0, length, length if count is None else count))
  return indices

def apply_label_noise(target: torch.Tensor, cfg: EConfig):
  if cfg.label_noise is None or cfg.label_noise <= 0:
    return target
  mask = torch.rand_like(target) <= cfg.label_noise
  noise = torch.randint_like(target, cfg.dataset_type.K)
  target[mask] = noise[mask]
  return target

# https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
# We need to keep the class name the same as base class methods rely on it
class MNIST(tv.datasets.MNIST):
  def __init__(self, cfg: EConfig, device: torch.device, *args, **kwargs):
    super().__init__(cfg.data_dir, *args, **kwargs)

    # Scale data to [0,1]
    self.data = self.data.unsqueeze(1).float().div(255)

    # Normalize it with the usual MNIST mean and std
    self.data = self.data.sub_(0.1307).div_(0.3081)

    # Label noise
    self.targets = apply_label_noise(self.targets, cfg)

    # Bootstrap sample
    if cfg.data_seed is not None:
      indices = bootstrap_indices(cfg.data_seed, len(self.data), cfg.train_dataset_size)
      self.data = torch.index_select(self.data, 0, indices)
      self.targets = torch.index_select(self.targets, 0, indices)
    elif cfg.train_dataset_size is not None:
      rng = np.random.RandomState(cfg.seed)
      indices = torch.from_numpy(rng.choice(len(self.data), cfg.train_dataset_size, replace=False))
      self.data = torch.index_select(self.data, 0, indices)
      self.targets = torch.index_select(self.targets, 0, indices)

    # Put both data and targets on GPU in advance
    self.data, self.targets = self.data.to(device), self.targets.to(device)

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

    # Label noise
    self.targets = apply_label_noise(self.targets, cfg)

    # Bootstrap sample
    if cfg.data_seed is not None:
      indices = bootstrap_indices(cfg.data_seed, len(self.data), cfg.train_dataset_size)
      self.data = torch.index_select(self.data, 0, indices)
      self.targets = torch.index_select(self.targets, 0, indices)
    elif cfg.train_dataset_size is not None:
      rng = np.random.RandomState(cfg.seed)
      indices = torch.from_numpy(rng.choice(len(self.data), cfg.train_dataset_size, replace=False))
      self.data = torch.index_select(self.data, 0, indices)
      self.targets = torch.index_select(self.targets, 0, indices)

    # Put both data and targets on GPU in advance
    self.data, self.targets = self.data.to(device), self.targets.to(device)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]
