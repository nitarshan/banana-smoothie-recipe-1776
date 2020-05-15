from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from .experiment_config import DatasetType


def get_dataloaders(dataset_name: DatasetType, data_path: Path, batch_size: int, device: torch.device, seed: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
  if dataset_name == DatasetType.MNIST:
    train = MNIST(device, seed, data_path, train=True, download=True)
    test = MNIST(device, None, data_path, train=False, download=True)
  elif dataset_name == DatasetType.CIFAR10:
    train = CIFAR10(device, seed, data_path, train=True, download=True)
    test = CIFAR10(device, None, data_path, train=False, download=True)
  else:
    raise KeyError

  val_split = 0.15
  validation_size = round(len(train) * val_split)
  train, val = torch.utils.data.random_split(train, (len(train) - validation_size, validation_size))

  train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
  train_eval_loader = DataLoader(train, batch_size=5000, shuffle=False, num_workers=0)
  val_loader = DataLoader(val, batch_size=5000, shuffle=False, num_workers=0)
  test_loader = DataLoader(test, batch_size=5000, shuffle=False, num_workers=0)
  return train_loader, train_eval_loader, val_loader, test_loader

def bootstrap_indices(seed: int, length: int) -> torch.Tensor:
  rng = np.random.RandomState(seed)
  indices = torch.from_numpy(rng.randint(0, length, length))
  return indices

# https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
# We need to keep the class name the same as base class methods rely on it
class MNIST(tv.datasets.MNIST):
  def __init__(self, device: torch.device, seed: Optional[int], *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Scale data to [0,1]
    self.data = self.data.unsqueeze(1).float().div(255)
    
    # Normalize it with the usual MNIST mean and std
    self.data = self.data.sub_(0.1307).div_(0.3081)
    
    # Put both data and targets on GPU in advance
    self.data, self.targets = self.data.to(device), self.targets.to(device)

    # Bootstrap sample
    if seed is not None:
      indices = bootstrap_indices(seed, len(self.data))
      self.data = torch.index_select(self.data, 0, indices)
      self.targets = torch.index_select(self.targets, 0, indices)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]

class CIFAR10(tv.datasets.CIFAR10):
  def __init__(self, device: torch.device, seed: Optional[int], *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Scale data to [0,1] floats
    self.data = self.data / 255

    # Normalize data
    self.data = (self.data - self.data.mean(axis=(0,1,2))) / self.data.std(axis=(0,1,2))

    # NHWC -> NCHW
    self.data = self.data.transpose((0, 3, 1, 2))

    # Numpy -> Torch
    self.data = torch.tensor(self.data, dtype=torch.float32)
    self.targets = torch.tensor(self.targets, dtype=torch.long)

    # Bootstrap sample
    if seed is not None:
      indices = bootstrap_indices(seed, len(self.data))
      self.data = torch.index_select(self.data, 0, indices)
      self.targets = torch.index_select(self.targets, 0, indices)
    
    # Put both data and targets on GPU in advance
    self.data, self.targets = self.data.to(device), self.targets.to(device)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]
