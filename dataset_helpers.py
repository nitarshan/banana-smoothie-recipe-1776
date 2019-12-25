from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision as tv

from experiment_config import DatasetType

@dataclass(frozen=True)
class DatasetProperties:
  name: DatasetType
  D: int
  K: int
  is_classification: bool

def get_dataset_properties(dataset_name: DatasetType) -> DatasetProperties:
  if dataset_name == DatasetType.MNIST:
    return DatasetProperties(DatasetType.MNIST, 28*28, 10, True)
  elif dataset_name == DatasetType.CIFAR10:
    return DatasetProperties(DatasetType.CIFAR10, 3*32*32, 10, True)
  elif dataset_name == DatasetType.CIFAR100:
    return DatasetProperties(DatasetType.CIFAR100, 3*32*32, 100, True)
  raise KeyError()

def get_dataloaders(dataset_name: DatasetType, data_path: Path, batch_size: int, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
  if dataset_name == DatasetType.MNIST:
    train = MNIST(device, data_path, train=True, download=True)
    test = MNIST(device, data_path, train=False, download=True)
  elif dataset_name == DatasetType.CIFAR10:
    train = CIFAR10(device, data_path, train=True, download=True)
    test = CIFAR10(device, data_path, train=False, download=True)
  else:
    raise KeyError

  val_split = 0.15
  validation_size = round(len(train) * val_split)
  train, val = torch.utils.data.random_split(train, (len(train) - validation_size, validation_size))

  train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(val, batch_size=5000, shuffle=False, num_workers=0)
  test_loader = DataLoader(test, batch_size=5000, shuffle=False, num_workers=0)
  return train_loader, val_loader, test_loader

# https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
# We need to keep the class name the same as base class methods rely on it
class MNIST(tv.datasets.MNIST):
  def __init__(self, device, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # Scale data to [0,1]
    self.data = self.data.unsqueeze(1).float().div(255)
    
    # Normalize it with the usual MNIST mean and std
    self.data = self.data.sub_(0.1307).div_(0.3081)
    
    # Put both data and targets on GPU in advance
    self.data, self.targets = self.data.to(device), self.targets.to(device)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]

class CIFAR10(tv.datasets.CIFAR10):
  def __init__(self, device, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.data = torch.from_numpy(self.data.transpose((0, 3, 1, 2)))
    self.targets = torch.tensor(self.targets, dtype=torch.long)

    # Scale data to [0,1]
    self.data = self.data.float().div(255)
    
    # Normalize it
    self.data = self.data.sub_(0.5).div_(0.5)
    
    # Put both data and targets on GPU in advance
    self.data, self.targets = self.data.to(device), self.targets.to(device)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]
