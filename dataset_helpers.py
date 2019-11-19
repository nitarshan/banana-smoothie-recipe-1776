from dataclasses import dataclass
import os
from pathlib import Path

import torch
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

def get_dataloaders(dataset_name: DatasetType, data_path: Path, use_cuda: bool = False) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader):
  kwargs = {'num_workers': 2 if use_cuda else 0, 'pin_memory': True}

  train, val, test = _get_datasets(dataset_name, 0.15, data_path)

  train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, **kwargs)
  val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=False, **kwargs)
  test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, **kwargs)

  return train_loader, val_loader, test_loader

def _get_datasets(dataset_name: DatasetType, val_split: float, data_path: Path) -> (torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset):
  mnist_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.1307,), (0.3081,)) # Normalize MNIST
  ])
  if dataset_name == DatasetType.MNIST:
    train, test = _get_torchvision_dataset(dataset_name, tv.datasets.MNIST, mnist_transforms, data_path)
  elif dataset_name == DatasetType.CIFAR10:
    train, test = _get_torchvision_dataset(dataset_name, tv.datasets.CIFAR10, mnist_transforms, data_path)
  elif dataset_name == DatasetType.CIFAR100:
    train, test = _get_torchvision_dataset(dataset_name, tv.datasets.CIFAR100, mnist_transforms, data_path)
  else:
    raise KeyError

  validation_size = round(len(train)*val_split)
  train, val = torch.utils.data.random_split(train, (len(train) - validation_size, validation_size))

  return train, val, test

def _get_torchvision_dataset(dataset_name: DatasetType, dataset_function, transforms, data_path: Path) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
  data_dir = data_path
  train = dataset_function(data_dir, train=True, download=True, transform=transforms)
  test = dataset_function(data_dir, train=False, download=True, transform=transforms)
  return train, test