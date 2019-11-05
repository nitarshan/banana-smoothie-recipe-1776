from dataclasses import dataclass
import os

import numpy as np
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
  elif dataset_name == DatasetType.REGRESSION:
    return DatasetProperties(DatasetType.REGRESSION, 1, 1, False)
  raise KeyError()

def get_dataloaders(dataset_name: DatasetType) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader):
  kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}

  train, val, test = _get_datasets(dataset_name, 0.15)

  train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, **kwargs)
  val_loader = torch.utils.data.DataLoader(val, batch_size=128, shuffle=False, **kwargs)
  test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=False, **kwargs)

  return train_loader, val_loader, test_loader

def _get_datasets(dataset_name: DatasetType, val_split: float) -> (torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset):
  transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.1307,), (0.3081,))
  ])
  if dataset_name == DatasetType.MNIST:
    train, test = _get_torchvision_dataset(dataset_name, tv.datasets.MNIST, transforms)
  elif dataset_name == DatasetType.CIFAR10:
    train, test = _get_torchvision_dataset(dataset_name, tv.datasets.CIFAR10, transforms)
  elif dataset_name == DatasetType.CIFAR100:
    train, test = _get_torchvision_dataset(dataset_name, tv.datasets.CIFAR100, transforms)
  elif dataset_name == DatasetType.REGRESSION:
    train, test = _get_regression_dataset()
  else:
    raise KeyError

  validation_size = round(len(train)*val_split)
  train, val = torch.utils.data.random_split(train, (len(train) - validation_size, validation_size))

  return train, val, test

def _get_torchvision_dataset(dataset_name: DatasetType, dataset_function, transforms) -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
  data_dir = os.path.join('data', dataset_name.name)
  train = dataset_function(data_dir, train=True, download=True, transform=transforms)
  test = dataset_function(data_dir, train=False, download=True, transform=transforms)
  return train, test

def _get_regression_dataset() -> (torch.utils.data.Dataset, torch.utils.data.Dataset):
  # Train
  s = np.random.uniform(-1.25,1.25,100)
  theta = np.sin(2*np.pi*s) + np.random.normal(scale=0.1,size=np.shape(s)[0])
  s = np.reshape(s,[-1,1])
  theta = np.reshape(theta,[-1,1])
  s = torch.tensor(s, dtype=torch.float64)
  theta = torch.tensor(theta, dtype=torch.float64)

  # Test
  rng = np.random.RandomState(0)
  p = rng.uniform(-1.25, 1.25, 1000)
  theta_p = np.sin(2*np.pi*p) + rng.random.normal(scale=0.1,size=np.shape(p)[0])
  p = np.reshape(p,[-1,1])
  theta_p = np.reshape(theta_p,[-1,1])
  p = torch.tensor(p, dtype=torch.float64)
  theta_p = torch.tensor(theta_p, dtype=torch.float64)

  train = torch.utils.data.TensorDataset(s, theta)
  test = torch.utils.data.TensorDataset(p, theta_p)

  return train, test
