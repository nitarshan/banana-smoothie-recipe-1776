from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision as tv

from .experiment_config import Config, DatasetType, HParams


def get_dataloaders(hparams: HParams, config: Config, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
  if hparams.dataset_type == DatasetType.MNIST:
    dataset = MNIST
  elif hparams.dataset_type == DatasetType.CIFAR10:
    dataset = CIFAR10
  elif hparams.dataset_type == DatasetType.CIFAR100:
    dataset = CIFAR100
  elif hparams.dataset_type == DatasetType.SVHN:
    dataset = SVHN
  else:
    raise KeyError

  train_key = {'split': 'train'} if hparams.dataset_type == DatasetType.SVHN else {'train': True}
  test_key = {'split': 'test'} if hparams.dataset_type == DatasetType.SVHN else {'train': False}
  train = dataset(hparams, config, device, download=True, **train_key)
  test = dataset(hparams, config, device, download=True, **test_key)

  train_loader = DataLoader(train, batch_size=hparams.batch_size, shuffle=True, num_workers=0)
  train_eval_loader = DataLoader(train, batch_size=5000, shuffle=False, num_workers=0)
  test_loader = DataLoader(test, batch_size=5000, shuffle=False, num_workers=0)
  return train_loader, train_eval_loader, test_loader


def process_data(hparams: HParams, data: Tensor, targets: Tensor, device: torch.device, train: bool):
  # Resize dataset
  dataset_size = hparams.train_dataset_size if train else hparams.test_dataset_size
  offset = 0 if train else 1
  if dataset_size is not None:
    rng = np.random.RandomState(hparams.data_seed + offset) if (hparams.data_seed is not None) else np.random
    indices = rng.choice(len(data), dataset_size, replace=False)
    indices = torch.from_numpy(indices)
    data = torch.index_select(data, 0, indices)
    targets = torch.index_select(targets, 0, indices)

  # Label noise
  if hparams.label_noise is not None:
    mask = torch.rand_like(targets) <= hparams.label_noise
    noise = torch.randint_like(targets, hparams.dataset_type.K)
    targets[mask] = noise[mask]

  # Put both data and targets on GPU in advance
  return data.to(device), targets.to(device)


# https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
# We need to keep the class name the same as base class methods rely on it
class MNIST(tv.datasets.MNIST):
  def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
    super().__init__(config.data_dir, *args, **kwargs)

    # Scale data to [0,1]
    self.data = self.data.unsqueeze(1).float().div(255)

    # Normalize it with the usual MNIST mean and std
    self.data = self.data.sub_(0.1307).div_(0.3081)

    self.data, self.targets = process_data(hparams, self.data, self.targets, device, self.train)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]


class CIFAR10(tv.datasets.CIFAR10):
  def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
    super().__init__(config.data_dir, *args, **kwargs)

    # Scale data to [0,1] floats
    self.data = self.data / 255

    # Normalize data
    self.data = (self.data - self.data.mean(axis=(0,1,2))) / self.data.std(axis=(0,1,2))

    # NHWC -> NCHW
    self.data = self.data.transpose((0, 3, 1, 2))

    # Numpy -> Torch
    self.data = torch.tensor(self.data, dtype=torch.float32)
    self.targets = torch.tensor(self.targets, dtype=torch.long)

    self.data, self.targets = process_data(hparams, self.data, self.targets, device, self.train)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]


class CIFAR100(tv.datasets.CIFAR100):
  def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
    super().__init__(config.data_dir, *args, **kwargs)

    # Scale data to [0,1] floats
    self.data = self.data / 255

    # Normalize data
    self.data = (self.data - self.data.mean(axis=(0,1,2))) / self.data.std(axis=(0,1,2))

    # NHWC -> NCHW
    self.data = self.data.transpose((0, 3, 1, 2))

    # Numpy -> Torch
    self.data = torch.tensor(self.data, dtype=torch.float32)
    self.targets = torch.tensor(self.targets, dtype=torch.long)

    self.data, self.targets = process_data(hparams, self.data, self.targets, device, self.train)

  def __getitem__(self, index):
    return self.data[index], self.targets[index]


class SVHN(tv.datasets.SVHN):
  def __init__(self, hparams: HParams, config: Config, device: torch.device, *args, **kwargs):
    super().__init__(config.data_dir, *args, **kwargs)

    # Scale data to [0,1] floats
    self.data = self.data / 255

    # NCHWC -> NHWC (SVHN)
    self.data = self.data.transpose((0, 2, 3, 1))

    # Normalize data
    self.data = (self.data - self.data.mean(axis=(0,1,2))) / self.data.std(axis=(0,1,2))

    # NHWC -> NCHW
    self.data = self.data.transpose((0, 3, 1, 2))

    # Numpy -> Torch
    self.data = torch.tensor(self.data, dtype=torch.float32)
    self.labels = torch.tensor(self.labels, dtype=torch.long)

    self.data, self.labels = process_data(hparams, self.data, self.labels, device, self.split == 'train')

  def __getitem__(self, index):
    return self.data[index], self.labels[index]
