from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_helpers import get_dataset_properties
from experiment_config import DatasetType, EConfig, ModelType

def get_model_for_config(e_config: EConfig) -> nn.Module:
  if e_config.model_type == ModelType.DEEP:
    return DeepNet([30, 30], e_config.dataset_type)
  elif e_config.model_type == ModelType.CONV:
    return ConvNet(e_config.dataset_type)
  raise KeyError

class ExperimentBaseModel(nn.Module):
  def __init__(self, dataset_type: DatasetType):
    super().__init__()
    self.dataset_properties = get_dataset_properties(dataset_type)
    self.layers = []

  def forward(self, x):
    raise NotImplementedError

  def get_flat_params(self) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in self.parameters()], dim=0)

  def get_norm(self, p=2) -> torch.FloatTensor:
    raise NotImplementedError
  
  @torch.no_grad()
  def get_num_params(self) -> int:
    return len(self.get_flat_params())

class DeepNet(ExperimentBaseModel):
  def __init__(self, hidden_sizes: List[int], dataset_type: DatasetType):
    super().__init__(dataset_type)
    self.hidden_size = hidden_sizes
    self.layers = nn.ModuleList(
      [nn.Linear(self.dataset_properties.D, hidden_sizes[0])] + # Input
      [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)] + # Hidden
      [nn.Linear(hidden_sizes[-1], self.dataset_properties.K)]) # Output

  def forward(self,x):
    x = x.view(-1,self.dataset_properties.D)
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    x = self.layers[-1](x)
    x = F.log_softmax(x, dim=1)
    return x

  def get_norm(self, p=2) -> torch.FloatTensor:
    return torch.norm(self.get_flat_params(), p=2)

# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
class ConvNet(ExperimentBaseModel):
  def __init__(self, dataset_type: DatasetType):
    super().__init__(dataset_type)
    self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    x = F.log_softmax(x, dim=1)
    return x

  def get_norm(self, p=2) -> torch.FloatTensor:
    return torch.norm(self.get_flat_params(), p=2)
