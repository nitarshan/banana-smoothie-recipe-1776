from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_helpers import get_dataset_properties
from experiment_config import DatasetType, ComplexityType, EConfig, ModelType

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
    return torch.cat([p.view(-1) for p in self.parameters()], dim=0)
  
  def path_norm(self) -> torch.FloatTensor:
    raise NotImplementedError

  def get_complexity(self, output: torch.Tensor, complexity_type: ComplexityType) -> torch.Tensor:
    def prod_of_fro():
      return torch.prod(torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in self.parameters()]))
    def param_norm():
      return torch.sum(torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in self.parameters()]))
      
    if complexity_type == ComplexityType.NONE:
      return torch.zeros((1,), device=output.device)
    elif complexity_type == ComplexityType.L2:
      return torch.norm(self.get_flat_params(), p=2)
    elif complexity_type == ComplexityType.PROD_OF_FRO:
      return prod_of_fro()
    elif complexity_type == ComplexityType.SUM_OF_FRO:
      d = len(self.parameters())
      return d * prod_of_fro() ** (1/d)
    elif complexity_type == ComplexityType.PARAM_NORM:
      return param_norm()
    elif complexity_type == ComplexityType.PATH_NORM:
      return self.path_norm(output)
    raise KeyError

class DeepNet(ExperimentBaseModel):
  def __init__(self, hidden_sizes: List[int], dataset_type: DatasetType):
    super().__init__(dataset_type)
    self.hidden_size = hidden_sizes
    self.layers = nn.ModuleList(
      [nn.Linear(self.dataset_properties.D, hidden_sizes[0])] + # Input
      [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)] + # Hidden
      [nn.Linear(hidden_sizes[-1], self.dataset_properties.K)]) # Output

  def forward(self, x, complexity_type: ComplexityType):
    x = x.view(-1,self.dataset_properties.D)
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    x = self.layers[-1](x)
    x = F.log_softmax(x, dim=1)
    complexity = self.get_complexity(x, complexity_type)
    return x, complexity
  
  def path_norm(self, x):
    x = torch.ones((1,self.dataset_properties.D), device=x.device)
    for layer in self.layers[:-1]:
      x = F.relu(F.linear(x, layer.weight ** 2, layer.bias ** 2))
    x = F.linear(x, self.layers[-1].weight ** 2, self.layers[-1].bias ** 2)
    x = torch.sqrt(torch.sum(x))
    return x

# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
class ConvNet(ExperimentBaseModel):
  def __init__(self, dataset_type: DatasetType):
    super().__init__(dataset_type)
    self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, self.dataset_properties.K)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    x = F.log_softmax(x, dim=1)
    return x

  def path_norm(self, x):
    x = torch.ones((1,self.dataset_properties.D), device=x.device)
    x = F.conv2d(input, self.conv1.weight ** 2, self.conv1.bias ** 2, self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups)
    x = self.pool(F.relu(x))
    x = F.conv2d(input, self.conv2.weight ** 2, self.conv2.bias ** 2, self.conv2.stride, self.conv2.padding, self.conv2.dilation, self.conv2.groups)
    x = self.pool(F.relu(x))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(F.linear(x, self.fc1.weight ** 2, self.fc1.bias ** 2))
    x = F.relu(F.linear(x, self.fc2.weight ** 2, self.fc2.bias ** 2))
    x = F.linear(x, self.fc3.weight ** 2, self.fc3.bias ** 2)
    return x
