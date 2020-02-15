from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_helpers import get_dataset_properties
from experiment_config import DatasetType, ComplexityType, EConfig, ModelType
from torchvision.models import resnet18


class ExperimentBaseModel(nn.Module):
  def __init__(self, dataset_type: DatasetType):
    super().__init__()
    self.dataset_properties = get_dataset_properties(dataset_type)

  def forward(self, x) -> torch.Tensor:
    raise NotImplementedError

def get_model_for_config(e_config: EConfig) -> ExperimentBaseModel:
  if e_config.model_type == ModelType.DEEP:
    return DeepNet(e_config.model_shape, e_config.dataset_type)
  elif e_config.model_type == ModelType.CONV:
    return ConvNet(e_config.dataset_type)
  elif e_config.model_type == ModelType.RESNET:
    return ResNet(e_config.dataset_type)
  raise KeyError

class DeepNet(ExperimentBaseModel):
  def __init__(self, hidden_sizes: List[int], dataset_type: DatasetType):
    super().__init__(dataset_type)
    self.hidden_size = hidden_sizes
    self.layers = nn.ModuleList(
      [nn.Linear(np.prod(self.dataset_properties.D), hidden_sizes[0])] + # Input
      [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)] + # Hidden
      [nn.Linear(hidden_sizes[-1], self.dataset_properties.K)]) # Output

  def forward(self, x) -> torch.Tensor:
    x = x.view(-1, np.prod(self.dataset_properties.D))
    for layer in self.layers[:-1]:
      x = F.relu(layer(x))
    x = self.layers[-1](x)
    return x

# https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
class ConvNet(ExperimentBaseModel):
  def __init__(self, dataset_type: DatasetType):
    super().__init__(dataset_type)
    self.conv1 = nn.Conv2d(self.dataset_properties.D[0], 6, kernel_size=5)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, self.dataset_properties.K)

  def forward(self, x) -> torch.Tensor:
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
class ResNet(ExperimentBaseModel):
  def __init__(self, dataset_type: DatasetType):
    super().__init__(dataset_type)
    self.resnet = resnet18(pretrained=False, num_classes=self.dataset_properties.K)
    self.resnet.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.resnet.maxpool = nn.Identity()

  def forward(self, x) -> torch.Tensor:
    return self.resnet(x)
