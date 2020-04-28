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
    return MLP(e_config.model_shape, e_config.dataset_type)
  elif e_config.model_type == ModelType.CONV:
    return ConvNet(e_config.dataset_type)
  elif e_config.model_type == ModelType.RESNET:
    depth = len(e_config.model_shape)
    width = e_config.model_shape[0]
    stack_planes = e_config.model_shape[2:]
    return ResNet(e_config.dataset_type, depth, width, [16,32,64])
  elif e_config.model_type == ModelType.NIN:
    depth = len(e_config.model_shape)
    width = e_config.model_shape[0]
    return NiN(depth, width, e_config.dataset_type)
  raise KeyError

class MLP(ExperimentBaseModel):
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


class NiNBlock(nn.Module):
  def __init__(self, inplanes: int, planes: int) -> None:
    super().__init__()
    self.relu = nn.ReLU(inplace=True)

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
    self.bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1)
    self.bn3 = nn.BatchNorm2d(planes)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    return x


class NiN(ExperimentBaseModel):
  def __init__(self, depth: int, width: int, dataset_type: DatasetType) -> None:
    super().__init__(dataset_type)

    self.base_width = 96

    blocks = []
    blocks.append(NiNBlock(self.dataset_properties.D[0], self.base_width*width))
    for _ in range(depth-1):
      blocks.append(NiNBlock(self.base_width*width,self.base_width*width))
    self.blocks = nn.Sequential(*blocks)

    self.relu = nn.ReLU(inplace=True)

    self.conv = nn.Conv2d(self.base_width*width, self.dataset_properties.K, kernel_size=1, stride=1)
    self.bn = nn.BatchNorm2d(10)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))

  def forward(self, x):
    x = self.blocks(x)

    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    
    x = self.avgpool(x)
    
    return x.squeeze()


class BasicBlock(nn.Module):
  def __init__(self, inplanes: int, planes: int, stack_idx: int, block_idx: int):
    super(BasicBlock, self).__init__()

    self.is_downsample_layer = (stack_idx > 1 and block_idx == 1)
    stride = 2 if self.is_downsample_layer else 1
    
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.downsample = nn.Sequential(
      nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
      nn.BatchNorm2d(planes),
    ) if self.is_downsample_layer else None

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

# https://gist.github.com/nitarshan/0dd8f076b37e6048575aadcefc155ae6
class ResNet(ExperimentBaseModel):
  def __init__(self, dataset_type: DatasetType, depth: int, width: int, stack_planes: List[int]):
    super(ResNet, self).__init__(dataset_type)
    
    prev_filters = stack_planes[0]
    inplanes = self.dataset_properties.D[0]

    self.conv1 = nn.Conv2d(inplanes, prev_filters * width, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(prev_filters * width)
    self.relu = nn.ReLU(inplace=True)

    stacks = []
    for stack_idx, filters in enumerate(stack_planes):
      stacks.append(self._make_stack(prev_filters * width, filters * width, depth, stack_idx + 1))
      prev_filters = filters
    self.stacks = nn.Sequential(*stacks)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(stack_planes[-1] * width, self.dataset_properties.K)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_stack(self, inplanes, planes, depth, stack_idx: int):
    layers = []
    layers.append(BasicBlock(inplanes, planes, stack_idx, 1))
    for block_idx in range(2, depth+1):
      layers.append(BasicBlock(planes, planes, stack_idx, block_idx))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.stacks(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x