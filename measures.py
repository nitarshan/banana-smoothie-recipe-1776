
from copy import deepcopy
from experiment_config import ComplexityType
from models import ExperimentBaseModel
from typing import List

import torch

# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
def _path_norm(model: ExperimentBaseModel, device: torch.device):
  model = deepcopy(model)
  model.eval()
  for param in model.parameters():
    if param.requires_grad:
      param.pow_(2)
  x = torch.ones([1] + model.dataset_properties.D, device=device)
  x = model(x)
  return x.sum().sqrt()

def _prod_of_fro(weights: List[torch.nn.Parameter]):
  weight_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in weights])
  return weight_norms.log().sum().exp()

def _param_norm(weights: List[torch.nn.Parameter]):
  weight_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in weights])
  return weight_norms.sum()

def _spec_norm(weights: List[torch.nn.Parameter]):
  weight_norms = torch.cat([p.svd()[1].max().unsqueeze(0) for p in weights])
  return weight_norms.log().sum().exp()

def _fft_spec_norm(weights: List[torch.nn.Parameter]):
  weight_norms = torch.cat([p.svd()[1].max().unsqueeze(0) for p in weights])
  return weight_norms.log().sum().exp()

def _fro_dist(weights: List[torch.nn.Parameter], init_weights: List[torch.nn.Parameter]):
  weight_norms = torch.cat([
    (p - q).norm('fro').unsqueeze(0) ** 2
    for p, q in zip(weights, init_weights)])
  return weight_norms.sum()

def get_measure(model: ExperimentBaseModel, init_model: ExperimentBaseModel, measure_type: ComplexityType, device: torch.device) -> torch.Tensor:
  weights_only = [p for name, p in model.named_parameters() if 'bias' not in name]
  init_weights_only = [p for name, p in init_model.named_parameters() if 'bias' not in name]
  d = len(weights_only)

  if measure_type == ComplexityType.NONE:
    return torch.zeros((1,), device=device)
  elif measure_type == ComplexityType.L2:
    flat_params = torch.cat([p.view(-1) for p in weights_only], dim=0)
    return flat_params.norm(p=2)
  elif measure_type == ComplexityType.PROD_OF_FRO:
    return _prod_of_fro(weights_only)
  elif measure_type == ComplexityType.SUM_OF_FRO:
    return d * _prod_of_fro(weights_only) ** (1/d)
  elif measure_type == ComplexityType.PARAM_NORM:
    return _param_norm(weights_only)
  elif measure_type == ComplexityType.PATH_NORM:
    return _path_norm(model, device)
  elif measure_type == ComplexityType.PARAMS:
    return torch.tensor(sum(p.numel() for p in model.parameters() if p.requires_grad))
  elif measure_type == ComplexityType.PROD_OF_SPEC:
    return _spec_norm(weights_only)
  elif measure_type == ComplexityType.SUM_OF_SPEC:
    return d * _spec_norm(weights_only) ** (1/d)
  elif measure_type == ComplexityType.FRO_DIST:
    return _fro_dist(weights_only, init_weights_only)
  raise KeyError