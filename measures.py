from contextlib import contextmanager
from copy import deepcopy
from typing import List

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from experiment_config import ComplexityType
from models import ExperimentBaseModel

# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
def _path_norm(model: ExperimentBaseModel, device: torch.device) -> torch.Tensor:
  model = deepcopy(model)
  model.eval()
  for param in model.parameters():
    if param.requires_grad:
      param.pow_(2)
  x = torch.ones([1] + model.dataset_properties.D, device=device)
  x = model(x)
  del model
  return x.sum().sqrt()

def _prod_of_fro(weights: List[torch.nn.Parameter]) -> torch.Tensor:
  weight_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in weights])
  return weight_norms.log().sum().exp()

def _param_norm(weights: List[torch.nn.Parameter]) -> torch.Tensor:
  weight_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in weights])
  return weight_norms.sum()

def _spec_norm(weights: List[torch.nn.Parameter]) -> torch.Tensor:
  weight_norms = torch.cat([p.svd()[1].max().unsqueeze(0) for p in weights])
  return weight_norms.log().sum().exp()

def _fft_spec_norm(weights: List[torch.nn.Parameter]) -> torch.Tensor:
  weight_norms = torch.cat([p.svd()[1].max().unsqueeze(0) for p in weights])
  return weight_norms.log().sum().exp()

def _fro_dist(weights: List[torch.nn.Parameter], init_weights: List[torch.nn.Parameter]) -> torch.Tensor:
  weight_norms = torch.cat([
    (p - q).norm('fro').unsqueeze(0) ** 2
    for p, q in zip(weights, init_weights)])
  return weight_norms.sum()

def _pacbayes_orig(model: ExperimentBaseModel, dataloader: DataLoader, device: torch.device, acc: float) -> torch.Tensor:
  weights_only = [p for name, p in model.named_parameters() if 'bias' not in name and '.bn' not in name]
  flat_params = torch.cat([p.view(-1) for p in weights_only], dim=0)
  norm = flat_params.norm(p=2) ** 2
  sigma = _pacbayes_sigma(model, dataloader, device, acc)
  m = len(dataloader.dataset)
  return norm / (4 * sigma ** 2) + np.log(m / sigma) + 10

def _pacbayes_flatness(model: ExperimentBaseModel, dataloader: DataLoader, device: torch.device, acc: float) -> torch.Tensor:
  sigma = _pacbayes_sigma(model, dataloader, device, acc)
  return torch.tensor(1 / sigma ** 2)

@contextmanager
def _perturbed_model(model: ExperimentBaseModel, sigma: float, device: torch.device):
  noise = [torch.normal(0,sigma,p.shape).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model

# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(model: ExperimentBaseModel, dataloader: DataLoader, device: torch.device, accuracy: float) -> float:
  SEARCH_DEPTH = 10
  MONTECARLO_SAMPLES = 3
  ACCURACY_DISPLACEMENT = 0.1
  DISPLACEMENT_TOLERANCE = 1e-2

  lower, upper = 0, 2
  sigma = 1

  for _ in range(SEARCH_DEPTH):
    sigma = (lower + upper) / 2
    accuracy_samples = []
    for _ in range(MONTECARLO_SAMPLES):
      with _perturbed_model(model, sigma, device) as p_model:
        loss_estimate = 0
        for data, target in dataloader:
          logits = p_model(data)
          pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
          batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
          loss_estimate += batch_correct.sum()
        loss_estimate /= len(dataloader.dataset)
        accuracy_samples.append(loss_estimate)
    displacement = abs(np.mean(accuracy_samples) - accuracy)
    if abs(displacement - ACCURACY_DISPLACEMENT) < DISPLACEMENT_TOLERANCE:
      break
    elif displacement > ACCURACY_DISPLACEMENT:
      # Too much perturbation
      upper = sigma
    else:
      # Not perturbed enough to reach target displacement
      lower = sigma
  return sigma

def get_measure(model: ExperimentBaseModel, init_model: ExperimentBaseModel, measure_type: ComplexityType, device: torch.device, dataloader: DataLoader = None, acc: float = None) -> torch.Tensor:
  weights_only = [p for name, p in model.named_parameters() if 'bias' not in name and '.bn' not in name]
  init_weights_only = [p for name, p in init_model.named_parameters() if 'bias' not in name and '.bn' not in name]
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
  elif measure_type == ComplexityType.PACBAYES_BOUND:
    return _pacbayes_orig(model, dataloader, device, acc)
  elif measure_type == ComplexityType.PACBAYES_SHARPNESS:
    return _pacbayes_flatness(model, dataloader, device, acc)
  raise KeyError
