from contextlib import contextmanager
from copy import deepcopy
from typing import List, Optional

import numpy as np
import torch
from torch.nn.parameter import Parameter
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

def _spec_dist(weights: List[torch.nn.Parameter], init_weights: List[torch.nn.Parameter]) -> torch.Tensor:
  weight_norms = torch.cat([
    (p - q).svd()[1].max().unsqueeze(0) ** 2
    for p, q in zip(weights, init_weights)])
  return weight_norms.sum()

def _fro_over_spec(weights: List[torch.nn.Parameter], init_weights: List[torch.nn.Parameter]) -> torch.Tensor:
  fro_weight_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in weights])
  spec_weight_norms = torch.cat([p.svd()[1].max().unsqueeze(0) for p in weights])
  norms = fro_weight_norms / spec_weight_norms
  return norms.sum()

def _pacbayes_bound(
  model: ExperimentBaseModel,
  dataloader: DataLoader,
  device: torch.device,
  acc: float,
  init_weights: Optional[List[Parameter]] = None
) -> torch.Tensor:
  sigma = _pacbayes_sigma(model, dataloader, device, acc, None)
  weights_only = [p for name, p in model.named_parameters() if 'bias' not in name and '.bn' not in name]
  flat_params = torch.cat([p.view(-1) for p in weights_only], dim=0)
  if init_weights is not None:
    flat_params2 = torch.cat([p.view(-1) for p in init_weights], dim=0)
    flat_params = flat_params - flat_params2
  norm = flat_params.norm(p=2) ** 2
  m = len(dataloader.dataset)
  return norm / (4 * sigma ** 2) + np.log(m / sigma) + 10

def _pacbayes_mag_bound(
  model: ExperimentBaseModel,
  dataloader: DataLoader,
  device: torch.device,
  acc: float,
  init_weights: Optional[List[Parameter]],
  orig: bool
) -> torch.Tensor:
  eps = 1e-3
  sigma = _pacbayes_sigma(model, dataloader, device, acc, eps)
  omega = sum(p.numel() for p in model.parameters() if p.requires_grad)
  weights_only = [p for name, p in model.named_parameters() if 'bias' not in name and '.bn' not in name]
  flat_params = torch.cat([p.view(-1) for p in weights_only], dim=0)
  flat_params2 = torch.cat([p.view(-1) for p in init_weights], dim=0)
  if orig:
    norm = flat_params.norm(p=2) ** 2
  else:
    norm = (flat_params - flat_params2).norm(p=2) ** 2
  m = len(dataloader.dataset)
  numerator = eps ** 2 + (sigma ** 2 + 1) * norm / omega
  denominator = eps ** 2 + sigma ** 2 * (flat_params - flat_params2) ** 2
  return 1/4 * (numerator / denominator).log().sum() + np.log(m / sigma) + 10

def _pacbayes_flatness(model: ExperimentBaseModel, dataloader: DataLoader, device: torch.device, acc: float, magnitude_eps: Optional[float]) -> torch.Tensor:
  sigma = _pacbayes_sigma(model, dataloader, device, acc, magnitude_eps)
  return torch.tensor(1 / sigma ** 2)

@contextmanager
def _perturbed_model(model: ExperimentBaseModel, sigma: float, device: torch.device, magnitude_eps: Optional[float] = None):
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2).to(device) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model

# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
def _pacbayes_sigma(
  model: ExperimentBaseModel,
  dataloader: DataLoader,
  device: torch.device,
  accuracy: float,
  magnitude_eps: Optional[float]
) -> float:
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
      with _perturbed_model(model, sigma, device, magnitude_eps) as p_model:
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

def _margin(model: ExperimentBaseModel, dataloader: DataLoader) -> torch.Tensor:
  margins = []
  m = len(dataloader.dataset)
  for data, target in dataloader:
    logits = model(data)
    correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
    logits[torch.arange(logits.shape[0]), target] = float('-inf')
    max_other_logit = logits.data.max(1).values  # get the index of the max logits
    margin = correct_logit - max_other_logit
    margins.append(margin)
  return torch.cat(margins).kthvalue(m // 10)[0]

def get_measure(
  model: ExperimentBaseModel,
  init_model: ExperimentBaseModel,
  measure_type: ComplexityType,
  device: torch.device,
  dataloader: DataLoader = None,
  acc: float = None
) -> torch.Tensor:
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
  elif measure_type == ComplexityType.PACBAYES_ORIG:
    if dataloader is None or acc is None:
      raise RuntimeError
    return _pacbayes_bound(model, dataloader, device, acc)
  elif measure_type == ComplexityType.PACBAYES_INIT:
    if dataloader is None or acc is None:
      raise RuntimeError
    return _pacbayes_bound(model, dataloader, device, acc, init_weights_only)
  elif measure_type == ComplexityType.PACBAYES_MAG_ORIG:
    if dataloader is None or acc is None:
      raise RuntimeError
    return _pacbayes_mag_bound(model, dataloader, device, acc, init_weights_only, True)
  elif measure_type == ComplexityType.PACBAYES_MAG_INIT:
    if dataloader is None or acc is None:
      raise RuntimeError
    return _pacbayes_mag_bound(model, dataloader, device, acc, init_weights_only, False)
  elif measure_type == ComplexityType.PACBAYES_FLATNESS:
    if dataloader is None or acc is None:
      raise RuntimeError
    return _pacbayes_flatness(model, dataloader, device, acc, None)
  elif measure_type == ComplexityType.PACBAYES_MAG_FLATNESS:
    if dataloader is None or acc is None:
      raise RuntimeError
    return _pacbayes_flatness(model, dataloader, device, acc, 1e-3)
  elif measure_type == ComplexityType.FRO_OVER_SPEC:
    return _fro_over_spec(weights_only, init_weights_only)
  elif measure_type == ComplexityType.DIST_SPEC_INIT:
    return _spec_dist(weights_only, init_weights_only)
  elif measure_type == ComplexityType.INVERSE_MARGIN:
    if dataloader is None:
      raise RuntimeError
    return torch.tensor(1).to(device) / _margin(model, dataloader) ** 2
  elif measure_type == ComplexityType.PROD_OF_FRO_OVER_MARGIN:
    if dataloader is None:
      raise RuntimeError
    return _prod_of_fro(weights_only) / _margin(model, dataloader) ** 2
  elif measure_type == ComplexityType.SUM_OF_FRO_OVER_MARGIN:
    if dataloader is None:
      raise RuntimeError
    return d * (_prod_of_fro(weights_only) / _margin(model, dataloader) ** 2) ** (1/d)
  elif measure_type == ComplexityType.PROD_OF_SPEC_OVER_MARGIN:
    if dataloader is None:
      raise RuntimeError
    return _spec_norm(weights_only) / _margin(model, dataloader) ** 2
  elif measure_type == ComplexityType.SUM_OF_SPEC_OVER_MARGIN:
    if dataloader is None:
      raise RuntimeError
    return d * (_spec_norm(weights_only) / _margin(model, dataloader) ** 2) ** (1/d)
  elif measure_type == ComplexityType.PATH_NORM_OVER_MARGIN:
    if dataloader is None:
      raise RuntimeError
    return _path_norm(model, device) / _margin(model, dataloader) ** 2
  raise KeyError
