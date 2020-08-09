from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from .experiment_config import ComplexityType as CT
from .models import ExperimentBaseModel


def get_weights_only(model: ExperimentBaseModel) -> List[Tensor]:
  return [p.view(p.shape[0],-1) for name, p in model.named_parameters() if 'bias' not in name and 'downsample.1' not in name]

# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
@torch.no_grad()
def reparam(model, prev_layer=None):
  for child in model.children():
    prev_layer = reparam(child, prev_layer)
    if child._get_name() == 'Conv2d':
      prev_layer = child
    elif child._get_name() == 'BatchNorm2d':
      scale = child.weight / ((child.running_var + child.eps).sqrt())
      prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
      perm = list(reversed(range(prev_layer.weight.dim())))
      prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
      child.bias.fill_(0)
      child.weight.fill_(1)
      child.running_mean.fill_(0)
      child.running_var.fill_(1)
  return prev_layer

def get_flat_params(weights_only: List[Tensor]) -> Tensor:
  return torch.cat([p.view(-1) for p in weights_only], dim=0)

def get_parameter_norms(
  weights_only: List[Tensor],
  norm_type: str
) -> Tensor:
  if norm_type == 'fro':
    return torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in weights_only])
  elif norm_type == 'spec':
    return torch.cat([p.svd()[1].max().unsqueeze(0) for p in weights_only])
  else:
    raise KeyError

@contextmanager
def _perturbed_model(
  model: ExperimentBaseModel,
  sigma: float,
  magnitude_eps: Optional[float] = None
):
  device = next(model.parameters()).device
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
  accuracy: float,
  magnitude_eps: Optional[float],
  search_depth: int = 10,
  montecarlo_samples: int = 3,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
) -> float:
  lower, upper = 0, 2
  sigma = 1

  for _ in range(search_depth):
    sigma = (lower + upper) / 2
    accuracy_samples = []
    for _ in range(montecarlo_samples):
      with _perturbed_model(model, sigma, magnitude_eps) as p_model:
        loss_estimate = 0
        for data, target in dataloader:
          logits = p_model(data)
          pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
          batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
          loss_estimate += batch_correct.sum()
        loss_estimate /= len(dataloader.dataset)
        accuracy_samples.append(loss_estimate)
    displacement = abs(np.mean(accuracy_samples) - accuracy)
    if abs(displacement - accuracy_displacement) < displacement_tolerance:
      break
    elif displacement > accuracy_displacement:
      # Too much perturbation
      upper = sigma
    else:
      # Not perturbed enough to reach target displacement
      lower = sigma
  return sigma

def _margin(
  model: ExperimentBaseModel,
  dataloader: DataLoader
) -> Tensor:
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

# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
def _path_norm(model: ExperimentBaseModel) -> Tensor:
  device = next(model.parameters()).device
  model = deepcopy(model)
  model.eval()
  for param in model.parameters():
    if param.requires_grad:
      param.data.pow_(2)
  x = torch.ones([1] + list(model.dataset_type.D), device=device)
  x = model(x)
  del model
  return x.sum().sqrt()

def _param_count(model: ExperimentBaseModel) -> int:
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _l2_norm(weights_only: List[Tensor]) -> Tensor:
  return get_flat_params(weights_only).norm(p=2)

def _l2_dist(weights_only: List[Tensor], init_weights_only: List[Tensor]) -> Tensor:
  return get_flat_params([(p - q) for p, q in zip(weights_only, init_weights_only)]).norm(p=2)

def _log_prod_of_fro(weights_only: List[Tensor]) -> Tensor:
  return get_parameter_norms(weights_only, 'fro').log().sum()

def _param_norm(weights_only: List[Tensor]) -> Tensor:
  return get_parameter_norms(weights_only, 'fro').sum()

def _log_spec_norm(weights_only: List[Tensor]) -> Tensor:
  return get_parameter_norms(weights_only, 'spec').log().sum()

def _fro_dist(weights_only: List[Tensor], init_weights_only: List[Tensor]) -> Tensor:
  return get_parameter_norms([(p - q) for p, q in zip(weights_only, init_weights_only)], 'fro').sum()

def _spec_dist(weights_only: List[Tensor], init_weights_only: List[Tensor]) -> Tensor:
  return get_parameter_norms([(p - q) for p, q in zip(weights_only, init_weights_only)], 'spec').sum()

def _fro_over_spec(weights_only: List[Tensor], init_weights_only: List[Tensor]) -> Tensor:
  fro_weight_norms = get_parameter_norms(weights_only, 'fro')
  spec_weight_norms = get_parameter_norms(weights_only, 'spec')
  return (fro_weight_norms / spec_weight_norms).sum()

def _pacbayes_bound(
  weight_norm: Tensor,
  sigma: float,
  m: int,
) -> Tensor:
  return (weight_norm ** 2) / (4 * sigma ** 2) + np.log(m / sigma) + 10

def _pacbayes_mag_bound(
  sigma: float,
  eps: float,
  omega: int,
  m: int,
  weights_only: List[Tensor],
  init_weights_only: List[Tensor],
  orig: bool
) -> Tensor:
  flat_params = get_flat_params(weights_only)
  init_flat_params = get_flat_params(init_weights_only)
  if orig:
    weight_norm = _l2_norm(weights_only) ** 2
  else:
    weight_norm = _l2_dist(weights_only, init_weights_only) ** 2
  numerator = eps ** 2 + (sigma ** 2 + 1) * weight_norm / omega
  denominator = eps ** 2 + sigma ** 2 * (flat_params - init_flat_params) ** 2
  return 1/4 * (numerator / denominator).log().sum() + np.log(m / sigma) + 10

# "PUBLIC" METHODS
@torch.no_grad()
def get_all_measures(
  model: ExperimentBaseModel,
  init_model: ExperimentBaseModel,
  dataloader: DataLoader,
  acc: float,
  *,
  use_reparam: bool = True,
) -> Dict[CT, float]:
  measures = {}

  model = deepcopy(model)
  init_model = deepcopy(init_model)
  if use_reparam:
    reparam(model)
    reparam(init_model)

  device = next(model.parameters()).device
  weights_only = get_weights_only(model)
  init_weights_only = get_weights_only(init_model)
  
  d = len(weights_only)

  measures[CT.L2] = _l2_norm(weights_only)
  measures[CT.L2_DIST] = _l2_dist(weights_only, init_weights_only)
  measures[CT.LOG_PROD_OF_FRO] = _log_prod_of_fro(weights_only)
  measures[CT.LOG_SUM_OF_FRO] = np.log(d) + (1/d) * measures[CT.LOG_PROD_OF_FRO]
  measures[CT.PARAM_NORM] = _param_norm(weights_only)
  measures[CT.PATH_NORM] = _path_norm(model)
  measures[CT.PARAMS] = torch.tensor(_param_count(model))
  measures[CT.LOG_PROD_OF_SPEC] = _log_spec_norm(weights_only)
  measures[CT.LOG_SUM_OF_SPEC] = np.log(d) + (1/d) * measures[CT.LOG_PROD_OF_SPEC]
  measures[CT.FRO_DIST] = _fro_dist(weights_only, init_weights_only)
  measures[CT.FRO_OVER_SPEC] = _fro_over_spec(weights_only, init_weights_only)
  measures[CT.DIST_SPEC_INIT] = _spec_dist(weights_only, init_weights_only)

  # Sharpness measures
  m = len(dataloader.dataset)
  omega = _param_count(model)

  eps = None
  sigma = _pacbayes_sigma(model, dataloader, acc, eps)
  measures[CT.PACBAYES_ORIG] = _pacbayes_bound(measures[CT.L2], sigma, m)
  measures[CT.PACBAYES_INIT] = _pacbayes_bound(measures[CT.L2_DIST], sigma, m)
  measures[CT.PACBAYES_FLATNESS] = torch.tensor(1 / sigma ** 2)
  
  mag_eps = 1e-3
  mag_sigma = _pacbayes_sigma(model, dataloader, acc, mag_eps)
  measures[CT.PACBAYES_MAG_ORIG] = _pacbayes_mag_bound(mag_sigma, mag_eps, omega, m, weights_only, init_weights_only, True)
  measures[CT.PACBAYES_MAG_INIT] = _pacbayes_mag_bound(mag_sigma, mag_eps, omega, m, weights_only, init_weights_only, False)
  measures[CT.PACBAYES_MAG_FLATNESS] = torch.tensor(1 / mag_sigma ** 2)
  
  # Margin measures
  margin = _margin(model, dataloader).abs()
  measures[CT.INVERSE_MARGIN] = torch.tensor(1).to(device) / margin ** 2
  measures[CT.LOG_PROD_OF_FRO_OVER_MARGIN] = measures[CT.LOG_PROD_OF_FRO] -  2 * margin.log()
  measures[CT.LOG_SUM_OF_FRO_OVER_MARGIN] = np.log(d) + (1/d) * (measures[CT.LOG_PROD_OF_FRO] -  2 * margin.log())
  measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] = measures[CT.LOG_PROD_OF_SPEC] - 2 * margin.log()
  measures[CT.LOG_SUM_OF_SPEC_OVER_MARGIN] = np.log(d) + (1/d) * (measures[CT.LOG_PROD_OF_SPEC] -  2 * margin.log())
  measures[CT.PATH_NORM_OVER_MARGIN] = measures[CT.PATH_NORM] / margin ** 2

  measures = {k: v.item() for k, v in measures.items()}
  return measures
