import torch

from experiment_config import ComplexityType
from models import ExperimentBaseModel

def calculate_complexity(model: ExperimentBaseModel, complexity_type: ComplexityType) -> float:
  if complexity_type == ComplexityType.L2:
    return _l2_complexity(model)
  raise KeyError

def _l2_complexity(model: ExperimentBaseModel) -> float:
  return sum([torch.norm(w, p='fro').numpy() for w in model.parameters()])