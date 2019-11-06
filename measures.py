import torch

from experiment_config import ComplexityType
from models import ExperimentBaseModel

def calculate_complexity(model: ExperimentBaseModel, complexity_type: ComplexityType) -> torch.FloatTensor:
  if complexity_type == ComplexityType.L2:
    return _l2_complexity(model)
  raise KeyError

def _l2_complexity(model: ExperimentBaseModel) -> torch.FloatTensor:
  return sum([torch.norm(w, p='fro') for w in model.parameters()])
