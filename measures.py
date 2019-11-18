import torch

from experiment_config import ComplexityType
from models import ExperimentBaseModel

def calculate_complexity(model: ExperimentBaseModel, complexity_type: ComplexityType) -> torch.Tensor:
  if complexity_type == ComplexityType.NONE:
    return torch.zeros((1,))
  elif complexity_type == ComplexityType.L2:
    return model.get_norm(2)
  raise KeyError
