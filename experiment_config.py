from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional
from pathlib import Path

class DatasetType(Enum):
  MNIST = 1
  CIFAR10 = 2
  CIFAR100 = 3
  REGRESSION = 4

class DatasetSubsetType(Enum):
  TRAIN = 1
  VAL = 2
  TEST = 3

class ModelType(Enum):
  DEEP = 1

class ComplexityType(Enum):
  L2 = 1

class OptimizerType(Enum):
  SGD = 1
  ADAM = 2

class ObjectiveType(Enum):
  MSE = 1
  CE = 2

@dataclass(frozen=False)
class ETrainingState:
  id: int
  epoch: int = 1

# Configuration for the experiment
@dataclass(frozen=True)
class EConfig:
  seed: int
  cuda: bool
  # Model
  model_type: ModelType
  # Dataset
  dataset_type: DatasetType
  # Training
  batch_size: int
  epochs: int
  optimizer_type: OptimizerType = OptimizerType.SGD
  lr: float = 0.001
  # Visibility (default no visibility)
  log_batch_freq: Optional[int] = 100
  save_epoch_freq: Optional[int] = 2
  log_tensorboard: bool = False
  data_dir: Path = Path('data')
  checkpoint_dir: Path = Path('checkpoints')
  resume_from_checkpoint: bool = False

  def to_tensorboard_dict(self):
    d = asdict(self)
    d["model_type"] = d["model_type"].name
    d["dataset_type"] = d["dataset_type"].name
    d["optimizer_type"] = d["optimizer_type"].name
    d["objective"] = d["objective"].name
    if d["log_batch_freq"] is None:
      d["log_batch_freq"] = "None"
    if d["save_epoch_freq"] is None:
      d["save_epoch_freq"] = "None"
    return d