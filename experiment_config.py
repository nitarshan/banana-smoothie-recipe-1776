from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Optional

class DatasetType(Enum):
  MNIST = 1
  CIFAR10 = 2
  CIFAR100 = 3

class DatasetSubsetType(IntEnum):
  TRAIN = 0
  VAL = 1
  TEST = 2

class ModelType(Enum):
  DEEP = 1
  CONV = 2

class ComplexityType(Enum):
  NONE = 1
  L2 = 2

class OptimizerType(Enum):
  SGD = 1
  ADAM = 2

class Verbosity(IntEnum):
  NONE = 1
  RUN = 2
  EPOCH = 3
  BATCH = 4

@dataclass(frozen=False)
class ETrainingState:
  id: int
  epoch: int = 1

# Configuration for the experiment
@dataclass(frozen=True)
class EConfig:
  seed: int
  use_cuda: bool
  # Model
  model_type: ModelType
  # Dataset
  dataset_type: DatasetType
  # Training
  batch_size: int
  epochs: int
  optimizer_type: OptimizerType = OptimizerType.SGD
  lr: float = 0.001
  complexity_type: ComplexityType = ComplexityType.NONE
  complexity_lambda: float = 0
  # Visibility (default no visibility)
  log_batch_freq: Optional[int] = 100
  save_epoch_freq: Optional[int] = 10
  log_tensorboard: bool = False
  data_dir: Path = Path('data')
  log_dir: Path = Path('logs')
  checkpoint_dir: Path = Path('checkpoints')
  resume_from_checkpoint: bool = False
  verbosity: Verbosity = Verbosity.NONE

  def to_tensorboard_dict(self) -> dict:
    d = asdict(self)
    d["model_type"] = d["model_type"].name
    d["dataset_type"] = d["dataset_type"].name
    d["optimizer_type"] = d["optimizer_type"].name
    d["complexity_type"] = d["complexity_type"].name
    #d["objective"] = d["objective"].name
    del d["log_batch_freq"]
    del d["save_epoch_freq"]
    del d["log_tensorboard"]
    del d["data_dir"]
    del d["log_dir"]
    del d["checkpoint_dir"]
    del d["resume_from_checkpoint"]

    return d
