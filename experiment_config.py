from numpy import infty

from dataclasses import asdict, dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Deque, Dict, NamedTuple, Optional, List
from collections import deque

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
  RESNET = 3

class ComplexityType(Enum):
  NONE = 1
  L2 = 2
  PROD_OF_FRO = 3
  SUM_OF_FRO = 4
  PARAM_NORM = 5
  PATH_NORM = 6
  PARAMS = 7
  PROD_OF_SPEC = 8
  SUM_OF_SPEC = 9
  FRO_DIST = 10

class OptimizerType(Enum):
  SGD = 1
  SGD_MOMENTUM = 2
  ADAM = 3

class Verbosity(IntEnum):
  NONE = 1
  RUN = 2
  EPOCH = 3
  LAGRANGIAN = 4
  BATCH = 5

class LagrangianType(Enum):
  NONE = 1
  PENALTY = 2
  AUGMENTED = 3

@dataclass(frozen=False)
class ETrainingState:
  id: int
  epoch: int = 1
  batch: int = 1
  global_batch: int = 1
  lagrangian_mu: Optional[float] = None
  lagrangian_lambda: Optional[float] = None
  prev_loss: Optional[float] = None
  prev_acc: Optional[float] = None
  prev_constraint: Optional[float] = None
  loss_hist: Deque[float] = deque([])
  constraint_hist: Deque[float] = deque([])
  constraint_to_beat = infty
  converged: bool = False
  convergence_test_hist: Deque[bool] = deque([])

# Configuration for the experiment
@dataclass(frozen=True)
class EConfig:
  seed: int
  data_seed: Optional[int]
  use_cuda: bool
  # Model
  model_type: ModelType
  model_shape: List[int]
  # Dataset
  dataset_type: DatasetType
  # Training
  batch_size: int
  epochs: int
  optimizer_type: OptimizerType = OptimizerType.SGD
  lr: float = 0.001
  complexity_type: ComplexityType = ComplexityType.NONE
  complexity_lambda: Optional[float] = None
  # Constrained Optimization
  lagrangian_type: LagrangianType = LagrangianType.NONE
  lagrangian_start_epoch: Optional[int] = None
  lagrangian_target: Optional[float] = None
  lagrangian_tolerance: Optional[float] = None
  lagrangian_start_mu: Optional[float] = None
  lagrangian_patience_batches: Optional[int] = None
  lagrangian_improvement_rate: Optional[float] = None
  ## Augmented Lagrangian Terms
  lagrangian_start_lambda: Optional[float] = None
  lagrangian_convergence_tolerance: Optional[float] = None
  # Global Convergence
  global_convergence_tolerance: Optional[float] = None
  global_convergence_patience: Optional[int] = None
  # Visibility (default no visibility)
  log_batch_freq: Optional[int] = 100
  log_epoch_freq: Optional[int] = 20
  save_epoch_freq: Optional[int] = None
  data_dir: Path = Path('data')
  log_dir: Path = Path('logs')
  checkpoint_dir: Path = Path('checkpoints')
  resume_from_checkpoint: bool = False
  verbosity: Verbosity = Verbosity.NONE

  # Validation
  def __post_init__(self):
    if self.lagrangian_type != LagrangianType.NONE:
      if self.lagrangian_start_epoch is None:
        raise KeyError
      if self.lagrangian_target is None:
        raise KeyError
      if self.lagrangian_tolerance is None:
        raise KeyError
      if self.lagrangian_start_mu is None:
        raise KeyError
      if self.lagrangian_patience_batches is None:
        raise KeyError
      if self.lagrangian_improvement_rate is None:
        raise KeyError
    if self.lagrangian_type == LagrangianType.AUGMENTED:
      if self.lagrangian_start_lambda is None:
        raise KeyError
      if self.lagrangian_convergence_tolerance is None:
        raise KeyError

  def to_tensorboard_dict(self) -> dict:
    d = asdict(self)
    d["model_type"] = d["model_type"].name
    d["dataset_type"] = d["dataset_type"].name
    d["optimizer_type"] = d["optimizer_type"].name
    d["complexity_type"] = d["complexity_type"].name
    d["lagrangian_type"] = d["lagrangian_type"].name
    if d["complexity_lambda"] is None:
      del d["complexity_lambda"]
    if d["lagrangian_start_epoch"] is None:
      del d["lagrangian_start_epoch"]
    if d["lagrangian_target"] is None:
      del d["lagrangian_target"]
    if d["lagrangian_tolerance"] is None:
      del d["lagrangian_tolerance"]
    if d["lagrangian_start_mu"] is None:
      del d["lagrangian_start_mu"]
    if d["lagrangian_patience_batches"] is None:
      del d["lagrangian_patience_batches"]
    if d["lagrangian_improvement_rate"] is None:
      del d["lagrangian_improvement_rate"]
    if d["lagrangian_start_lambda"] is None:
      del d["lagrangian_start_lambda"]
    if d["lagrangian_convergence_tolerance"] is None:
      del d["lagrangian_convergence_tolerance"]
    del d["log_batch_freq"]
    del d["save_epoch_freq"]
    del d["data_dir"]
    del d["log_dir"]
    del d["checkpoint_dir"]
    del d["resume_from_checkpoint"]
    del d["verbosity"]

    return d

class EvaluationMetrics(NamedTuple):
  acc: float
  avg_loss: float
  complexity: float
  complexity_loss: float
  num_correct: int
  num_to_evaluate_on: int
  all_complexities: Dict[ComplexityType, float]
