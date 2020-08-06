from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
import hashlib
from pathlib import Path
import time
from typing import Dict, List, NamedTuple, Optional


class DatasetType(Enum):
  MNIST = (1, (1, 28, 28), 10)
  CIFAR10 = (2, (3, 32, 32), 10)
  CIFAR100 = (3, (3, 32, 32), 100)
  SVHN = (4, (3, 32, 32), 10)
  
  def __init__(self, id, image_shape, num_classes):
    self.D = image_shape
    self.K = num_classes

class DatasetSubsetType(IntEnum):
  TRAIN = 0
  TEST = 1

class ModelType(Enum):
  DEEP = 1
  NIN = 2

class ComplexityType(Enum):
  NONE = 1
  L2 = 2
  LOG_PROD_OF_FRO = 3
  LOG_SUM_OF_FRO = 4
  PARAM_NORM = 5
  PATH_NORM = 6
  PARAMS = 7
  LOG_PROD_OF_SPEC = 8
  LOG_SUM_OF_SPEC = 9
  FRO_DIST = 10
  PACBAYES_ORIG = 11
  PACBAYES_INIT = 12
  PACBAYES_MAG_ORIG = 13
  PACBAYES_MAG_INIT = 14
  PACBAYES_FLATNESS = 15
  PACBAYES_MAG_FLATNESS = 16
  FRO_OVER_SPEC = 17
  DIST_SPEC_INIT = 18
  INVERSE_MARGIN = 19
  LOG_PROD_OF_FRO_OVER_MARGIN = 20
  LOG_SUM_OF_FRO_OVER_MARGIN = 21
  LOG_PROD_OF_SPEC_OVER_MARGIN = 22
  LOG_SUM_OF_SPEC_OVER_MARGIN = 23
  PATH_NORM_OVER_MARGIN = 24
  L2_DIST = 25

  @classmethod
  def data_dependent_measures(cls):
    return {
      cls.PACBAYES_ORIG,
      cls.PACBAYES_INIT,
      cls.PACBAYES_MAG_ORIG,
      cls.PACBAYES_MAG_INIT,
      cls.PACBAYES_FLATNESS,
      cls.PACBAYES_MAG_FLATNESS,
      cls.INVERSE_MARGIN,
      cls.LOG_PROD_OF_FRO_OVER_MARGIN,
      cls.LOG_SUM_OF_FRO_OVER_MARGIN,
      cls.LOG_PROD_OF_SPEC_OVER_MARGIN,
      cls.LOG_SUM_OF_SPEC_OVER_MARGIN,
      cls.PATH_NORM_OVER_MARGIN,
    }

  @classmethod
  def acc_dependent_measures(cls):
    return {
      cls.PACBAYES_ORIG,
      cls.PACBAYES_INIT,
      cls.PACBAYES_MAG_ORIG,
      cls.PACBAYES_MAG_INIT,
      cls.PACBAYES_FLATNESS,
      cls.PACBAYES_MAG_FLATNESS,
    }

class OptimizerType(Enum):
  SGD = 1
  SGD_MOMENTUM = 2
  ADAM = 3

class Verbosity(IntEnum):
  NONE = 1
  RUN = 2
  EPOCH = 3
  BATCH = 4

@dataclass(frozen=False)
class State:
  epoch: int = 1
  batch: int = 1
  global_batch: int = 1
  converged: bool = False
  ce_check_freq: int = 0
  ce_check_milestones: Optional[List[float]] = None

# Hyperparameters that uniquely determine the experiment
@dataclass(frozen=True)
class HParams:
  seed: int = 0
  use_cuda: bool = True
  # Model
  model_type: ModelType = ModelType.NIN
  model_depth: int = 2
  model_width: int = 8
  base_width: int = 25
  # Dataset
  dataset_type: DatasetType = DatasetType.CIFAR10
  data_seed: Optional[int] = 42
  train_dataset_size: Optional[int] = None
  test_dataset_size: Optional[int] = None
  label_noise: Optional[float] = None
  # Training
  batch_size: int = 32
  epochs: int = 300
  optimizer_type: OptimizerType = OptimizerType.SGD_MOMENTUM
  lr: float = 0.01
  # Cross-entropy stopping criterion
  ce_target: Optional[float] = 0.01
  ce_target_milestones: Optional[List[float]] = field(default_factory=lambda: [0.05, 0.025, 0.015])

  def to_tensorboard_dict(self) -> dict:
    d = asdict(self)
    d = {x: y for (x,y) in d.items() if y is not None}
    d = {x:(y.name if isinstance(y, Enum) else y) for x,y in d.items()}

    return d
  
  @property
  def md5(self):
    return hashlib.md5(str(self).encode('utf-8')).hexdigest()
  
  @property
  def wandb_md5(self):
    dictionary = self.to_tensorboard_dict()
    dictionary['seed'] = 0
    return hashlib.md5(str(dictionary).encode('utf-8')).hexdigest()

# Configuration which doesn't affect experiment results
@dataclass(frozen=True)
class Config:
  id: int = field(default_factory=lambda: time.time_ns())
  log_batch_freq: Optional[int] = None
  log_epoch_freq: Optional[int] = 10
  save_epoch_freq: Optional[int] = 1
  root_dir: Path = Path('.')
  data_dir: Path = Path('data')
  verbosity: Verbosity = Verbosity.EPOCH
  use_tqdm: bool = False

  # Validation
  def __post_init__(self):
    # Set up directories
    for directory in ('results', 'checkpoints'):
      (self.root_dir / directory).mkdir(parents=True, exist_ok=True)
    self.data_dir.mkdir(parents=True, exist_ok=True)

  @property
  def checkpoint_dir(self):
    return self.root_dir / 'checkpoints'
  
  @property
  def results_dir(self):
    return self.root_dir / 'results'

class EvaluationMetrics(NamedTuple):
  acc: float
  avg_loss: float
  num_correct: int
  num_to_evaluate_on: int
  all_complexities: Dict[ComplexityType, float]
