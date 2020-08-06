#!/usr/bin/env python3
from pathlib import Path
import pickle
import time
from typing import Optional, List

import torch

from source.experiment import Experiment
from source.experiment_config import DatasetType, HParams, State, ModelType, OptimizerType, Verbosity
from source.logs import BaseLogger


# A class that enables dot access for a dictionary.
class Bunch:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def to_dict(self):
        return self.__dict__

    def fancy_print(self, prefix=''):
        str_list = []
        for key, val in self.__dict__.items():
            str_list.append(prefix + "%s = %s" % (key, val))
        return '\n'.join(str_list)


hp: HParams = Bunch(**HParams.__dict__)

# Launch a training run
def train(
  logger: BaseLogger,
  seed: int = hp.seed,
  use_cuda: bool = hp.use_cuda,
  # Model
  model_type: ModelType = hp.model_type,
  model_depth: int = hp.model_depth,
  model_width: int = hp.model_width,
  base_width: int = hp.base_width,
  # Dataset
  dataset_type: DatasetType = hp.dataset_type,
  data_seed: Optional[int] = hp.data_seed,
  train_dataset_size: Optional[int] = hp.train_dataset_size,
  test_dataset_size: Optional[int] = hp.test_dataset_size,
  label_noise: Optional[float] = hp.label_noise,
  # Training
  batch_size: int = hp.batch_size,
  epochs: int = hp.epochs,
  optimizer_type: OptimizerType = hp.optimizer_type,
  lr: float = hp.lr,
  # Cross-entropy stopping criterion
  ce_target: Optional[float] = hp.ce_target,
  ce_target_milestones: Optional[List[float]] = [0.05, 0.025, 0.015],
  # Visibility (default no visibility)
  log_batch_freq: Optional[int] = hp.log_batch_freq,
  log_epoch_freq: Optional[int] = hp.log_epoch_freq,
  save_epoch_freq: Optional[int] = hp.save_epoch_freq,
  root_dir: Path = hp.root_dir,
  data_dir: Path = hp.data_dir,
  verbosity: Verbosity = hp.verbosity,
  use_tqdm: bool = hp.use_tqdm,
) -> None:
  experiment_id = time.time_ns()
  print('[Experiment {}]'.format(experiment_id))
  print("[Experiment {}] CUDA devices:".format(experiment_id), torch.cuda.device_count())

  root_path = Path(root_dir)
  if data_dir is None:
    data_path = root_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
  else:
    data_path = Path(data_dir)

  hparams = HParams(
    seed=seed,
    use_cuda=use_cuda,
    # Model
    model_type=model_type,
    model_depth=model_depth,
    model_width=model_width,
    base_width=base_width,
    # Dataset
    dataset_type=dataset_type,
    data_seed=data_seed,
    train_dataset_size=train_dataset_size,
    test_dataset_size=test_dataset_size,
    label_noise=label_noise,
    # Training
    batch_size=batch_size,
    epochs=epochs,
    optimizer_type=optimizer_type,
    lr=lr,
    # Cross-entropy stopping criterion
    ce_target=ce_target,
    ce_target_milestones=ce_target_milestones,
    # Visibility (default no visibility)
    log_batch_freq=log_batch_freq,
    log_epoch_freq=log_epoch_freq,
    save_epoch_freq=save_epoch_freq,
    root_dir=root_dir,
    data_dir=data_dir,
    verbosity=verbosity,
    use_tqdm=use_tqdm,
  )
  state = State(
    id=experiment_id,
    ce_check_milestones=ce_target_milestones,
  )

  def dump_results(epoch, val_eval, train_eval):
    results = {
      'e_state': state,
      'e_config': hparams,
      'final_results_val': val_eval,
      'final_results_train': train_eval,
    }
    with open(hparams.results_dir / '{}.epoch{}.pkl'.format(experiment_id, epoch), mode='wb') as results_file:
      pickle.dump(results, results_file)

  print('[Experiment {}]'.format(experiment_id), hparams)
  device = torch.device('cuda' if use_cuda else 'cpu')
  Experiment(state, device, hparams, logger, dump_results).train()
