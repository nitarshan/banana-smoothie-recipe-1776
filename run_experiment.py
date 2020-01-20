#!/usr/bin/env python3
from pathlib import Path
import pickle
import time
from typing import Optional, Tuple, List
from collections import deque

from logs import CometLogger
import fire
import torch

from experiment import Experiment
from experiment_config import (
  ComplexityType, DatasetType, EConfig, ETrainingState, LagrangianType,
  ModelType, OptimizerType, Verbosity)

def setup_paths(root_dir: str, experiment_id: int) -> Tuple[Path, Path, Path, Path]:
  print('[Experiment {}] Setting up directories'.format(experiment_id))
  root_path = Path(root_dir)
  results_path = root_path / 'results'
  results_path.mkdir(parents=True, exist_ok=True)
  log_path = root_path / 'logs'
  data_path = root_path / 'data'
  data_path.mkdir(parents=True, exist_ok=True)
  checkpoint_path = root_path / 'checkpoints'
  checkpoint_path.mkdir(parents=True, exist_ok=True)
  print('[Experiment {}] Results path {}'.format(experiment_id, results_path))
  return results_path, log_path, data_path, checkpoint_path

# Run a single
def single(
  root_dir: str,
  model_type: str,
  model_width: int,
  model_depth: int,
  dataset_type: str,
  optimizer_type: str,
  lr: float,
  epochs: int,
  batch_size: int,
  complexity_type: str,
  complexity_lambda: Optional[float] = None,
  lagrangian_type: str = LagrangianType.NONE.name,
  lagrangian_target: Optional[float] = None,
  lagrangian_start_epoch: Optional[int] = None,
  lagrangian_start_mu: Optional[float] = None,
  lagrangian_tolerance: Optional[float] = None,
  lagrangian_patience_batches: Optional[int] = None,
  lagrangian_improvement_rate: Optional[float] = None,
  lagrangian_start_lambda: Optional[float] = None,
  lagrangian_convergence_tolerance: Optional[float] = None,
  global_convergence_tolerance: Optional[float] = None,
  global_convergence_patience_windows: Optional[int] = None,
  use_cuda: bool = True,
  comet_api_key: Optional[str] = None,
  comet_tag: Optional[str] = None,
  logger: Optional[object] = None,
  save_epoch_freq: Optional[int] = None,
  seed: Optional[int] = None,
  data_seed: Optional[int] = None,
) -> None:
  experiment_id = time.time_ns()
  print('[Experiment {}]'.format(experiment_id))
  print("[Experiment {}] CUDA devices:".format(experiment_id), torch.cuda.device_count())

  results_path, log_path, data_path, checkpoint_path = setup_paths(root_dir, experiment_id)

  if seed is None:
    seed = experiment_id % (2**32)
  e_config = EConfig(
    seed=seed,
    data_seed=data_seed,
    use_cuda=use_cuda,
    model_type= ModelType[model_type],
    model_shape=[model_width]*model_depth,
    dataset_type=DatasetType[dataset_type],
    batch_size=batch_size,
    optimizer_type=OptimizerType[optimizer_type],
    lr=lr,
    epochs=epochs,
    save_epoch_freq=save_epoch_freq,
    complexity_type=ComplexityType[complexity_type],
    complexity_lambda=complexity_lambda,
    lagrangian_type=LagrangianType[lagrangian_type],
    lagrangian_start_epoch=lagrangian_start_epoch,
    lagrangian_target=lagrangian_target,
    lagrangian_tolerance=lagrangian_tolerance,
    lagrangian_start_mu=lagrangian_start_mu,
    lagrangian_patience_batches=lagrangian_patience_batches,
    lagrangian_improvement_rate=lagrangian_improvement_rate,
    lagrangian_start_lambda=lagrangian_start_lambda,
    lagrangian_convergence_tolerance=lagrangian_convergence_tolerance,
    global_convergence_tolerance=global_convergence_tolerance,
    global_convergence_patience_windows=global_convergence_patience_windows,
    log_dir=log_path,
    data_dir=data_path,
    checkpoint_dir=checkpoint_path,
    verbosity=Verbosity.LAGRANGIAN
  )
  e_state = ETrainingState(
    id=experiment_id,
    lagrangian_mu=e_config.lagrangian_start_mu,
    lagrangian_lambda=e_config.lagrangian_start_lambda,
    loss_hist=deque([], lagrangian_patience_batches or 1),
    constraint_hist=deque([], lagrangian_patience_batches or 1),
    convergence_test_hist=deque([], global_convergence_patience_windows or 1),
  )
  print('[Experiment {}]'.format(experiment_id), e_config)
  device = torch.device('cuda' if use_cuda else 'cpu')
  if logger is None and comet_api_key is not None:
    logger = CometLogger(comet_api_key, comet_tag, e_config.to_tensorboard_dict())
  val_eval, train_eval = Experiment(e_state, device, e_config, logger).train()

  results = {
    'e_state': e_state,
    'e_config': e_config,
    'final_results_val': val_eval,
    'final_results_train': train_eval,
  }
  with open(results_path / '{}.pkl'.format(experiment_id), mode='wb') as results_file:
    pickle.dump(results, results_file)

if __name__ == '__main__':
  try:
    fire.Fire()
  except KeyboardInterrupt:
    print("Manually ended training")
