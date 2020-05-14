#!/usr/bin/env python3
from pathlib import Path
import pickle
import time
from typing import Optional, Tuple, List
from collections import deque

from logs import CometLogger, WandbLogger
import fire
import torch

from experiment import Experiment
from experiment_config import (
  ComplexityType, DatasetType, EConfig, ETrainingState, LagrangianType,
  ModelType, OptimizerType, Verbosity)

def setup_paths(root_dir: str, experiment_id: int, data_dir: Optional[str]) -> Tuple[Path, Path, Path, Path]:
  print('[Experiment {}] Setting up directories'.format(experiment_id))
  root_path = Path(root_dir)
  results_path = root_path / 'results'
  results_path.mkdir(parents=True, exist_ok=True)
  log_path = root_path / 'logs'
  if data_dir is None:
    data_path = root_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
  else:
    data_path = Path(data_dir)
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
  global_convergence_method: Optional[str] = None,
  global_convergence_tolerance: Optional[float] = None,
  global_convergence_patience: Optional[int] = None,
  global_convergence_target: Optional[float] = None,
  global_convergence_evaluation_freq_milestones: Optional[List[float]] = None,
  use_cuda: bool = True,
  comet_api_key: Optional[str] = None,
  comet_tag: Optional[str] = None,
  logger: Optional[object] = None,
  log_epoch_freq: Optional[int] = 20,
  save_epoch_freq: Optional[int] = 1,
  seed: Optional[int] = None,
  data_seed: Optional[int] = None,
  data_dir: Optional[str] = None,
  use_wandb: bool = False,
  use_tqdm: bool = False,
  use_dataset_cross_entropy_stopping: bool = False,
  base_width: int = 32,
) -> None:
  experiment_id = time.time_ns()
  print('[Experiment {}]'.format(experiment_id))
  print("[Experiment {}] CUDA devices:".format(experiment_id), torch.cuda.device_count())

  results_path, log_path, data_path, checkpoint_path = setup_paths(root_dir, experiment_id, data_dir)

  if seed is None:
    seed = 0
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
    log_epoch_freq=log_epoch_freq,
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
    global_convergence_method=global_convergence_method,
    global_convergence_tolerance=global_convergence_tolerance,
    global_convergence_patience=global_convergence_patience,
    global_convergence_target=global_convergence_target,
    global_convergence_evaluation_freq_milestones=global_convergence_evaluation_freq_milestones,
    log_dir=log_path,
    data_dir=data_path,
    checkpoint_dir=checkpoint_path,
    verbosity=Verbosity.LAGRANGIAN,
    use_tqdm=use_tqdm,
    use_dataset_cross_entropy_stopping=use_dataset_cross_entropy_stopping,
    base_width=base_width,
  )
  e_state = ETrainingState(
    id=experiment_id,
    loss_hist=deque([], lagrangian_patience_batches or 1),
    subepoch_ce_check_milestones=global_convergence_evaluation_freq_milestones.copy(),
  )

  def dump_results(epoch, val_eval, train_eval):
    results = {
      'e_state': e_state,
      'e_config': e_config,
      'final_results_val': val_eval,
      'final_results_train': train_eval,
    }
    with open(results_path / '{}.epoch{}.pkl'.format(experiment_id, epoch), mode='wb') as results_file:
      pickle.dump(results, results_file)

  print('[Experiment {}]'.format(experiment_id), e_config)
  device = torch.device('cuda' if use_cuda else 'cpu')
  if logger is None:
    if use_wandb:
      logger = WandbLogger(comet_tag, e_config.to_tensorboard_dict())
    elif comet_api_key is not None:
      logger = CometLogger(comet_api_key, comet_tag, e_config.to_tensorboard_dict())
  val_eval, train_eval = Experiment(e_state, device, e_config, logger, dump_results).train()

if __name__ == '__main__':
  try:
    fire.Fire()
  except KeyboardInterrupt:
    print("Manually ended training")
