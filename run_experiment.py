#!/usr/bin/env python3
from pathlib import Path
import time
from typing import List, Optional
import pickle

import fire
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import multiprocessing as mp

from experiment import Experiment
from experiment_config import (
  ComplexityType, DatasetType, EConfig, ETrainingState, LagrangianType, ModelType, OptimizerType, Verbosity)

# Has to be top-level fn, in order to be pickled by mp.Pool
def _train_process(x: ETrainingState, y: EConfig):
  sns.set()
  device = torch.device('cuda' if y.use_cuda else 'cpu', ((mp.current_process()._identity[0]-1) % torch.cuda.device_count()) if y.use_cuda else 0)
  return Experiment(x, device, y).train()

def setup_paths(root_dir: str, experiment_id: int):
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

# Leverage multiprocessing to parallelize experiments
def multi(
  root_dir: str,
  runs: int = 3,
  complexities: List[float] = [0, 0.05, 0.1],
  use_cuda: bool = False
) -> None:
  experiment_id = time.time_ns()
  print('[Experiment {}]'.format(experiment_id))
  print("[Experiment {}] CPU cores:".format(experiment_id), mp.cpu_count())
  print("[Experiment {}] CUDA devices:".format(experiment_id), torch.cuda.device_count())

  results_path, log_path, data_path, checkpoint_path = setup_paths(root_dir, experiment_id)

  args_queue = []
  for complexity_lambda in complexities:
    for _ in range(runs):
      id = time.time_ns()
      seed = id % (2**32)
      e_state = ETrainingState(id=id)
      e_config = EConfig(
        seed=seed,
        use_cuda=use_cuda,
        model_type= ModelType.DEEP,
        dataset_type=DatasetType.MNIST,
        batch_size=128,
        epochs=10,
        save_epoch_freq = 50,
        log_tensorboard=True,
        complexity_type=ComplexityType.L2,
        complexity_lambda=complexity_lambda,
        log_dir=log_path,
        data_dir=data_path,
        verbosity=Verbosity.EPOCH
      )
      args_queue.append((e_state, e_config))

  if use_cuda:
    results = [_train_process(x, y) for x, y in args_queue]
  else:
    print('[Experiment {}] Parallel Experiment Threads {}'.format(experiment_id, mp.cpu_count()))
    with mp.Pool(mp.cpu_count()) as pool:
      results = pool.starmap(_train_process, args_queue)

  results = [(cfg[1].complexity_lambda, r[0], r[1], r[2]) for cfg, r in zip(args_queue, results)]
  results = np.array(results)
  results = pd.DataFrame(results, columns=['lambda', 'val_acc', 'val_risk', 'complexity'])
  results.to_pickle(results_path / '{}.pkl'.format(experiment_id))
  fig = sns.lineplot(x='lambda', y='val_acc', data=results).set_title('L2 Lambda vs Accuracy')
  writer.add_figure('{}/lambda_vs_acc'.format(experiment_id), fig.get_figure(), 1, True)
  fig = sns.lineplot(x='lambda', y='val_risk', data=results).set_title('L2 Lambda vs Empirical Risk')
  writer.add_figure('{}/lambda_vs_risk'.format(experiment_id), fig.get_figure(), 1, True)
  fig = sns.lineplot(x='lambda', y='complexity', data=results).set_title('L2 Lambda vs Complexity')
  writer.add_figure('{}/lambda_vs_complexity'.format(experiment_id), fig.get_figure(), 1, True)
  writer.flush()
  writer.close()

# Run a single
def single(
  root_dir: str,
  model_type: str,
  dataset_type: str,
  optimizer_type: str,
  lr: float,
  epochs: int,
  batch_size: int,
  complexity_type: str,
  complexity_lambda: Optional[float],
  lagrangian_type: str = LagrangianType.NONE.name,
  lagrangian_target: Optional[float] = None,
  lagrangian_start_epoch: Optional[int] = None,
  lagrangian_start_mu: Optional[float] = None,
  lagrangian_tolerance: Optional[float] = None,
  lagrangian_patience_batches: Optional[int] = None,
  lagrangian_improvement_rate: Optional[float] = None,
  lagrangian_start_lambda: Optional[float] = None,
  lagrangian_lambda_omega: Optional[float] = None,
  use_cuda: bool = True,
  log_tensorboard: bool = False,
  save_epoch_freq: Optional[int] = None,
) -> None:
  experiment_id = time.time_ns()
  print('[Experiment {}]'.format(experiment_id))
  print("[Experiment {}] CPU cores:".format(experiment_id), mp.cpu_count())
  print("[Experiment {}] CUDA devices:".format(experiment_id), torch.cuda.device_count())

  results_path, log_path, data_path, checkpoint_path = setup_paths(root_dir, experiment_id)

  seed = experiment_id % (2**32)
  e_config = EConfig(
    seed=seed,
    use_cuda=use_cuda,
    model_type= ModelType[model_type],
    dataset_type=DatasetType[dataset_type],
    batch_size=batch_size,
    optimizer_type=OptimizerType[optimizer_type],
    lr=lr,
    epochs=epochs,
    save_epoch_freq=save_epoch_freq,
    log_tensorboard=log_tensorboard,
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
    lagrangian_lambda_omega=lagrangian_lambda_omega,
    log_dir=log_path,
    data_dir=data_path,
    checkpoint_dir=checkpoint_path,
    verbosity=Verbosity.EPOCH
  )
  e_state = ETrainingState(
    id=experiment_id,
    lagrangian_mu=e_config.lagrangian_start_mu,
    lagrangian_lambda=e_config.lagrangian_start_lambda
  )
  print('[Experiment {}]'.format(experiment_id), e_config)
  device = torch.device('cuda' if use_cuda else 'cpu')
  val_eval, train_eval = Experiment(e_state, device, e_config).train()

  results = {
    'e_state': e_state,
    'e_config': e_config,
    'final_results_val': val_eval,
    'final_results_train': train_eval,
  }
  with open(results_path / '{}.pkl'.format(experiment_id), mode='wb') as results_file:
    pickle.dump(results, results_file)

if __name__ == '__main__':
  mp.set_start_method('spawn')
  try:
    fire.Fire()
  except KeyboardInterrupt:
    print("Manually ended training")
