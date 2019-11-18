#!/usr/bin/env python3
from pathlib import Path
import time

import fire
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from experiment import Experiment
from experiment_config import (
  ComplexityType, DatasetType, EConfig, ETrainingState, ModelType, Verbosity)

# New Experiment
def main(
  root_dir: str,
  runs: int,
  complexity: float,
  use_cuda: bool,
  seed: int,
  model_type: str,
  dataset_type: str,
  batch_size: int,
  epochs: int,
  save_epoch_freq: int,
  log_tensorboard: bool,
  complexity_type: str,
  complexity_lambda: float,
) -> None:
  experiment_id = time.time_ns()
  print('[Experiment {}]'.format(experiment_id))
  print("[Experiment {}] CPU cores:".format(experiment_id), mp.cpu_count())
  print("[Experiment {}] CUDA devices:".format(experiment_id), torch.cuda.device_count())

  print('[Experiment {}] Setting up directories'.format(experiment_id))
  root_path = Path(root_dir)
  results_path = root_path / 'results'
  results_path.mkdir(parents=True, exist_ok=True)
  print('[Experiment {}] Results path {}'.format(experiment_id, results_path))
  log_path = root_path / 'logs' / str(experiment_id)
  writer = SummaryWriter(log_path)
  print('[Experiment {}] Log path {}'.format(experiment_id, results_path))
  data_path = root_path / 'data'
  data_path.mkdir(parents=True, exist_ok=True)
  print('[Experiment {}] Data path {}'.format(experiment_id, results_path))

  id = time.time_ns()
  seed = id % (2**32)
  e_state = ETrainingState(id=id)
  e_config = EConfig(
    seed=seed,
    use_cuda=use_cuda,
    model_type= ModelType[model_type],
    dataset_type=DatasetType[dataset_type],
    batch_size=batch_size,
    epochs=epochs,
    save_epoch_freq = save_epoch_freq,
    log_tensorboard=log_tensorboard,
    complexity_type=ComplexityType[complexity_type],
    complexity_lambda=complexity_lambda,
    log_dir=log_path,
    data_dir=data_path,
    verbosity=Verbosity.EPOCH
  )

  sns.set()
  device = torch.device('cuda' if use_cuda else 'cpu')
  acc, avg_loss, complexity_loss, correct = Experiment(e_state, device, e_config).train()

  results = np.array([complexity_lambda, acc, avg_loss, complexity_loss])
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

if __name__ == '__main__':
  sns.set()
  fire.Fire(main)
