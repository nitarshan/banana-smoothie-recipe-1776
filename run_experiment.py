#!/usr/bin/env python3
from pathlib import Path
import time
from typing import List

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

# Has to be top-level fn, in order to be pickled by mp.Pool
def _train_process(x,y):
  sns.set()
  device = torch.device('cuda' if y.cuda else 'cpu', mp.current_process()._identity[0]-1 if y.cuda else 0)
  return Experiment(x, device, y).train()

# New Experiment
def deep_mnist_l2_experiment(
  root_dir: str,
  runs: int = 3,
  complexities: List[float] = [0, 0.05, 0.1],
  use_cuda: bool = False
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
        epochs=50,
        save_epoch_freq = 50,
        log_tensorboard=True,
        complexity_type=ComplexityType.L2,
        complexity_lambda=complexity_lambda,
        log_dir=log_path,
        data_dir=data_path,
        verbosity=Verbosity.EPOCH
      )
      args_queue.append((e_state, e_config))

  num_processes = 1 if use_cuda else mp.cpu_count()
  print('[Experiment {}] # Processes {}'.format(experiment_id, num_processes))
  with mp.Pool(num_processes) as pool:
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

if __name__ == '__main__':
  mp.set_start_method('spawn')
  fire.Fire(deep_mnist_l2_experiment)
