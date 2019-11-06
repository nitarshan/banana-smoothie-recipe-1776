#!/usr/bin/env python3
from pathlib import Path
import time
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from experiment import Experiment
from experiment_config import (
  ComplexityType, DatasetType, EConfig, ETrainingState, ModelType, Verbosity)

# Has to be top-level fn, in order to be pickled by mp.Pool
def _train_process(x,y):
  return Experiment(x,y).train()

# New Experiment
def deep_mnist_l2_experiment(runs: int, complexities: List[float]) -> None:
  experiment_id = time.time_ns()
  print('Experiment {}'.format(experiment_id))

  results_path = Path('results')
  results_path.mkdir(parents=True, exist_ok=True)
  log_path = Path('logs') / str(experiment_id)
  writer = SummaryWriter(log_path)
  data_path = Path('data')
  data_path.mkdir(parents=True, exist_ok=True)

  args_queue = []
  for complexity_lambda in complexities:
    for _ in range(runs):
      id = time.time_ns()
      seed = id % (2**32)
      e_state = ETrainingState(id=id)
      e_config = EConfig(
        seed=seed,
        cuda=False,
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
        verbosity=Verbosity.RUN
      )
      args_queue.append((e_state, e_config))
  with mp.Pool() as pool:
    results = pool.starmap(_train_process, args_queue)

  results = [(cfg[1].complexity_lambda, r[0], r[1], r[2]) for cfg, r in zip(args_queue, results)]
  results = np.array(results)
  results = pd.DataFrame(results, columns=['lambda', 'val_acc', 'val_risk', 'complexity'])
  results.to_pickle('results/{}.pkl'.format(experiment_id))
  fig = sns.lineplot(x='lambda', y='val_acc', data=results).set_title('L2 Lambda vs Accuracy')
  writer.add_figure('{}/lambda_vs_acc'.format(experiment_id), fig.get_figure(), 1, True)
  fig = sns.lineplot(x='lambda', y='val_risk', data=results).set_title('L2 Lambda vs Empirical Risk')
  writer.add_figure('{}/lambda_vs_risk'.format(experiment_id), fig.get_figure(), 1, True)
  fig = sns.lineplot(x='lambda', y='complexity', data=results).set_title('L2 Lambda vs Complexity')
  writer.add_figure('{}/lambda_vs_complexity'.format(experiment_id), fig.get_figure(), 1, True)
  writer.flush()
  writer.close()

# Resume Training, reusing previous config (example)
# e_state = ETrainingState(id=1572230002, epoch=7)
# e_config = None

# Resume Training, with updated config (example)
# e_state = ETrainingState(id=1572230002, epoch=11)
# e_config = EConfig(
#   seed=args.seed,
#   cuda=False,
#   model_type= ModelType.DEEP,
#   dataset_type=DatasetType.MNIST,
#   batch_size=128,
#   epochs=20,
# )

if __name__ == '__main__':
  sns.set()
  deep_mnist_l2_experiment(3, [0, 0.1, 0.2, 0.3, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.7, 0.8, 0.9, 1.0])
