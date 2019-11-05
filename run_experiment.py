#!/usr/bin/env python3
import time
from typing import List

import numpy as np
from torch import multiprocessing as mp

from experiment import Experiment
from experiment_config import (
  ComplexityType, DatasetType, EConfig, ETrainingState, ModelType)

# Has to be top-level fn, in order to be pickled by mp.Pool
def _train_process(x,y):
  return Experiment(x,y).train()

# New Experiment
def launch_experiment(runs: int = 1, complexities: List[float] = [0, 0.1, 0.2, 0.3, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.7, 0.8, 0.9, 1.0]):
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
      )
      args_queue.append((e_state, e_config))
  with mp.Pool() as pool:
    results = pool.starmap(_train_process, args_queue)

  results = [(cfg[1].complexity_lambda, r[0], r[1]) for cfg, r in zip(args_queue, results)]
  np.save('results', results)

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
  launch_experiment(3)
