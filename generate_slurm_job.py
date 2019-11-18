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

# New Experiment
def main(
  root_dir: str,
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
  for complexity_lambda in [0, 0.05, 0.1]:
    for _ in range(3):
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
      print('python run.py {} {} {} {} {} {} {} {} {}'.format(

      ))
      

if __name__ == '__main__':
  mp.set_start_method('spawn')
  fire.Fire(main)
