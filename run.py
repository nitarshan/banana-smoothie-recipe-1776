#!/usr/bin/env python3
import pickle
import time
from collections import deque

import torch
import simple_parsing

from ccm.experiment import Experiment
from ccm.experiment_config import EConfig, ETrainingState
from ccm.logs import WandbLogger


if __name__=='__main__':
  # Prepare experiment settings
  parser = simple_parsing.ArgumentParser()
  parser.add_arguments(EConfig, dest="config")
  parser.add_arguments(ETrainingState, dest="state")
  
  args = parser.parse_args()
  cfg: EConfig = args.config
  state: ETrainingState = args.state

  experiment_id = time.time_ns()
  state.id = experiment_id
  state.loss_hist = deque([], cfg.lagrangian_patience_batches or 1)
  state.subepoch_ce_check_milestones = cfg.global_convergence_evaluation_freq_milestones.copy()

  # Run experiment
  device = torch.device('cuda' if cfg.use_cuda else 'cpu')
  logger = WandbLogger('default', cfg.to_tensorboard_dict())
  def dump_results(epoch, val_eval, train_eval):
    results = {
      'e_state': state,
      'e_config': cfg,
      'final_results_val': val_eval,
      'final_results_train': train_eval,
    }
    with open(cfg.results_dir / '{}.epoch{}.pkl'.format(experiment_id, epoch), mode='wb') as results_file:
      pickle.dump(results, results_file)

  Experiment(state, device, cfg, logger, dump_results).train()
