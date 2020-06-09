#!/usr/bin/env python3
import pickle
import time

import simple_parsing
import torch

from source.experiment import Experiment
from source.experiment_config import HParams, ETrainingState
from source.logs import WandbLogger


if __name__=='__main__':
  # Prepare experiment settings
  parser = simple_parsing.ArgumentParser()
  parser.add_arguments(HParams, dest="config")
  parser.add_arguments(ETrainingState, dest="state")
  
  args = parser.parse_args()
  cfg: HParams = args.config
  state: ETrainingState = args.state

  experiment_id = time.time_ns()
  state.id = experiment_id
  state.ce_check_milestones = cfg.ce_target_milestones.copy()

  # Run experiment
  device = torch.device('cuda' if cfg.use_cuda else 'cpu')
  logger = WandbLogger('default', cfg.to_tensorboard_dict(), cfg.wandb_md5)
  def dump_results(epoch, val_eval, train_eval):
    results = {
      'e_state': state,
      'e_config': cfg,
      'final_results_val': val_eval,
      'final_results_train': train_eval,
    }
    with open(cfg.results_dir / '{}.pkl'.format(experiment_id, epoch), mode='wb') as results_file:
      pickle.dump(results, results_file)

  Experiment(state, device, cfg, logger, dump_results).train()
