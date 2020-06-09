#!/usr/bin/env python3
import pickle
import time

import simple_parsing
import torch

from source.experiment import Experiment
from source.experiment_config import HParams, State
from source.logs import WandbLogger


if __name__=='__main__':
  # Prepare experiment settings
  parser = simple_parsing.ArgumentParser()
  parser.add_arguments(HParams, dest="config")
  parser.add_arguments(State, dest="state")
  
  args = parser.parse_args()
  hparams: HParams = args.config
  state: State = args.state

  experiment_id = time.time_ns()
  state.id = experiment_id
  state.ce_check_milestones = hparams.ce_target_milestones.copy()

  # Run experiment
  device = torch.device('cuda' if hparams.use_cuda else 'cpu')
  logger = WandbLogger('default', hparams.to_tensorboard_dict(), hparams.wandb_md5)
  def dump_results(epoch, val_eval, train_eval):
    results = {
      'e_state': state,
      'e_config': hparams,
      'final_results_val': val_eval,
      'final_results_train': train_eval,
    }
    with open(hparams.results_dir / '{}.pkl'.format(experiment_id, epoch), mode='wb') as results_file:
      pickle.dump(results, results_file)

  Experiment(state, device, hparams, logger, dump_results).train()
