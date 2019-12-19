import time
from typing import Optional

from comet_ml import Experiment as CometExperiment
import torch
from torch.utils.tensorboard import SummaryWriter

from experiment_config import EConfig, ETrainingState, LagrangianType, Verbosity
from experiment_config import DatasetSubsetType

class BaseLogger(object):
  def log_metrics(self, step:int, metrics:dict):
    raise NotImplementedError()

  def log_hparams(self, hps:dict, metrics: dict):
    raise NotImplementedError()

  def log_batch_end(self, cfg: EConfig, state: ETrainingState, cross_entropy: torch.Tensor, complexity: torch.Tensor, loss: torch.Tensor, constraint: torch.Tensor):
    if cfg.log_batch_freq is not None and state.global_batch % cfg.log_batch_freq == 0:
      # Collect metrics for logging
      metrics = {
        'cross_entropy/minibatch': cross_entropy.item(),
        'complexity/minibatch': complexity.item(),
        #'complexity/{}/minibatch'.format(cfg.complexity_type.name): complexity.item(),
        'loss/minibatch': loss.item(),
        'loss/running_avg_{}_batches'.format(cfg.lagrangian_patience_batches): np.mean(state.cross_entropy_hist)
      }
      if cfg.lagrangian_type != LagrangianType.NONE:
        metrics.update({
          'constraint_mu/minibatch': state.lagrangian_mu,
          'constraint/minibatch': constraint.item(),
          'loss_constraint_only/minibatch': (loss.item() - cross_entropy.item())
        })
        if cfg.lagrangian_type == LagrangianType.AUGMENTED:
          metrics.update({
            'constraint_lambda/minibatch': state.lagrangian_lambda,
          })
      # Send metrics to logger
      self.log_metrics(step=state.global_batch, metrics=metrics)

class DefaultLogger(BaseLogger):
  """
  Log to Tensorboard
  """
  def __init__(self, log_file):
    self.log_file = log_file
    self.writer = SummaryWriter(log_file)

  def log_metrics(self, step:int, metrics:dict):
    for m, v in metrics.items():
      self.writer.add_scalar(m, v, step)

  def log_hparams(self, hps:dict, metrics:dict):
    self.writer.add_hparams(hps, metrics)

  def __del__(self):
    self.writer.flush()
    self.writer.close()

class CometLogger(BaseLogger):
  """
  Log to Comet.ml
  """
  def __init__(self, api_key:str, tag: Optional[str]=None, hps: Optional[dict]=None):
    self.writer = CometExperiment(
      api_key=api_key,
      project_name='causal-complexity-measures',
      workspace='nitarshan',
      auto_metric_logging=False,
      auto_output_logging="simple")
    if tag is not None:
      self.writer.add_tag(tag)
    if hps is not None:
      self.writer.log_parameters(hps)

  def log_metrics(self, step:int, metrics:dict):
    self.writer.log_metrics(metrics, step=step)

  def log_hparams(self, hps:dict, metrics: dict):
    pass

  def __del__(self):
    self.writer.end()

class Printer(object):
  def __init__(self, experiment_id: int, verbosity: Verbosity):
    self.experiment_id = experiment_id
    self.verbosity = verbosity
    self.start_time = None
  
  def train_start(self, device):
    if self.verbosity >= Verbosity.RUN:
      self.start_time = time.time()
      print('[{}] Training starting using {}'.format(self.experiment_id, device))
  
  def train_end(self):
    if self.verbosity >= Verbosity.RUN:
      print('[{}] Training complete in {}s'.format(self.experiment_id, time.time() - self.start_time))

  def batch_end(self, cfg: EConfig, state: ETrainingState, data, loader, loss):
    if self.verbosity >= Verbosity.BATCH and cfg.log_batch_freq is not None and state.batch % cfg.log_batch_freq == 0:
      print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        state.id, state.epoch, state.batch * len(data), len(loader.dataset), 100. * state.batch / len(loader), loss.item()))
  
  def lambda_increase(self, state: ETrainingState):
    if self.verbosity >= Verbosity.LAGRANGIAN:
      print('[{}][Epoch {} Batch {}] Increasing Lagrangian lambda to {:.2g}'.format(
        state.id, state.epoch, state.batch, state.lagrangian_lambda))
  
  def mu_increase(self, state: ETrainingState):
    if self.verbosity >= Verbosity.LAGRANGIAN:
      print('[{}][Epoch {} Batch {}] Increasing Lagrangian mu to {:.2g}'.format(
        state.id, state.epoch, state.batch, state.lagrangian_mu))
  
  def epoch_metrics(self, epoch: int, train_eval, val_eval) -> None:
    if self.verbosity >= Verbosity.EPOCH:
      print('[{}][Epoch {}][GEN L: {:.2g} E: {:.2f}pp][{} L: {:.4g}, C: {:.4g}, A: {:.0f}/{} ({:.2f}%)][{} L: {:.4g}, C: {:.4g}, A: {:.0f}/{} ({:.2f}%)]'.format(
        self.experiment_id, epoch,
        train_eval[1] - val_eval[1], 100. * (train_eval[0] - val_eval[0]),
        DatasetSubsetType.VAL.name, val_eval[1], val_eval[2], val_eval[3], val_eval[4], 100. * val_eval[0],
        DatasetSubsetType.TRAIN.name, train_eval[1], train_eval[2], train_eval[3], train_eval[4], 100. * train_eval[0]))