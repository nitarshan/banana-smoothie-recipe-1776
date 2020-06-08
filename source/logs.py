import time
from typing import Dict, Optional

import numpy as np
from torch import Tensor
import wandb

from .experiment_config import (
  ComplexityType, DatasetSubsetType, EConfig, ETrainingState, EvaluationMetrics,
  Verbosity)


class BaseLogger(object):
  def log_metrics(self, step: int, metrics: Dict[str, float]):
    raise NotImplementedError()

  def log_batch_end(
    self,
    cfg: EConfig,
    state: ETrainingState,
    cross_entropy: Tensor,
    loss: Tensor,
  ) -> None:
    if cfg.log_batch_freq is not None and state.global_batch % cfg.log_batch_freq == 0:
      # Collect metrics for logging
      metrics = {
        'cross_entropy/minibatch': cross_entropy.item(),
        'loss/minibatch': loss.item(),
      }
      # Send metrics to logger
      self.log_metrics(step=state.global_batch, metrics=metrics)
  
  def log_generalization_gap(self, state: ETrainingState, train_acc: float, val_acc: float, train_loss: float, val_loss: float, all_complexities: Dict[ComplexityType, float]) -> None:
    self.log_metrics(
      state.global_batch,
      {
        'generalization/error': train_acc - val_acc,
        'generalization/loss': train_loss - val_loss,
        **{'complexity/{}'.format(k.name): v for k,v in all_complexities.items()}
      })
  
  def log_epoch_end(self, cfg: EConfig, state: ETrainingState, datasubset: DatasetSubsetType, avg_loss: float, acc: float) -> None:
    self.log_metrics(
      state.global_batch,
      {
        'cross_entropy/{}'.format(datasubset.name.lower()): avg_loss,
        'accuracy/{}'.format(datasubset.name.lower()): acc,
      })


class WandbLogger(BaseLogger):
  def __init__(self, tag: Optional[str] = None, hps: Optional[dict] = None, group: Optional[str] = None):
    wandb.init(project='rgm', config=hps, tags=[tag], group=group)

  def log_metrics(self, step: int, metrics: dict):
    wandb.log(metrics, step=step)


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
        state.id, state.epoch, state.batch * len(data), len(loader.dataset), 100. * state.batch / len(loader),
        loss.item()))

  def epoch_metrics(self, cfg: EConfig, state: ETrainingState, epoch: int, train_eval: EvaluationMetrics, val_eval: EvaluationMetrics) -> None:
    if self.verbosity >= Verbosity.EPOCH:
      print(
        '[{}][{}][GL: {:.2g} GE: {:.2f}pp][AL: {:.3f} AC: {:.3f}][C: {:.2g} C: {:.2g} T: {:.2g}][{} L: {:.4g}, A: {:.2f}%][{} L: {:.4g}, A: {:.2f}%][CL: {:.4g}]'.format(
          self.experiment_id, epoch,
          train_eval.avg_loss - val_eval.avg_loss, 100. * (train_eval.acc - val_eval.acc),
          0.0, 0.0,
          0.0, 0.0, 0.0,
          DatasetSubsetType.TEST.name, val_eval.avg_loss, 100. * val_eval.acc,
          DatasetSubsetType.TRAIN.name, train_eval.avg_loss, 100. * train_eval.acc,
          0.0))
