import time
from typing import Dict, Optional

import numpy as np
import torch
import wandb

from .experiment_config import (
  ComplexityType, DatasetSubsetType, EConfig, ETrainingState, EvaluationMetrics,
  LagrangianType, Verbosity)
from .lagrangian import Lagrangian


class BaseLogger(object):
  def log_metrics(self, step: int, metrics: Dict[str, float]):
    raise NotImplementedError()

  def log_batch_end(
    self,
    cfg: EConfig,
    state: ETrainingState,
    lagrangian: Lagrangian,
    cross_entropy: torch.Tensor,
    complexity: torch.Tensor,
    loss: torch.Tensor,
    constraint: torch.Tensor
  ) -> None:
    if cfg.log_batch_freq is not None and state.global_batch % cfg.log_batch_freq == 0:
      # Collect metrics for logging
      metrics = {
        'cross_entropy/minibatch': cross_entropy.item(),
        'complexity/minibatch': complexity.item(),
        # 'complexity/{}/minibatch'.format(cfg.complexity_type.name): complexity.item(),
        'loss/minibatch': loss.item(),
        'loss/running_avg_{}_batches'.format(cfg.lagrangian_patience_batches): np.mean(state.loss_hist)
      }
      if cfg.lagrangian_type != LagrangianType.NONE:
        metrics.update({
          'constraint_mu/minibatch': lagrangian.lagrangian_mu,
          'constraint/minibatch': constraint.item(),
          'loss_constraint_only/minibatch': (loss.item() - cross_entropy.item())
        })
        if cfg.lagrangian_type == LagrangianType.AUGMENTED:
          metrics.update({
            'constraint_lambda/minibatch': lagrangian.lagrangian_lambda,
          })
      # Send metrics to logger
      self.log_metrics(step=state.global_batch, metrics=metrics)
  
  def log_generalization_gap(self, state: ETrainingState, train_acc: float, val_acc: float, train_loss: float, val_loss: float, complexity: float, all_complexities: Dict[ComplexityType, float]) -> None:
    self.log_metrics(
      state.global_batch,
      {
        'generalization/error': train_acc - val_acc,
        'generalization/loss': train_loss - val_loss,
        'complexity': complexity,
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

  def lagrangian_update(self, state: ETrainingState, constraint_hist, lagrangian_mu, lagrangian_lambda):
    if self.verbosity >= Verbosity.LAGRANGIAN:
      print('[{}][{}|{}][AL: {:.3f} AC: {:.3f}] Lagrangian mu to {:.2g} and lambda to {:.2g}'.format(
        state.id, state.epoch, state.batch, np.mean(state.loss_hist), np.mean(constraint_hist), lagrangian_mu, lagrangian_lambda))

  def epoch_metrics(self, cfg: EConfig, state: ETrainingState, constraint_hist, epoch: int, train_eval: EvaluationMetrics, val_eval: EvaluationMetrics) -> None:
    if self.verbosity >= Verbosity.EPOCH:
      print(
        '[{}][{}][GL: {:.2g} GE: {:.2f}pp][AL: {:.3f} AC: {:.3f}][C: {:.2g} C: {:.2g} T: {:.2g}][{} L: {:.4g}, A: {:.2f}%][{} L: {:.4g}, A: {:.2f}%][CL: {:.4g}]'.format(
          self.experiment_id, epoch,
          train_eval.avg_loss - val_eval.avg_loss, 100. * (train_eval.acc - val_eval.acc),
          np.mean(state.loss_hist), np.mean(constraint_hist),
          train_eval.complexity, train_eval.complexity - cfg.lagrangian_target, cfg.lagrangian_target * cfg.lagrangian_tolerance,
          DatasetSubsetType.TEST.name, val_eval.avg_loss, 100. * val_eval.acc,
          DatasetSubsetType.TRAIN.name, train_eval.avg_loss, 100. * train_eval.acc,
          train_eval.complexity_loss))
