import time
from typing import Optional, Tuple

from comet_ml import Experiment as CometExperiment
import numpy as np
import torch
import torch.nn.functional as F

from dataset_helpers import get_dataloaders
from experiment_config import (
  DatasetSubsetType, EConfig, ETrainingState, LagrangianType, OptimizerType,
  Verbosity)
from logs import BaseLogger, DefaultLogger, Printer
from models import get_model_for_config


class Experiment:
  def __init__(self, e_state:ETrainingState, device: torch.device, e_config: Optional[EConfig]=None,
               logger: Optional[BaseLogger]=None):
    self.e_state = e_state
    self.device = device
    resume_from_checkpoint = (e_config is None) or e_config.resume_from_checkpoint

    if resume_from_checkpoint:
      cfg_state, model_state, optim_state, np_rng_state, torch_rng_state = self.load_state()
      self.cfg: EConfig = cfg_state
    if e_config is not None:
      self.cfg: EConfig = e_config

    # Random Seeds
    torch.manual_seed(self.cfg.seed)
    np.random.seed(self.cfg.seed)

    # Logging
    if logger is None:
      log_file = self.cfg.log_dir / self.cfg.model_type.name / self.cfg.dataset_type.name / self.cfg.optimizer_type.name / self.cfg.complexity_type.name / str(self.cfg.complexity_lambda) / str(self.cfg.complexity_normalization) / str(self.e_state.id)
      self.logger = DefaultLogger(log_file)
    else:
      self.logger = logger
    # Printing
    self.printer = Printer(self.e_state.id, self.cfg.verbosity)

    # Model
    self.model = get_model_for_config(self.cfg)
    self.model.to(device)

    # Optimizer
    if self.cfg.optimizer_type == OptimizerType.SGD:
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr)
    elif self.cfg.optimizer_type == OptimizerType.SGD_MOMENTUM:
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9)
    elif self.cfg.optimizer_type == OptimizerType.ADAM:
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
    else:
      raise KeyError

    self.risk_objective = F.nll_loss

    # Load data
    self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.cfg.dataset_type, self.cfg.data_dir, self.cfg.batch_size, self.device)

    # Cleanup when resuming from checkpoint
    if resume_from_checkpoint:
      self.model.load_state_dict(model_state)
      self.optimizer.load_state_dict(optim_state)
      np.random.set_state(np_rng_state)
      torch.set_rng_state(torch_rng_state)

  def _train_epoch(self) -> None:
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):
      global_batch_idx = (self.e_state.epoch - 1) * len(self.train_loader) + batch_idx
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      self.optimizer.zero_grad()
      output, complexity = self.model(data, self.cfg.complexity_type)
      cross_entropy = self.risk_objective(output, target)
      loss = cross_entropy

      if self.cfg.complexity_lambda is not None:
        loss += self.cfg.complexity_lambda * complexity

      constraint = torch.zeros(1, device=data.device)
      do_constraint_optimization = (
        self.cfg.lagrangian_type != LagrangianType.NONE
        and self.e_state.epoch >= self.cfg.lagrangian_start_epoch
      )
      if do_constraint_optimization:
        constraint = (complexity - self.cfg.lagrangian_target) ** 2
        if self.cfg.lagrangian_type == LagrangianType.PENALTY:
          loss += self.e_state.lagrangian_mu * constraint ** 2
        elif self.cfg.lagrangian_type == LagrangianType.AUGMENTED:
          loss += self.e_state.lagrangian_mu / 2 * constraint ** 2 + self.e_state.lagrangian_lambda * constraint

      loss.backward()
      self.optimizer.step()

      update_constraint_optimization_parameters = (
        do_constraint_optimization
        and global_batch_idx % self.cfg.lagrangian_patience_batches == 0
        and constraint > self.cfg.lagrangian_tolerance
      )
      if update_constraint_optimization_parameters:
        if self.e_state.prev_loss is None:
          loss_delta = None
        else:
          loss_delta = loss - self.e_state.prev_loss
        self.e_state.prev_loss = loss.item()

        update_lagrangian_lambda = (
          self.cfg.lagrangian_type == LagrangianType.AUGMENTED
          and loss_delta is not None
          and (
            loss_delta > 0
            or torch.abs(loss_delta) < self.cfg.lagrangian_lambda_omega
          )
        )
        if update_lagrangian_lambda:
          self.e_state.lagrangian_lambda += self.e_state.lagrangian_mu * constraint.item()
          print('[{}][Epoch {} Batch {}] Increasing Lagrangian alpha to {:.2g}'.format(
                self.e_state.id, self.e_state.epoch, batch_idx, self.e_state.lagrangian_lambda))

        update_prev_constraint = (
          self.cfg.lagrangian_type == LagrangianType.PENALTY
          or update_lagrangian_lambda
        )
        if update_prev_constraint:
          update_lagrangian_mu = (
            self.e_state.prev_constraint is not None
            and (constraint.item() > self.cfg.lagrangian_improvement_rate * self.e_state.prev_constraint)
          )
          if update_lagrangian_mu:
            self.e_state.lagrangian_mu *= 10
            print('[{}][Epoch {} Batch {}] Increasing Lagrangian mu to {:.2g}'.format(
              self.e_state.id, self.e_state.epoch, batch_idx, self.e_state.lagrangian_mu))
            # Reset the optimizer as we've changed the objective
            if self.cfg.optimizer_type == OptimizerType.SGD_MOMENTUM:
              self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9)
            elif self.cfg.optimizer_type == OptimizerType.ADAM:
              self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
          self.e_state.prev_constraint = constraint.item()

      if self.cfg.verbosity >= Verbosity.BATCH and self.cfg.log_batch_freq is not None and batch_idx % self.cfg.log_batch_freq == 0:
        print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          self.e_state.id, self.e_state.epoch, batch_idx * len(data), len(self.train_loader.dataset), 100. * batch_idx / len(self.train_loader), loss.item()))

      # Log everything
      if self.cfg.log_batch_freq is not None and global_batch_idx % self.cfg.log_batch_freq == 0:
        # Collect metrics for logging
        metrics={'cross_entropy/train_minibatch': cross_entropy.item(),
                 'complexity/{}/train_minibatch'.format(self.cfg.complexity_type.name): complexity.item(),
                 'loss/train_minibatch': loss.item()}
        if self.cfg.lagrangian_type != LagrangianType.NONE:
          metrics.update({'constraint_mu/train_minibatch': self.e_state.lagrangian_mu})
          if self.cfg.lagrangian_type == LagrangianType.AUGMENTED:
            metrics.update({'constraint_lambda/train_minibatch': self.e_state.lagrangian_lambda,
                            'constraint/train_minibatch': constraint.item()})
        # Send metrics to logger
        self.logger.log_metrics(step=global_batch_idx, metrics=metrics)

  def train(self):
    self.printer.train_start(self.device)
        
    for epoch in range(self.e_state.epoch, self.cfg.epochs + 1):
      self.e_state.epoch = epoch
      self._train_epoch()

      if epoch==1 or epoch==self.cfg.epochs or epoch % self.cfg.log_epoch_freq == 0:
        val_eval = self.evaluate(DatasetSubsetType.VAL)
        train_eval = self.evaluate(DatasetSubsetType.TRAIN)

        self.logger.log_metrics(
          self.e_state.epoch,
          {
            'generalization/error': train_eval[0] - val_eval[0],
            'generalization/loss': train_eval[1] - val_eval[1],
          })
        
        self.printer.epoch_metrics(self.e_state.epoch, train_eval, val_eval)

      if self.cfg.save_epoch_freq is not None and epoch % self.cfg.save_epoch_freq == 0:
        self.save_state()

    self.printer.train_end()

    del self.logger
    return val_eval, train_eval

  @torch.no_grad()
  def evaluate(self, dataset_subset_type: DatasetSubsetType):
    self.model.eval()
    total_loss = 0
    num_correct = 0

    data_loader = [self.train_loader, self.val_loader, self.test_loader][dataset_subset_type]
    num_to_evaluate_on = len(data_loader.dataset)
    complexities = []

    for data, target in data_loader:
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      prob, com = self.model(data, self.cfg.complexity_type)
      complexities.append(com)
      total_loss += self.risk_objective(prob, target, reduction='sum').item()  # sum up batch loss
      pred = prob.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
      batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
      num_correct += batch_correct.sum()

    avg_loss = total_loss / num_to_evaluate_on
    complexity_loss = sum(complexities).item() / len(complexities)
    acc = num_correct / num_to_evaluate_on

    self.logger.log_metrics(
      self.e_state.epoch,
      {
        'complexity/{}/{}'.format(self.cfg.complexity_type.name, dataset_subset_type.name): complexity_loss,
        'cross_entropy/{}'.format(dataset_subset_type.name): avg_loss,
        'accuracy/{}'.format(dataset_subset_type.name): acc.item(),
      })
    if self.cfg.epochs == self.e_state.epoch:
      self.logger.log_hparams(
        self.cfg.to_tensorboard_dict(),
        {
          'hparam/complexity': complexity_loss,
          'hparam/accuracy': acc.item(),
          'hparam/loss': avg_loss
        })

    return acc, avg_loss, complexity_loss, num_correct, num_to_evaluate_on

  def save_state(self) -> None:
    checkpoint_path = self.cfg.checkpoint_dir / str(self.e_state.id)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_path / (str(self.e_state.epoch) + '.pt')
    torch.save({
      'config': self.cfg,
      'model': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'np_rng': np.random.get_state(),
      'torch_rng': torch.get_rng_state(),
    }, checkpoint_file)

  def load_state(self) -> Tuple[EConfig, dict, dict, np.ndarray, torch.ByteTensor]:
    checkpoint_file = self.cfg.checkpoint_dir / str(self.e_state.id) / (str(self.e_state.epoch - 1) + '.pt')
    checkpoint = torch.load(checkpoint_file)
    return checkpoint['config'], checkpoint['model'], checkpoint['optimizer'], checkpoint['np_rng'], checkpoint['torch_rng']
