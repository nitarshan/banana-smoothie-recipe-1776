from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataset_helpers import get_dataloaders
from experiment_config import (
  DatasetSubsetType, EConfig, ETrainingState, EvaluationMetrics, LagrangianType,
  OptimizerType)
from logs import BaseLogger, DefaultLogger, Printer
from models import get_model_for_config
from torch.optim.lr_scheduler import MultiStepLR


class NanLossException(Exception):
  pass

class InfeasibleException(Exception):
  pass


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
      log_file = self.cfg.log_dir / self.cfg.model_type.name / self.cfg.dataset_type.name / self.cfg.optimizer_type.name / self.cfg.complexity_type.name / str(self.cfg.complexity_lambda) / str(self.e_state.id)
      self.logger = DefaultLogger(log_file)
    else:
      self.logger = logger
    # Printing
    self.printer = Printer(self.e_state.id, self.cfg.verbosity)

    # Model
    self.model = get_model_for_config(self.cfg)
    print("Number of parameters", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    self.model.to(device)

    # Optimizer
    self.optimizer = self._reset_optimizer()
    self.scheduler = self._reset_scheduler()

    self.risk_objective = F.cross_entropy

    # Load data
    self.train_loader, self.train_val_loader, self.val_loader, self.test_loader = get_dataloaders(self.cfg.dataset_type, self.cfg.data_dir, self.cfg.batch_size, self.device)

    # Cleanup when resuming from checkpoint
    if resume_from_checkpoint:
      self.model.load_state_dict(model_state)
      self.optimizer.load_state_dict(optim_state)
      np.random.set_state(np_rng_state)
      torch.set_rng_state(torch_rng_state)

  def _reset_optimizer(self) -> torch.optim.Optimizer:
    if self.cfg.optimizer_type == OptimizerType.SGD:
      return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr)
    elif self.cfg.optimizer_type == OptimizerType.SGD_MOMENTUM:
      return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9)
    elif self.cfg.optimizer_type == OptimizerType.ADAM:
      return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
    else:
      raise KeyError
  
  def _reset_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
    return MultiStepLR(self.optimizer, milestones=[60000], gamma=0.2) 

  def _make_train_loss(self, cross_entropy: torch.Tensor, complexity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Assembles a training loss function based on the optimization method of the experiment

    Parameters:
    -----------
    cross_entropy: torch.Tensor, dtype=float
      The cross-entropy of the model
    complexity: torch.Tensor, dtype=float
      The complexity of the model

    Returns:
    --------
    loss: torch.Tensor, dtype=float
      The value of the assembled loss function
    constraint: torch.Tensor, dtype=float
      The value of the constraint (constant for unconstrained opt)
    is_constraint: bool
      Whether or not the loss includes a constraint

    """
    loss = cross_entropy.clone()
    constraint = torch.zeros(1, device=cross_entropy.device)
    is_constrained = False

    # Unconstrained optimization (optionally with complexity as regularizer)
    if self.cfg.lagrangian_type == LagrangianType.NONE:
      if self.cfg.complexity_lambda is not None and self.cfg.complexity_lambda > 0:
        loss += self.cfg.complexity_lambda * complexity

    # Constrained optimization
    elif self.e_state.epoch >= self.cfg.lagrangian_start_epoch:
      constraint = torch.abs(complexity - self.cfg.lagrangian_target)
      is_constrained = True

      # Penalty method
      if self.cfg.lagrangian_type == LagrangianType.PENALTY:
        loss += (self.e_state.lagrangian_mu / 2) * constraint ** 2
      # Augmented Lagrangian Method
      elif self.cfg.lagrangian_type == LagrangianType.AUGMENTED:
        loss += (self.e_state.lagrangian_mu / 2) * constraint ** 2 + self.e_state.lagrangian_lambda * constraint
      # Other
      else:
        raise ValueError("Unknown optimization method specified.")

    return loss, constraint, is_constrained

  def _train_epoch(self) -> None:
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      
      self.e_state.batch = batch_idx
      self.e_state.global_batch = (self.e_state.epoch - 1) * len(self.train_loader) + self.e_state.batch
      self.optimizer.zero_grad()
      
      logits, complexity = self.model(data, self.cfg.complexity_type)
      cross_entropy = self.risk_objective(logits, target)

      # Assemble the loss function based on the optimization method
      loss, constraint, is_constrained = self._make_train_loss(cross_entropy, complexity)
      self.e_state.loss_hist.append(loss.item())
      self.e_state.constraint_hist.append(constraint.item())

      if torch.isnan(loss):
        raise NanLossException()

      # Update network parameters
      loss.backward()
      self.optimizer.step()

      # Update lagrangian parameters
      if is_constrained:
        self._update_constraint_parameters()

      # Log everything
      self.printer.batch_end(self.cfg, self.e_state, data, self.train_loader, loss)
      self.logger.log_batch_end(self.cfg, self.e_state, cross_entropy, complexity, loss, constraint)

  def _update_constraint_parameters(self) -> None:
    loss = np.mean(self.e_state.loss_hist)
    constraint = np.mean(self.e_state.constraint_hist)

    def _check_convergence():
      """Has the loss converged?"""
      if self.e_state.prev_loss is None:
        self.e_state.prev_loss = loss
        return False
      else:
        loss_delta = loss - self.e_state.prev_loss
        loss_improvement_rate = loss_delta / len(self.e_state.loss_hist)
        self.logger.log_metrics(step=self.e_state.global_batch,
                                metrics={"minibatch/loss_improvement_rate": loss_improvement_rate})
        self.e_state.prev_loss = loss
        return abs(loss_improvement_rate) < self.cfg.lagrangian_lambda_omega

    def _check_constraint_violated():
      """Is the constraint still violated?"""
      return constraint > self.cfg.lagrangian_tolerance * self.cfg.lagrangian_target

    def _check_constrained_improved_sufficiently():
      """Did the constraint improve sufficiently?"""
      return constraint <= self.cfg.lagrangian_improvement_rate * self.e_state.constraint_to_beat

    def _check_patience():
      """Have we reached the end of our patience?"""
      return self.e_state.global_batch % self.cfg.lagrangian_patience_batches == 0

    # Check if we have reached the end of a patience window and the constraint is still violated
    if _check_patience() and _check_constraint_violated():

      # Check if the subproblem has converged
      if self.e_state.prev_constraint is not None and _check_convergence():
        if not _check_constrained_improved_sufficiently():
          # Update mu
          self.e_state.lagrangian_mu *= 10
          self.printer.mu_increase(self.e_state)
          self.scheduler = self._reset_scheduler()

        else:
          # Update lambda
          if self.cfg.lagrangian_type == LagrangianType.AUGMENTED:
            self.e_state.lagrangian_lambda += self.e_state.lagrangian_mu * constraint.item()
            self.printer.lambda_increase(self.e_state)
            self.scheduler = self._reset_scheduler()
          self.e_state.constraint_to_beat = constraint

      self.e_state.prev_constraint = constraint

    if self.e_state.lagrangian_mu > 1e10 or self.e_state.lagrangian_lambda > 1e10:
      raise InfeasibleException()

  def train(self) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
    self.printer.train_start(self.device)
    
    self.e_state.global_batch = 0
    for epoch in range(self.e_state.epoch, self.cfg.epochs + 1):
      self.e_state.epoch = epoch
      self._train_epoch()
      self.scheduler.step(epoch)

      if epoch==1 or epoch==self.cfg.epochs or epoch % self.cfg.log_epoch_freq == 0:
        val_eval = self.evaluate(DatasetSubsetType.VAL)
        train_eval = self.evaluate(DatasetSubsetType.TRAIN)
        
        self.logger.log_generalization_gap(self.e_state, train_eval.acc, val_eval.acc, train_eval.avg_loss, val_eval.avg_loss)
        self.printer.epoch_metrics(self.cfg, self.e_state, self.e_state.epoch, train_eval, val_eval)

      if self.cfg.save_epoch_freq is not None and epoch % self.cfg.save_epoch_freq == 0:
        self.save_state()

    self.logger.log_train_end(self.cfg)
    self.printer.train_end()

    del self.logger
    return val_eval, train_eval

  @torch.no_grad()
  def evaluate(self, dataset_subset_type: DatasetSubsetType) -> EvaluationMetrics:
    self.model.eval()
    cross_entropy_loss = 0
    constraint_loss = 0
    complexity = 0
    num_correct = 0

    data_loader = [self.train_val_loader, self.val_loader, self.test_loader][dataset_subset_type]
    num_to_evaluate_on = len(data_loader.dataset)

    for data, target in data_loader:
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      logits, complexity = self.model(data, self.cfg.complexity_type)
      cross_entropy = self.risk_objective(logits, target, reduction='sum')
      cross_entropy_loss += cross_entropy.item()  # sum up batch loss
      total_loss, _, _ = self._make_train_loss(cross_entropy, complexity)
      constraint_loss = (total_loss - cross_entropy).item()
      
      pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
      batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
      num_correct += batch_correct.sum()

    complexity = complexity.item()
    cross_entropy_loss /= num_to_evaluate_on
    acc = num_correct / num_to_evaluate_on

    self.logger.log_epoch_end(self.cfg, self.e_state, dataset_subset_type, cross_entropy_loss, acc, complexity)

    return EvaluationMetrics(acc, cross_entropy_loss, complexity, constraint_loss, num_correct, len(data_loader.dataset))

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
