from copy import deepcopy
from typing import Optional
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from .dataset_helpers import get_dataloaders
from .experiment_config import (
  ComplexityType, DatasetSubsetType, EConfig, ETrainingState, EvaluationMetrics, OptimizerType)
from .lagrangian import Lagrangian
from .logs import BaseLogger, Printer
from .measures import get_all_measures, get_single_measure
from .models import get_model_for_config


class Experiment:
  def __init__(
    self,
    e_state: ETrainingState,
    device: torch.device,
    e_config: EConfig,
    logger: BaseLogger,
    result_save_callback: Optional[object] = None
  ):
    self.e_state = e_state
    self.device = device
    self.cfg = e_config
    
    # Random Seeds
    random.seed(self.cfg.seed)
    np.random.seed(self.cfg.seed)
    torch.manual_seed(self.cfg.seed)
    torch.cuda.manual_seed_all(self.cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Logging
    self.logger = logger
    # Printing
    self.printer = Printer(self.e_state.id, self.cfg.verbosity)
    self.result_save_callback = result_save_callback

    # Model
    self.model = get_model_for_config(self.cfg)
    print("Number of parameters", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    self.model.to(device)
    self.init_model = deepcopy(self.model)

    # Optimizer
    self.optimizer = self._reset_optimizer()

    # Constrained Optimization handler
    self.lagrangian = Lagrangian(
      self.cfg.lagrangian_type,
      self.cfg.lagrangian_tolerance,
      self.cfg.lagrangian_target,
      self.cfg.lagrangian_start_epoch,
      self.cfg.lagrangian_improvement_rate,
      self.cfg.lagrangian_patience_batches,
      self.cfg.lagrangian_convergence_tolerance,
      self.cfg.lagrangian_start_mu,
      self.cfg.lagrangian_start_lambda,
      self.cfg.global_convergence_patience,
      self.cfg.complexity_lambda,
      self.logger,
    )

    # Load data
    self.train_loader, self.train_eval_loader, self.test_loader = get_dataloaders(self.cfg, self.device)

    # Resume from checkpoint if available
    self.load_state()

  def save_state(self, postfix: str = '') -> None:
    checkpoint_file = self.cfg.checkpoint_dir / (self.cfg.md5 + postfix + '.pt')
    torch.save({
      'config': self.cfg,
      'state': self.e_state,
      'model': self.model.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      # 'lagrangian': self.lagrangian,
      'np_rng': np.random.get_state(),
      'torch_rng': torch.get_rng_state(),
    }, checkpoint_file)

  def load_state(self) -> None:
    try:
      checkpoint_file = self.cfg.checkpoint_dir / (self.cfg.md5 + '.pt')
      checkpoint = torch.load(checkpoint_file)
      if checkpoint is not None:
        self.e_state = checkpoint['state']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.lagrangian = checkpoint['lagrangian']
        np.random.set_state(checkpoint['np_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
      print(f'loading from checkpoint at epoch {self.e_state.epoch} global batch {self.e_state.global_batch}')
    except FileNotFoundError:
      print('no checkpoint found')

  def _reset_optimizer(self) -> torch.optim.Optimizer:
    if self.cfg.optimizer_type == OptimizerType.SGD:
      return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr)
    elif self.cfg.optimizer_type == OptimizerType.SGD_MOMENTUM:
      return torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.9)
    elif self.cfg.optimizer_type == OptimizerType.ADAM:
      return torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
    else:
      raise KeyError
  
  def _train_epoch(self) -> None:
    self.model.train()
    ce_check_batches = [(len(self.train_loader)//(2**(self.e_state.subepoch_ce_check_freq))) * (i+1) for i in range(2**(self.e_state.subepoch_ce_check_freq)-1)]
    ce_check_batches.append(len(self.train_loader)-1)

    for batch_idx, (data, target) in enumerate(self.train_loader):
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      
      self.e_state.batch = batch_idx
      self.e_state.global_batch = (self.e_state.epoch - 1) * len(self.train_loader) + self.e_state.batch
      self.model.train()
      self.optimizer.zero_grad()
      
      if self.cfg.complexity_type in {ComplexityType.PATH_NORM, ComplexityType.PATH_NORM_OVER_MARGIN}:
        orig_model = deepcopy(self.model)

      logits = self.model(data)
      cross_entropy = F.cross_entropy(logits, target)

      cross_entropy.backward()
      loss = cross_entropy.clone()

      if self.cfg.complexity_type in {ComplexityType.PATH_NORM, ComplexityType.PATH_NORM_OVER_MARGIN}:
        for param in self.model.parameters():
          if param.requires_grad:
            param.data.pow_(2)

      complexity = get_single_measure(self.model, self.init_model, self.cfg.complexity_type, intervention_mode=True)

      # Assemble the loss function based on the optimization method
      complexity_loss, constraint, is_constrained = self.lagrangian.make_loss(torch.zeros((), device=cross_entropy.device), complexity, self.e_state.epoch)

      if is_constrained and self.cfg.complexity_type.name != 'NONE':
        complexity_loss.backward()

      loss += complexity_loss.clone()

      self.e_state.loss_hist.append(loss.item())
      
      self.model.train()

      if self.cfg.complexity_type in {ComplexityType.PATH_NORM, ComplexityType.PATH_NORM_OVER_MARGIN}:
        for param, orig_param in zip(self.model.parameters(), orig_model.parameters()):
          if param.requires_grad:
            param.data = orig_param.data

      self.optimizer.step()

      # Update lagrangian parameters
      check_global_convergence = (not is_constrained) and (self.e_state.global_batch % self.cfg.lagrangian_patience_batches == 0)
      if is_constrained:
        params_updated, check_global_convergence = self.lagrangian.update_parameters(np.mean(self.e_state.loss_hist), len(self.e_state.loss_hist), self.e_state.global_batch)
        if params_updated:
          self.printer.lagrangian_update(self.e_state, self.lagrangian.constraint_hist, self.lagrangian.lagrangian_mu, self.lagrangian.lagrangian_lambda)

      # Log everything
      self.printer.batch_end(self.cfg, self.e_state, data, self.train_loader, loss)
      self.logger.log_batch_end(self.cfg, self.e_state, self.lagrangian, cross_entropy, complexity, loss, constraint)

      # Cross-entropy stopping check
      if self.cfg.use_dataset_cross_entropy_stopping and batch_idx == ce_check_batches[0]:
        ce_check_batches.pop(0)
        is_last_batch = batch_idx == (len(self.train_loader)-1)
        dataset_ce = self.evaluate_cross_entropy(DatasetSubsetType.TRAIN, log=is_last_batch)[0]
        if dataset_ce < self.cfg.global_convergence_target:
          self.e_state.converged = True
        else:
          while len(self.e_state.subepoch_ce_check_milestones) > 0 and dataset_ce <= self.e_state.subepoch_ce_check_milestones[0]:
            passed_milestone = self.e_state.subepoch_ce_check_milestones[0]
            print(f'passed ce milestone {passed_milestone}')
            self.e_state.subepoch_ce_check_milestones.pop(0)
            self.e_state.subepoch_ce_check_freq += 1
            if self.cfg.save_epoch_freq is not None:
              self.save_state(f'_ce_{passed_milestone}')

      if self.e_state.converged:
        break

  def train(self) -> None:
    self.printer.train_start(self.device)
    train_eval, val_eval = None, None
    
    self.e_state.global_batch = 0
    for epoch in trange(self.e_state.epoch, self.cfg.epochs + 1, disable=(not self.cfg.use_tqdm)):
      self.e_state.epoch = epoch
      self._train_epoch()
      
      is_evaluation_epoch = (epoch==1 or epoch==self.cfg.epochs or epoch % self.cfg.log_epoch_freq == 0)
      if is_evaluation_epoch or self.e_state.converged:
        train_eval = self.evaluate(DatasetSubsetType.TRAIN, (epoch==self.cfg.epochs or self.e_state.converged))
        val_eval = self.evaluate(DatasetSubsetType.TEST)
        self.logger.log_generalization_gap(self.e_state, train_eval.acc, val_eval.acc, train_eval.avg_loss, val_eval.avg_loss, train_eval.complexity, train_eval.all_complexities)
        self.printer.epoch_metrics(self.cfg, self.e_state, self.lagrangian.constraint_hist, epoch, train_eval, val_eval)
      
      if epoch==self.cfg.epochs or self.e_state.converged:
        self.result_save_callback(epoch, val_eval, train_eval)

      # Save state
      is_save_epoch = self.cfg.save_epoch_freq is not None and (epoch % self.cfg.save_epoch_freq == 0 or epoch==self.cfg.epochs or self.e_state.converged)
      if is_save_epoch:
        self.save_state()

      if self.e_state.converged:
        print('Converged')
        break

    self.printer.train_end()

    if train_eval is None or val_eval is None:
      raise RuntimeError


  @torch.no_grad()
  def evaluate(self, dataset_subset_type: DatasetSubsetType, compute_all_measures: bool = False) -> EvaluationMetrics:
    self.model.eval()
    data_loader = [self.train_eval_loader, self.test_loader][dataset_subset_type]

    cross_entropy_loss, acc, complexity, constraint_loss, num_correct = self.evaluate_cross_entropy(dataset_subset_type, True, False)

    all_complexities = {}
    if dataset_subset_type == DatasetSubsetType.TRAIN and compute_all_measures:
      all_complexities = get_all_measures(self.model, self.init_model, data_loader, acc)

    self.logger.log_epoch_end(self.cfg, self.e_state, dataset_subset_type, cross_entropy_loss, acc)

    return EvaluationMetrics(acc, cross_entropy_loss, complexity, constraint_loss, num_correct, len(data_loader.dataset), all_complexities)

  @torch.no_grad()
  def evaluate_cross_entropy(self, dataset_subset_type: DatasetSubsetType, calculate_complexity: bool = False, log: bool = False):
    self.model.eval()
    cross_entropy_loss = 0
    constraint_loss = 0
    complexity = 0
    num_correct = 0

    data_loader = [self.train_eval_loader, self.test_loader][dataset_subset_type]
    num_to_evaluate_on = len(data_loader.dataset)

    for data, target in data_loader:
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      logits = self.model(data)
      if calculate_complexity:
        complexity = get_single_measure(self.model, self.init_model, self.cfg.complexity_type)
      cross_entropy = F.cross_entropy(logits, target, reduction='sum')
      cross_entropy_loss += cross_entropy.item()  # sum up batch loss
      if calculate_complexity:
        total_loss, _, _ = self.lagrangian.make_loss(cross_entropy, complexity, self.e_state.epoch)
        constraint_loss = (total_loss - cross_entropy).item()
      
      pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
      batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
      num_correct += batch_correct.sum()

    if calculate_complexity:
      complexity = complexity.item()
    cross_entropy_loss /= num_to_evaluate_on
    acc = num_correct.item() / num_to_evaluate_on
    
    if log:
      self.logger.log_epoch_end(self.cfg, self.e_state, dataset_subset_type, cross_entropy_loss, acc)
    return cross_entropy_loss, acc, complexity, constraint_loss, num_correct
