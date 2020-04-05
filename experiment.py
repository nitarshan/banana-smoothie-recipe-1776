from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange

from convergence_detection import ConvergeOnPlateau
from dataset_helpers import get_dataloaders
from experiment_config import (
  DatasetSubsetType, EConfig, ETrainingState, EvaluationMetrics, LagrangianType,
  ModelType, OptimizerType)
from lagrangian import Lagrangian
from logs import BaseLogger, DefaultLogger, Printer
from measures import get_all_measures, get_single_measure
from models import get_model_for_config


class Experiment:
  def __init__(
    self,
    e_state: ETrainingState,
    device: torch.device,
    e_config: Optional[EConfig] = None,
    logger: Optional[BaseLogger] = None,
    result_save_callback: Optional[object] = None
  ):
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
    self.result_save_callback = result_save_callback

    # Model
    self.model = get_model_for_config(self.cfg)
    print("Number of parameters", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    self.model.to(device)
    self.init_model = deepcopy(self.model)

    # Optimizer
    self.optimizer = self._reset_optimizer()
    self.scheduler = self._reset_scheduler()
    self.detector = ConvergeOnPlateau(
      mode=e_config.global_convergence_method,
      patience=e_config.global_convergence_patience,
      threshold=e_config.global_convergence_tolerance,
      target=e_config.global_convergence_target,
      verbose=False
    )

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
    self.train_loader, self.train_eval_loader, self.val_loader, self.test_loader = get_dataloaders(self.cfg.dataset_type, self.cfg.data_dir, self.cfg.batch_size, self.device, self.cfg.data_seed)

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
  
  def _reset_scheduler(self) -> Optional[MultiStepLR]:
    if self.cfg.model_type == ModelType.RESNET:
      return None
      # https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
      #return MultiStepLR(self.optimizer, milestones=[25, 40], gamma=0.1)
    return None

  def _train_epoch(self) -> None:
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      
      self.e_state.batch = batch_idx
      self.e_state.global_batch = (self.e_state.epoch - 1) * len(self.train_loader) + self.e_state.batch
      self.optimizer.zero_grad()
      
      if self.cfg.complexity_type.name == 'PATH_NORM' or self.cfg.complexity_type.name == 'PATH_NORM_OVER_MARGIN':
        orig_model = deepcopy(self.model)

      logits = self.model(data)
      cross_entropy = F.cross_entropy(logits, target)

      cross_entropy.backward()
      loss = cross_entropy.clone()

      if self.cfg.complexity_type.name == 'PATH_NORM' or self.cfg.complexity_type.name == 'PATH_NORM_OVER_MARGIN':
        for param in self.model.parameters():
          if param.requires_grad:
            param.data.pow_(2)

      complexity = get_single_measure(self.model, self.init_model, self.cfg.complexity_type, intervention_mode=True)

      # Assemble the loss function based on the optimization method
      complexity_loss, constraint, is_constrained = self.lagrangian.make_loss(torch.zeros(()), complexity, self.e_state.epoch)

      if self.cfg.complexity_type.name != 'NONE':
        complexity_loss.backward()

      loss += complexity_loss.clone()

      self.e_state.loss_hist.append(loss.item())
      
      self.model.train()

      if self.cfg.complexity_type.name == 'PATH_NORM' or self.cfg.complexity_type.name == 'PATH_NORM_OVER_MARGIN':
        for param, orig_param in zip(self.model.parameters(), orig_model.parameters()):
          if param.requires_grad:
            param.data = orig_param.data

      self.optimizer.step()

      # Update lagrangian parameters
      check_global_convergence = (not is_constrained) and (self.e_state.global_batch % self.cfg.lagrangian_patience_batches == 0)
      if is_constrained:
        params_updated, check_global_convergence = self.lagrangian.update_parameters(np.mean(self.e_state.loss_hist), len(self.e_state.loss_hist), self.e_state.global_batch)
        if params_updated:
          self.scheduler = self._reset_scheduler()
          self.printer.lagrangian_update(self.e_state, self.lagrangian.constraint_hist, self.lagrangian.lagrangian_mu, self.lagrangian.lagrangian_lambda)
      
      if check_global_convergence:
        self.e_state.converged = self.detector.step(np.mean(self.e_state.loss_hist))

      # Log everything
      self.printer.batch_end(self.cfg, self.e_state, data, self.train_loader, loss)
      self.logger.log_batch_end(self.cfg, self.e_state, self.lagrangian, cross_entropy, complexity, loss, constraint)

      if self.e_state.converged:
        break

  def train(self) -> Tuple[EvaluationMetrics, EvaluationMetrics]:
    self.printer.train_start(self.device)
    train_eval, val_eval = None, None
    
    self.e_state.global_batch = 0
    for epoch in trange(self.e_state.epoch, self.cfg.epochs + 1, disable=(not self.cfg.use_tqdm)):
      self.e_state.epoch = epoch
      self._train_epoch()
      if self.scheduler is not None:
        self.scheduler.step() # DO NOT pass in epoch param, LR Scheduler is buggy

      if epoch==1 or epoch==self.cfg.epochs or epoch % self.cfg.log_epoch_freq == 0 or self.e_state.converged:
        train_eval = self.evaluate(DatasetSubsetType.TRAIN, (epoch==self.cfg.epochs or self.e_state.converged))
        val_eval = self.evaluate(DatasetSubsetType.VAL)
        self.logger.log_generalization_gap(self.e_state, train_eval.acc, val_eval.acc, train_eval.avg_loss, val_eval.avg_loss, train_eval.complexity, train_eval.all_complexities)
        self.printer.epoch_metrics(self.cfg, self.e_state, self.lagrangian.constraint_hist, self.e_state.epoch, train_eval, val_eval)
        self.result_save_callback(epoch, val_eval, train_eval)

      if self.cfg.save_epoch_freq is not None and epoch % self.cfg.save_epoch_freq == 0:
        self.save_state()
      
      if self.e_state.converged:
        print('Converged')
        break

    self.logger.log_train_end(self.cfg)
    self.printer.train_end()
    del self.logger

    if train_eval is None or val_eval is None:
      raise RuntimeError
    return val_eval, train_eval

  @torch.no_grad()
  def evaluate(self, dataset_subset_type: DatasetSubsetType, compute_all_measures: bool = False) -> EvaluationMetrics:
    self.model.eval()
    cross_entropy_loss = 0
    constraint_loss = 0
    complexity = 0
    num_correct = 0

    data_loader = [self.train_eval_loader, self.val_loader, self.test_loader][dataset_subset_type]
    num_to_evaluate_on = len(data_loader.dataset)

    for data, target in data_loader:
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      logits = self.model(data)
      complexity = get_single_measure(self.model, self.init_model, self.cfg.complexity_type)
      cross_entropy = F.cross_entropy(logits, target, reduction='sum')
      cross_entropy_loss += cross_entropy.item()  # sum up batch loss
      total_loss, _, _ = self.lagrangian.make_loss(cross_entropy, complexity, self.e_state.epoch)
      constraint_loss = (total_loss - cross_entropy).item()
      
      pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
      batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
      num_correct += batch_correct.sum()

    complexity = complexity.item()
    cross_entropy_loss /= num_to_evaluate_on
    acc = num_correct.item() / num_to_evaluate_on

    all_complexities = {}
    if dataset_subset_type == DatasetSubsetType.TRAIN and compute_all_measures:
      all_complexities = get_all_measures(self.model, self.init_model, data_loader, acc)

    self.logger.log_epoch_end(self.cfg, self.e_state, dataset_subset_type, cross_entropy_loss, acc)

    return EvaluationMetrics(acc, cross_entropy_loss, complexity, constraint_loss, num_correct, len(data_loader.dataset), all_complexities)

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
