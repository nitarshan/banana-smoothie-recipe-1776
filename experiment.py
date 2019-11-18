import pathlib
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset_helpers import get_dataloaders
from experiment_config import (
  ComplexityType, EConfig, ETrainingState, OptimizerType, Verbosity)
from measures import calculate_complexity
from models import get_model_for_config

class Experiment:
  def __init__(self, e_state:ETrainingState, device: torch.device, e_config: Optional[EConfig]=None):
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

    # Model
    self.model = get_model_for_config(self.cfg)
    self.model.to(device)

    # Optimizer
    if self.cfg.optimizer_type == OptimizerType.SGD:
      self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.lr)
    else:
      raise KeyError

    self.risk_objective = F.nll_loss

    # Load data
    self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.cfg.dataset_type, self.cfg.data_dir)

    # Cleanup when resuming from checkpoint
    if resume_from_checkpoint:
      self.model.load_state_dict(model_state)
      self.optimizer.load_state_dict(optim_state)
      np.random.set_state(np_rng_state)
      torch.set_rng_state(torch_rng_state)

    if self.cfg.log_tensorboard:
      log_file = self.cfg.log_dir / str(self.e_state.id) / self.cfg.complexity_type.name / str(self.cfg.complexity_lambda)
      self.writer = SummaryWriter(log_file)
      #self.writer.add_graph(self.model, self.train_loader.dataset[0][0])

  def _train_epoch(self) -> None:
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      self.optimizer.zero_grad()
      output = self.model(data)
      risk = self.risk_objective(output, target)
      complexity = torch.zeros(1)
      if self.cfg.complexity_type == ComplexityType.L2:
        complexity = calculate_complexity(self.model, self.cfg.complexity_type)
      loss = risk + self.cfg.complexity_lambda * complexity
      loss.backward()
      self.optimizer.step()

      if self.cfg.log_tensorboard:
        self.writer.add_scalar('train/risk', risk.item(), self.e_state.epoch * len(self.train_loader) + batch_idx)
        self.writer.add_scalar('train/complexity', complexity.item(), self.e_state.epoch * len(self.train_loader) + batch_idx)
        self.writer.add_scalar('train/loss', loss.item(), self.e_state.epoch * len(self.train_loader) + batch_idx)

      if self.cfg.verbosity >= Verbosity.BATCH and self.cfg.log_batch_freq is not None and batch_idx % self.cfg.log_batch_freq == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            self.e_state.epoch,
            batch_idx * len(data),
            len(self.train_loader.dataset),
            100. * batch_idx / len(self.train_loader),
            loss.item()))
    if self.e_state.epoch % self.cfg.save_epoch_freq == 0:
      self.save_state()

  def train(self):
    if self.cfg.verbosity >= Verbosity.RUN:
      start_time = time.time()
      print('[{}] Training starting using {}'.format(self.e_state.id, self.device))
    
    for epoch in range(self.e_state.epoch, self.cfg.epochs + 1):
      self.e_state.epoch = epoch
      self._train_epoch()
      self.evaluate(type=1)

    if self.cfg.verbosity >= Verbosity.RUN:
      print('[{}] Training complete in {}s'.format(self.e_state.id, time.time() - start_time))

    if self.cfg.log_tensorboard:
      self.writer.flush()
      self.writer.close()
    return self.evaluate(type=1, verbose=False)

  @torch.no_grad()
  def evaluate(self, type, verbose=True):
    self.model.eval()
    total_loss = 0
    num_correct = 0
    correct = torch.FloatTensor(0, 1)

    data_loader = [self.train_loader, self.val_loader, self.test_loader][type]
    num_to_evaluate_on = len(data_loader.dataset)

    for data, target in data_loader:
      data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
      prob = self.model(data)
      total_loss += self.risk_objective(prob, target, reduction='sum').item()  # sum up batch loss
      pred = prob.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
      batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
      correct = torch.cat([correct, batch_correct], 0)
      num_correct += batch_correct.sum()

    avg_loss = total_loss / num_to_evaluate_on
    complexity_loss = calculate_complexity(self.model, self.cfg.complexity_type).item()
    acc = num_correct / num_to_evaluate_on
    if verbose and self.cfg.verbosity >= Verbosity.EPOCH:
      print('After {} epochs, {} loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(self.e_state.epoch,
            ['Training', 'Validation', 'Test'][type],
            avg_loss, num_correct, num_to_evaluate_on, 100. * acc))
    if verbose and self.cfg.log_tensorboard:
      self.writer.add_scalar('val/acc', acc, self.e_state.epoch)
      self.writer.add_scalar('val/loss', avg_loss, self.e_state.epoch)
      if self.cfg.epochs == self.e_state.epoch:
        self.writer.add_hparams(self.cfg.to_tensorboard_dict(), {'hparam/accuracy': acc, 'hparam/loss': avg_loss})
    return acc, avg_loss, complexity_loss, correct

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

  def load_state(self) -> (EConfig, dict, dict, np.ndarray, torch.ByteTensor):
    checkpoint_file = pathlib.Path('checkpoints') / str(self.e_state.id) / (str(self.e_state.epoch - 1) + '.pt')
    checkpoint = torch.load(checkpoint_file)
    return checkpoint['config'], checkpoint['model'], checkpoint['optimizer'], checkpoint['np_rng'], checkpoint['torch_rng']
