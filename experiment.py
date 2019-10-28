from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.random import get_rng_state

from dataset_helpers import get_dataloaders
from experiment_config import (
  DatasetType, EConfig, ETrainingState, OptimizerType)
from models import get_model_for_config
import pathlib

class Experiment:
  def __init__(self, e_state:ETrainingState, e_config: Optional[EConfig]=None):
    self.e_state = e_state
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
    if self.cfg.cuda:
      print('Using CUDA')
      self.model.cuda()
    else:
      print('Using CPU')
    
    # Optimizer
    if self.cfg.optimizer_type == OptimizerType.SGD:
      self.optimizer = optim.SGD(self.model.parameters(), lr=self.cfg.lr)
    else:
      raise KeyError

    if self.cfg.dataset_type == DatasetType.REGRESSION:
      self.objective = F.mse_loss
    else:
      self.objective = F.nll_loss

    # Load data
    self.train_loader, self.val_loader, self.test_loader = get_dataloaders(self.cfg.dataset_type)

    # Cleanup when resuming from checkpoint
    if resume_from_checkpoint:
      self.model.load_state_dict(model_state)
      self.optimizer.load_state_dict(optim_state)
      np.random.set_state(np_rng_state)
      torch.set_rng_state(torch_rng_state)

  def _train_epoch(self) -> None:
    self.model.train()
    for batch_idx, (data, target) in enumerate(self.train_loader):
      if self.cfg.cuda:
        data, target = data.cuda(), target.cuda()

      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.objective(output, target)
      loss.backward()
      self.optimizer.step()

      if self.cfg.log_batch_freq is not None and batch_idx % self.cfg.log_batch_freq == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            self.e_state.epoch,
            batch_idx * len(data),
            len(self.train_loader.dataset),
            100. * batch_idx / len(self.train_loader),
            loss.item()))
    if self.e_state.epoch % self.cfg.save_epoch_freq == 0:
      self.save_state()

  def train(self) -> None:
    for epoch in range(self.e_state.epoch, self.cfg.epochs + 1):
      self.e_state.epoch = epoch
      self._train_epoch()
      self.evaluate(type=1)

    print('Training complete!! For hidden size = {} and layers = {}'.format(self.model.num_hidden, self.model.num_layers))
  
  @torch.no_grad()
  def evaluate(self, type, probs_required=False):
    self.model.eval()
    total_loss = 0
    num_correct = 0
    correct = torch.FloatTensor(0, 1)
    probs = None

    data_loader = [self.train_loader, self.val_loader, self.test_loader][type]
    num_to_evaluate_on = len(data_loader.dataset)

    for data, target in data_loader:
      if self.cfg.cuda:
        data, target = data.cuda(), target.cuda()
      prob = self.model(data)
      total_loss += self.objective(prob, target, reduction='sum').item()  # sum up batch loss
      if self.cfg.dataset_type != DatasetType.REGRESSION:
        pred = prob.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
        correct = torch.cat([correct, batch_correct], 0)
        num_correct += batch_correct.sum()
      if probs_required:
        if probs is None:
          probs = prob.data
        else:
          probs = torch.cat([probs, prob.data], 0)

    avg_loss = total_loss / num_to_evaluate_on
    if self.cfg.dataset_type != DatasetType.REGRESSION:
      acc = num_correct / num_to_evaluate_on
      print('\nAfter {} epochs ({} iterations), {} set: Average loss: {:.4f},Accuracy: {}/{} ({:.2f}%)\n'.format(self.e_state.epoch,
            self.e_state.epoch, ['Training', 'Validation', 'Test'][type],
            avg_loss, num_correct, num_to_evaluate_on, 100. * acc))
      if probs_required:
        return acc, avg_loss, correct, probs
      else:
        return acc, avg_loss, correct
    else:
      print('\nAfter {} epochs ({} iterations), {} set: Average loss: {:.4f}\n'.format(self.e_state.epoch,
        self.e_state.epoch, ['Training', 'Validation', 'Test'][type],
        avg_loss))
      if probs_required:
        return avg_loss, avg_loss, avg_loss, probs
      else:
        return avg_loss, avg_loss, avg_loss
  
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
