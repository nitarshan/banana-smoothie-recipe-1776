from collections import deque
from typing import Tuple

import numpy as np
import torch

from .experiment_config import LagrangianType

class Lagrangian(object):
  def __init__(
    self,
    lagrangian_type: LagrangianType,
    lagrangian_tolerance,
    lagrangian_target,
    lagrangian_start_epoch,
    lagrangian_improvement_rate,
    lagrangian_patience_batches,
    lagrangian_convergence_tolerance,
    lagrangian_mu,
    lagrangian_lambda,
    global_convergence_patience,
    complexity_lambda,
    logger,
  ):
    self.ltype = lagrangian_type
    self.lagrangian_tolerance = lagrangian_tolerance
    self.lagrangian_target = lagrangian_target
    self.lagrangian_start_epoch = lagrangian_start_epoch
    self.lagrangian_improvement_rate = lagrangian_improvement_rate
    self.lagrangian_patience_batches = lagrangian_patience_batches
    self.lagrangian_convergence_tolerance = lagrangian_convergence_tolerance

    self.lagrangian_mu = lagrangian_mu
    self.lagrangian_lambda = lagrangian_lambda

    self.complexity_lambda = complexity_lambda

    self.constraint_hist = deque([], lagrangian_patience_batches or 1)
    self.convergence_test_hist = deque([], global_convergence_patience or 1)
    self.prev_loss = None
    self.prev_constraint = None
    self.constraint_to_beat = np.infty

    self.logger = logger

  def make_loss(
    self,
    cross_entropy: torch.Tensor,
    complexity: torch.Tensor,
    epoch: int,
  ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    loss = cross_entropy.clone()
    constraint = torch.zeros(1, device=cross_entropy.device)
    is_constrained = False

    # Unconstrained optimization (optionally with complexity as regularizer)
    if self.ltype == LagrangianType.NONE:
      if self.complexity_lambda is not None and self.complexity_lambda > 0:
        loss += self.complexity_lambda * complexity

    # Constrained optimization
    elif epoch >= self.lagrangian_start_epoch:
      constraint = torch.abs(complexity - self.lagrangian_target)
      is_constrained = True

      # Penalty method
      if self.ltype == LagrangianType.PENALTY:
        loss += (self.lagrangian_mu / 2) * constraint ** 2
      # Augmented Lagrangian Method
      elif self.ltype == LagrangianType.AUGMENTED:
        loss += (self.lagrangian_mu / 2) * constraint ** 2 + self.lagrangian_lambda * constraint
      # Other
      else:
        raise ValueError("Unknown optimization method specified.")
    
    self.constraint_hist.append(constraint.item())

    if torch.isnan(loss):
      raise RuntimeError('NaN Lagrangian Loss')
    
    return loss, constraint, is_constrained


  def _check_convergence(self, loss: float, loss_hist_len: int, global_batch: int, tolerance: float, convergence_patience: int) -> bool:
    if self.prev_loss is None:
      self.prev_loss = loss
      return False
    else:
      loss_delta = loss - self.prev_loss
      loss_improvement_rate = loss_delta / loss_hist_len
      self.logger.log_metrics(step=global_batch,
                              metrics={"minibatch/loss_improvement_rate": loss_improvement_rate})
      self.prev_loss = loss
      self.convergence_test_hist.append(abs(loss_improvement_rate) < tolerance)
      #print(loss_improvement_rate, sum(self.convergence_test_hist))
      return sum(self.convergence_test_hist) >= convergence_patience


  def update_parameters(self, loss: float, loss_hist_len: int, global_batch: int) -> Tuple[bool, bool]:
    constraint = np.mean(self.constraint_hist)

    def _check_constraint_violated():
      """Is the constraint still violated?"""
      return constraint > self.lagrangian_tolerance * self.lagrangian_target

    def _check_constrained_improved_sufficiently():
      """Did the constraint improve sufficiently?"""
      return constraint <= self.lagrangian_improvement_rate * self.constraint_to_beat

    def _check_patience():
      """Have we reached the end of our patience?"""
      return global_batch % self.lagrangian_patience_batches == 0
    
    params_updated = False
    check_global_convergence = False

    # Check if we have reached the end of a patience window and the constraint is still violated
    if _check_patience() and _check_constraint_violated():
      # Check if the subproblem has converged
      if self.prev_constraint is not None and self._check_convergence(loss, loss_hist_len, global_batch, self.lagrangian_convergence_tolerance, 1):
        if not _check_constrained_improved_sufficiently():
          self.lagrangian_mu *= 10
          params_updated = True
        else:
          if self.ltype == LagrangianType.AUGMENTED:
            self.lagrangian_lambda += self.lagrangian_mu * constraint.item()
            params_updated = True
          self.constraint_to_beat = constraint
        
        if params_updated:
          self.prev_loss = None
          self.convergence_test_hist.clear()

      self.prev_constraint = constraint
    elif _check_patience() and not _check_constraint_violated():
      check_global_convergence = True

    if self.lagrangian_mu > 1e10 or self.lagrangian_lambda > 1e10:
      raise RuntimeError("Lagrangian parameters are infeasible")

    return params_updated, check_global_convergence
