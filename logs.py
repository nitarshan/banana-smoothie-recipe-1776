from experiment_config import Verbosity
from comet_ml import Experiment as CometExperiment
from torch.utils.tensorboard import SummaryWriter
from experiment_config import DatasetSubsetType

class BaseLogger(object):
  def log_metrics(self, step:int, metrics:dict):
    raise NotImplementedError()

  def log_hparams(self, hps:dict, metrics: dict):
    raise NotImplementedError()

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
  def __init__(self, api_key:str):
    self.writer = CometExperiment(
      api_key=api_key,
      project_name='causal-complexity-measures',
      workspace='nitarshan',
      auto_metric_logging=False,
      auto_output_logging="simple")

  def log_metrics(self, step:int, metrics:dict):
    self.writer.log_metrics(metrics, step=step)

  def log_hparams(self, hps:dict, metrics: dict):
    self.writer.log_parameters(hps)

  def __del__(self):
    self.writer.end()

class Printer(object):
  def __init__(self, experiment_id: int, verbosity: Verbosity):
    self.experiment_id = experiment_id
    self.verbosity = verbosity
  
  def epoch_metrics(self, epoch: int, train_eval, val_eval) -> None:
    if self.verbosity >= Verbosity.EPOCH:
      print('[{}][Epoch {}][GEN L: {:.2g} E: {:.2f}pp][{} L: {:.4g}, C: {:.4g}, A: {:.0f}/{} ({:.2f}%)][{} L: {:.4g}, C: {:.4g}, A: {:.0f}/{} ({:.2f}%)]'.format(
        self.experiment_id, epoch,
        train_eval[1] - val_eval[1], 100. * (train_eval[0] - val_eval[0]),
        DatasetSubsetType.VAL.name, val_eval[1], val_eval[2], val_eval[3], val_eval[4], 100. * val_eval[0],
        DatasetSubsetType.TRAIN.name, train_eval[1], train_eval[2], train_eval[3], train_eval[4], 100. * train_eval[0]))