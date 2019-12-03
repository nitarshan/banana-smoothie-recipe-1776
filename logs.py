from torch.utils.tensorboard import SummaryWriter


class BaseLogger(object):
    def log_metrics(self, step:int, metrics:dict):
        raise NotImplementedError()

    def log_hparams(self, hps:dict):
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