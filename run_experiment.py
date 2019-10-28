import argparse
from datetime import datetime

from experiment import Experiment
from experiment_config import DatasetType, EConfig, ETrainingState, ModelType

# Can optionally pass experiment parameters via command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1776)
args = parser.parse_args()

# New Experiment
e_state = ETrainingState(id=1572232475,epoch=3)
e_config = None

# Resume Training, reusing previous config (example)
# e_state = ETrainingState(id=1572230002, epoch=7)
# e_config = None

# Resume Training, with updated config (example)
# e_state = ETrainingState(id=1572230002, epoch=11)
# e_config = EConfig(
#   seed=args.seed,
#   cuda=False,
#   model_type= ModelType.DEEP,
#   dataset_type=DatasetType.MNIST,
#   batch_size=128,
#   epochs=20,
# )

Experiment(e_state, e_config).train()