import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_helpers import get_dataset_properties
from experiment_config import DatasetType, EConfig, ModelType

def get_model_for_config(e_config: EConfig) -> nn.Module:
    if e_config.model_type == ModelType.DEEP:
        return DeepNet(30, 2, DatasetType.MNIST)
    raise KeyError()

class ExperimentBaseModel(nn.Module):
    def get_params(self):
        """Return parameters of the neural network as a vector"""
        return torch.cat([p.data.view(-1) for p in self.parameters()], dim=0)

    def forward(self, x):
        raise NotImplementedError()

    def get_weight_norms(self, p=2):
        raise NotImplementedError()

class DeepNet(ExperimentBaseModel):
    """Neural Net with num_layers layers"""
    def __init__(self, num_hidden: int, num_layers: int, dataset_type: DatasetType):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dataset_properties = get_dataset_properties(dataset_type)
        self.layers = nn.ModuleList(
            [nn.Linear(self.dataset_properties.D,num_hidden)] + # Input
            [nn.Linear(num_hidden,num_hidden) for _ in range(num_layers-1)] + # Hidden
            [nn.Linear(num_hidden,self.dataset_properties.K)]) # Output

    def forward(self,x):
        x = x.view(-1,self.dataset_properties.D)

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        if self.dataset_properties.is_classification:
            x = F.log_softmax(x, dim=1)
        
        return x

    def get_weight_norms(self, p=2):
        wn = [x for x in self.layers.parameters()]
        wn = [w.norm(p) for w in wn]
        return wn
