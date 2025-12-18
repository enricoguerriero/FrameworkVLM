import torch.nn as nn
import torch

class ClassifierHead(nn.Module):

    def __init__(self, in_dim, dims, num_classes, activation="relu", dropout=0.2, bias=None):
        super().__init__()
        
        dims = [in_dim] + dims + [num_classes]
        act_lookup = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}
        act_cls = act_lookup.get(activation.lower(), nn.ReLU)
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=(bias is not None)))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1], bias=(bias is not None)))
        if bias is not None:
            with torch.no_grad():
                layers[-1].bias.copy_(bias)
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)