import torch
import torch.nn as nn
import numpy as np
from .enum.activation_enum import *


class MLP(nn.Module):

    def __init__(self, fc_in_dim, fc_dims, dropout=None, batch_norm=None, activation=nn.ReLU()):
        """
        The MLP(Multi-Layer Perceptrons) module
        :param fc_in_dim: The dimension of input tensor
        :param fc_dims: The num_neurons of each layer, should be array-like
        :param dropout: The dropout rate of the MLP module, can be number or array-like ranges (0,1), by default None
        :param batch_norm: Whether to use batch normalization after each layer, by default None
        :param activation: The activation function used in each layer, by default nn.ReLU()
        """
        super(MLP, self).__init__()
        self.fc_dims = fc_dims
        layer_dims = [fc_in_dim]
        layer_dims.extend(fc_dims)
        layers = []

        if not dropout:
            dropout = np.repeat(0, len(fc_dims))
        if isinstance(dropout, float):
            dropout = np.repeat(dropout, len(fc_dims))

        for i in range(len(layer_dims) - 1):
            fc_layer = nn.Linear(in_features=layer_dims[i], out_features=layer_dims[i + 1])
            nn.init.xavier_uniform_(fc_layer.weight)
            layers.append(fc_layer)
            if batch_norm:
                batch_norm_layer = nn.BatchNorm1d(num_features=layer_dims[i + 1])
                layers.append(batch_norm_layer)
            layers.append(activation)
            if dropout[i]:
                dropout_layer = nn.Dropout(dropout[i])
                layers.append(dropout_layer)
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature):
        y = self.mlp(feature)
        return y
