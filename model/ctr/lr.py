import torch
import torch.nn as nn
from model.basic.output_layer import OutputLayer


class LR(nn.Module):
    def __init__(self, feat_dim, out_type='binary'):
        super(LR, self).__init__()
        self.feat_dim = feat_dim
        self.weights = nn.Embedding(num_embeddings=feat_dim, embedding_dim=1)
        self.bias = nn.Parameter(torch.randn(1))
        self.output_layer = OutputLayer(1, out_type)

    def forward(self, feat_index, feat_value):
        weights = self.weights(feat_index)  # N * F * 1
        feat_value = torch.unsqueeze(feat_value, dim=2)  # N * F * 1
        first_order = torch.mul(feat_value, weights)  # N * F * 1
        first_order = torch.squeeze(first_order, dim=2)  # N * F
        y = torch.sum(first_order, dim=1)
        y += self.bias

        y = self.output_layer(y)
        return y
