import torch
import torch.nn as nn
from model.basic.mlp import MLP
from model.basic.output_layer import OutputLayer
from model.basic.functional import bi_interaction


class DCN(nn.Module):
    def __init__(self, emb_dim, num_feats, num_cate_fields, num_cont_fields, cross_depth, fc_dims=None,
                 dropout=None, batch_norm=None, out_type='binary'):
        super(DCN, self).__init__()
        self.emb_dim = emb_dim
        self.num_feats = num_feats
        self.num_cate_fields = num_cate_fields
        self.num_cont_fields = num_cont_fields

        self.cross_depth = cross_depth
        # embedding for category features
        self.emb_layer = nn.Embedding(num_embeddings=num_feats - num_cont_fields, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        # deep network
        if not fc_dims:
            fc_dims = [32, 32]
        self.fc_dims = fc_dims
        x0_dim = num_cont_fields + num_cate_fields * emb_dim
        self.deep = MLP(x0_dim, fc_dims, dropout, batch_norm)

        # cross network
        cross_layers = []
        for _ in range(cross_depth):
            cross_layers.append(CrossLayer(x0_dim))
        self.cross = nn.ModuleList(cross_layers)

        self.out_layer = OutputLayer(in_dim=fc_dims[-1] + x0_dim, out_type=out_type)

    def forward(self, continuous_value, categorical_index):
        cate_emb_value = self.emb_layer(categorical_index)  # N * num_cate_fields * emb_dim
        # N * (num_cate_fields * emb_dim)
        cate_emb_value = cate_emb_value.reshape((-1, self.num_cate_fields * self.emb_dim))
        x0 = torch.cat([continuous_value, cate_emb_value], 1)

        y_dnn = self.deep(x0)

        xi = x0
        for cross_layer in self.cross_depth:
            xi = cross_layer(x0, xi)

        output = torch.cat([y_dnn, xi], dim=1)
        output = self.out_layer(output)
        return output


class CrossLayer(nn.Module):
    def __init__(self, x_dim):
        super(CrossLayer, self).__init__()
        self.x_dim = x_dim
        self.weights = nn.Parameter(torch.zeros(x_dim, 1))  # x_dim * 1
        nn.init.xavier_uniform_(self.weights.data)
        self.bias = nn.Parameter(torch.randn(x_dim))  # x_dim

    def forward(self, x0, xi):
        # x0,x1: N * x_dim
        x = torch.mul(xi, self.weights)  # N * x_dim
        x = torch.sum(x, dim=1)  # N
        x = x.unsqueeze(dim=1)  # N * 1
        x = torch.mul(x, x0)  # N * x_dim
        x = x + self.bias + xi
        return x
