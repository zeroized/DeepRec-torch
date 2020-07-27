import torch
import torch.nn as nn
from model.basic.mlp import MLP
from model.basic.output_layer import OutputLayer
from model.basic.functional import bi_interaction

"""
Model: NFM: Neural Factorization Machines
Version: 
Reference: Xiangnan He and Tat-Seng Chua. 2017. 
           Neural Factorization Machines for Sparse Predictive Analytics. 
           In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information 
            Retrieval (SIGIR ’17). 
           Association for Computing Machinery, New York, NY, USA, 355–364. 
           DOI:https://doi.org/10.1145/3077136.3080777
"""

class NFM(nn.Module):
    def __init__(self, emb_dim, num_feats, num_fields, fc_dims=None, dropout=None, batch_norm=None, out_type='binary'):
        super(NFM, self).__init__()
        self.emb_dim = emb_dim
        self.num_feats = num_feats
        self.num_fields = num_fields

        self.first_order_weights = nn.Embedding(num_embeddings=num_feats, embedding_dim=1)
        nn.init.xavier_uniform_(self.first_order_weights.weight)
        self.first_order_bias = nn.Parameter(torch.randn(1))

        self.emb_layer = nn.Embedding(num_embeddings=num_feats, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        self.bi_intaraction_layer = BiInteractionLayer()
        if not fc_dims:
            fc_dims = [32, 32]
        self.fc_dims = fc_dims
        self.fc_layers = MLP(emb_dim, fc_dims, dropout, batch_norm)

        self.h = nn.Parameter(torch.zeros(1, fc_dims[-1]))  # 1 * fc_dims[-1]
        nn.init.xavier_uniform_(self.h.data)
        self.output_layer = OutputLayer(in_dim=1, out_type=out_type)

    def forward(self, feat_index, feat_value):
        # feat_index, feat_value: N * num_fields
        first_order_weights = self.first_order_weights(feat_index)  # N * num_fields * 1
        first_order_weights = first_order_weights.squeeze()
        first_order = torch.mul(feat_value, first_order_weights)  # N * num_fields
        first_order = torch.sum(first_order, dim=1)  # N

        feat_emb = self.emb_layer(feat_index)  # N * num_fields * emb_dim
        feat_value = feat_value.unsqueeze(dim=2)  # N * num_fields * 1
        feat_emb_value = torch.mul(feat_emb, feat_value)  # N * num_fields * emb_dim
        bi = self.bi_intaraction_layer(feat_emb_value)  # N * emb_dim

        fc_out = self.fc_layers(bi)  # N * fc_dims[-1]
        out = torch.mul(fc_out, self.h)  # N * fc_dims[-1]
        out = torch.sum(out, dim=1)  # N
        out = out + first_order + self.first_order_bias  # N
        out = out.unsqueeze(dim=1)  # N * 1
        out = self.output_layer(out)
        return out


class BiInteractionLayer(nn.Module):
    def __init__(self):
        super(BiInteractionLayer, self).__init__()

    def forward(self, feat_emb_value):
        # square_of_sum = torch.sum(feat_emb_value, dim=1)  # N * emb_dim
        # square_of_sum = torch.mul(square_of_sum, square_of_sum)  # N * emb_dim

        # sum_of_square = torch.mul(feat_emb_value, feat_emb_value)  # N * num_fields * emb_dim
        # sum_of_square = torch.sum(sum_of_square, dim=1)  # N * emb_dim

        # bi_out = square_of_sum - sum_of_square

        bi_out = bi_interaction(feat_emb_value)
        return bi_out
