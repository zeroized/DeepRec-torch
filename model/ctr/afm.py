import torch
import torch.nn as nn

from model.basic.attention import LocationBasedAttention
from model.basic.output_layer import OutputLayer
from model.ctr.pnn import build_cross

"""
Model: AFM: Attentional Factorization Machines
Version: IJCAI 2017
Reference: Xiao, J., Ye, H., He, X., Zhang, H., Wu, F., & Chua, T. (2017). 
           Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks. 
           Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence
"""


class AFM(nn.Module):
    def __init__(self, emb_dim, num_feats, num_fields, att_weight_dim, out_type='binary'):
        super(AFM, self).__init__()
        self.emb_dim = emb_dim
        self.num_feats = num_feats
        self.num_fields = num_fields
        self.att_weight_dim = att_weight_dim
        self.first_order_weights = nn.Embedding(num_embeddings=num_feats, embedding_dim=1)
        nn.init.xavier_uniform_(self.first_order_weights.weight)
        self.bias = nn.Parameter(torch.randn(1))
        self.emb_layer = nn.Embedding(num_embeddings=num_feats, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)
        self.num_pairs = num_fields * (num_fields - 1) / 2

        self.att_layer = LocationBasedAttention(emb_dim, att_weight_dim)
        self.p = nn.Parameter(torch.randn(emb_dim))

        self.output_layer = OutputLayer(1, out_type)

    def forward(self, feat_index, feat_value):
        feat_value = feat_value.unsqueeze(2)  # N * num_fields * 1
        # first order
        first_order_weight = self.first_order_weights(feat_index)  # N * num_fields * 1
        y_first_order = torch.mul(first_order_weight, feat_value)  # N * num_fields * 1
        y_first_order = torch.sum(y_first_order, dim=1)  # N * 1
        y_first_order = y_first_order.squeeze(1)

        feat_emb = self.emb_layer(feat_index)  # N * num_fields * emb_dim
        feat_emb_value = torch.mul(feat_emb, feat_value)  # N * num_fields * emb_dim

        p, q = build_cross(self.num_fields, feat_emb_value)  # N * num_pairs * emb_dim
        pair_wise_inter = torch.mul(p, q)

        att_signal = self.att_layer(pair_wise_inter)  # N * num_pairs
        att_signal = att_signal.unsqueeze(dim=2)  # N * num_pairs * 1

        att_inter = torch.mul(att_signal, pair_wise_inter)  # N * num_pairs * emb_dim
        att_pooling = torch.sum(att_inter, dim=1)  # N * emb_dim

        att_pooling = torch.mul(att_pooling, self.p)  # N * emb_dim
        att_pooling = torch.sum(att_pooling, dim=1)  # N

        y = self.bias + y_first_order + att_pooling
        y = self.output_layer(y)
        return y
