import torch
import torch.nn as nn
from model.basic.output_layer import OutputLayer
from model.ctr.pnn import build_cross


class AFM(nn.Module):
    def __init__(self, emb_dim, feat_dim, num_fields, att_weight_dim, out_type='binary'):
        super(AFM, self).__init__()
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.num_fields = num_fields
        self.att_weight_dim = att_weight_dim
        self.first_order_weights = nn.Embedding(num_embeddings=feat_dim, embedding_dim=1)
        self.bias = nn.Parameter(torch.randn(1))
        self.emb_layer = nn.Embedding(num_embeddings=feat_dim, embedding_dim=emb_dim)
        self.num_pairs = num_fields * (num_fields - 1) / 2

        self.att_pooling_layer = AttentionPairWiseInteractionLayer(self.num_pairs, emb_dim, att_weight_dim)

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

        att_pooling = self.att_pooling_layer(feat_emb_value)  # N
        y = self.bias + y_first_order + att_pooling
        y = self.output_layer(y)
        return y


class AttentionPairWiseInteractionLayer(nn.Module):
    def __init__(self, num_pairs, emb_dim, att_weight_dim, activation=nn.ReLU()):
        super(AttentionPairWiseInteractionLayer, self).__init__()
        self.num_pairs = num_pairs
        self.emb_dim = emb_dim
        self.att_weight_dim = att_weight_dim
        self.att_weights = nn.Parameter(torch.randn(att_weight_dim, emb_dim))
        self.att_bias = nn.Parameter(torch.randn(att_weight_dim))
        self.att_h_weights = nn.Parameter(torch.randn(att_weight_dim))
        self.activation = activation
        self.pair_wise_weights = nn.Linear(in_features=num_pairs, out_features=1)

    def forward(self, feat_emb_value):
        p, q = build_cross(self.num_fields, feat_emb_value)  # N * num_pairs * emb_dim
        pair_wise_inter = torch.mul(p, q)  # N * num_pairs * emb_dim

        pair_wise_inter_T = pair_wise_inter.transpose(1, 2)  # N * emb_dim * num_pairs
        attention = torch.matmul(self.att_weights, pair_wise_inter_T)  # N * att_weight_dim * num_pairs
        attention = torch.add(attention, self.att_bias)  # N * att_weight_dim * num_pairs
        attention = nn.ReLU()(attention)
        att_signal = torch.matmul(self.att_h_weights, attention)  # N * num_pairs
        att_signal = nn.Softmax()(att_signal)  # N * num_pairs
        att_signal = att_signal.unsqueeze(2)  # N * num_pairs * 1

        att_pair_wise_inter = torch.mul(att_signal, pair_wise_inter)  # N * num_pairs * emb_dim
        att_pair_wise_sum = torch.sum(att_pair_wise_inter, dim=2)  # N * num_pairs

        att_pooling = self.pair_wise_weights(att_pair_wise_sum)  # N
        return att_pooling
