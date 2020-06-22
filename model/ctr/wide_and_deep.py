import torch
import torch.nn as nn

from model.basic.output_layer import OutputLayer
from model.basic.mlp import MLP
from model.ctr.lr import LR


class WideAndDeep(nn.Module):
    def __init__(self, emb_dim, num_feats, num_cate_fields, num_cont_fields, num_cross_feats, fc_dims=None, dropout=None,
                 batch_norm=None, out_type='binary'):
        super(WideAndDeep, self).__init__()
        self.emb_dim = emb_dim
        self.num_feats = num_feats
        self.num_cate_fields = num_cate_fields
        self.num_cont_fields = num_cont_fields
        self.num_cross_feats = num_cross_feats
        if not fc_dims:
            fc_dims = [32, 32]
        self.emb_layer = nn.Embedding(num_embeddings=num_feats - num_cont_fields, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        self.deep = MLP(num_cont_fields + num_cate_fields * emb_dim, fc_dims, dropout, batch_norm)
        self.wide = LR(num_cross_feats, out_type='regression')
        self.out_layer = OutputLayer(in_dim=fc_dims[-1] + 1, out_type=out_type)

    def forward(self, continuous_value, categorical_index, cross_feat_index):
        cate_emb_value = self.emb_layer(categorical_index)  # N * num_cate_fields * emb_dim
        cate_emb_value = cate_emb_value.reshape(
            (-1, self.num_cate_fields * self.emb_dim))  # N * (num_cate_fields * emb_dim)
        deep_in = torch.cat([continuous_value, cate_emb_value], 1)  # N * (num_cate_fields * emb_dim + num_cont_fields)
        y_deep = self.deep(deep_in)
        y_wide = self.wide(cross_feat_index)
        y = torch.cat([y_deep, y_wide])
        y = self.out_layer(y)
        return y
