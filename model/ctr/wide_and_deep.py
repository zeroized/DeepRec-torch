import torch
import torch.nn as nn

from model.basic.output_layer import OutputLayer
from model.basic.mlp import MLP

"""
Model: WDL: Wide and Deep Learning
Version: DLRS 2016
Reference: Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, 
            Greg Corrado, Wei Chai, Mustafa Ispir, Rohan Anil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, 
            and Hemal Shah. 2016. 
           Wide & Deep Learning for Recommender Systems. 
           In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems (DLRS 2016). 
           Association for Computing Machinery, New York, NY, USA, 7–10. 
           DOI:https://doi.org/10.1145/2988450.2988454
"""


class WideAndDeep(nn.Module):
    def __init__(self, emb_dim, num_feats, num_cate_fields, num_cont_fields, num_cross_feats, fc_dims=None,
                 dropout=None, batch_norm=None, out_type='binary'):
        super(WideAndDeep, self).__init__()
        self.emb_dim = emb_dim
        self.num_feats = num_feats
        self.num_cate_fields = num_cate_fields
        self.num_cont_fields = num_cont_fields
        self.num_cross_feats = num_cross_feats

        # first order weight for category features
        self.cate_weights = nn.Embedding(num_embeddings=num_feats - num_cont_fields, embedding_dim=1)
        nn.init.xavier_uniform_(self.cate_weights.weight)

        # first order weight for continuous features
        self.cont_weights = nn.Linear(in_features=num_cont_fields, out_features=1)
        nn.init.xavier_uniform_(self.cont_weights)

        self.wide_bias = nn.Parameter(torch.randn(1))

        if not fc_dims:
            fc_dims = [32, 32]
        fc_dims.append(1)
        self.fc_dims = fc_dims

        # embedding for deep network
        self.emb_layer = nn.Embedding(num_embeddings=num_feats - num_cont_fields, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        self.deep = MLP(num_cont_fields + num_cate_fields * emb_dim, fc_dims, dropout, batch_norm)
        self.out_layer = OutputLayer(in_dim=1, out_type=out_type)

    def forward(self, continuous_value, categorical_index, cross_feat_index):
        first_order_cate = self.cate_weights(categorical_index)
        first_order_cont = self.cont_weights(continuous_value)
        y_wide = first_order_cate + first_order_cont + self.wide_bias

        cate_emb_value = self.emb_layer(categorical_index)  # N * num_cate_fields * emb_dim
        # N * (num_cate_fields * emb_dim)
        cate_emb_value = cate_emb_value.reshape((-1, self.num_cate_fields * self.emb_dim))
        deep_in = torch.cat([continuous_value, cate_emb_value], 1)  # N * (num_cate_fields * emb_dim + num_cont_fields)
        y_deep = self.deep(deep_in)  # N * 1
        y = y_deep + y_wide
        y = self.out_layer(y)
        return y
