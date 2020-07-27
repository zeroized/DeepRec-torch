import torch
import torch.nn as nn
from model.basic.mlp import MLP
from model.ctr.fm import FM
from model.basic.output_layer import OutputLayer

"""
Model: DeepFM
Version: IJCAI 2017
Reference: Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). 
           DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. 
           Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence
"""


class DeepFM(nn.Module):

    def __init__(self, emb_dim, feat_dim, num_fields, fc_dims=None, dropout=None, batch_norm=None, out_type='binary'):
        super(DeepFM, self).__init__()
        # embedding layer is embedded in the FM sub-module
        self.emb_dim = emb_dim

        # fm
        self.fm = FM(emb_dim, feat_dim, out_type='regression')

        # dnn
        if not fc_dims:
            fc_dims = [32, 32, 32]
        self.fc_dims = fc_dims
        self.num_fields = num_fields
        self.dnn = MLP(emb_dim * num_fields, fc_dims, dropout, batch_norm)

        # output
        self.output_layer = OutputLayer(fc_dims[-1] + 1, out_type)

    def forward(self, feat_index, feat_value):
        # embedding
        emb_layer = self.fm.get_embedding()
        feat_emb = emb_layer(feat_index)

        # compute y_FM
        y_fm = self.fm(feat_index, feat_value)  # N
        y_fm = y_fm.unsqueeze(1)  # N * 1

        # compute y_dnn
        # reshape the embedding matrix to a vector
        dnn_in = feat_emb.reshape(-1, self.emb_dim * self.num_fields)  # N * (emb_dim * num_fields)
        y_dnn = self.dnn(dnn_in)  # N * fc_dims[-1]

        # compute output
        y = torch.cat((y_fm, y_dnn), dim=1)  # N * (fc_dims[-1] + 1)
        y = self.output_layer(y)
        return y
