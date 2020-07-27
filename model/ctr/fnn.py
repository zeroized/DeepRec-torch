import torch
import torch.nn as nn
from model.ctr.fm import FM
from model.basic.mlp import MLP
from model.basic.output_layer import OutputLayer

"""
Model: FNN: Factorization-machine supported Neural Network
Version: arXiv [v1] Mon, 11 Jan 2016 10:04:40 UTC
Reference: Zhang, W., Du, T., & Wang, J. (2016). 
           Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction. 
           arXiv: Learning,.
"""


class FNN(nn.Module):

    def __init__(self, emb_dim, num_feats, num_fields, fc_dims=None, dropout=None, batch_norm=None, out_type='binary',
                 train_fm=True):
        super(FNN, self).__init__()
        # set model object to training FNN or training FM embedding
        self.fm_trained = not train_fm

        # embedding layer is embedded in the FM sub-module
        self.emb_dim = emb_dim
        self.num_feats = num_feats

        # fc layers
        if not fc_dims:
            fc_dims = [32, 32]
        self.fc_dims = fc_dims
        self.num_fields = num_fields
        self.fc_layers = MLP(emb_dim * num_fields, fc_dims, dropout, batch_norm)

        # fm model as the pre-trained embedding layer
        self.fm = FM(emb_dim, num_feats, out_type)

        # output
        self.output_layer = OutputLayer(fc_dims[-1], out_type)

    def forward(self, feat_index, feat_value):
        if not self.fm_trained:
            y = self.fm(feat_index, feat_value)
        else:
            emb_layer = self.fm.get_embedding()
            feat_emb = emb_layer(feat_index)

            # reshape the embedding matrix to a vector
            fc_in = feat_emb.reshape(-1, self.emb_dim * self.num_fields)

            y = self.mlp(fc_in)

            # compute output
            y = self.output_layer(y)
        return y

    def train_fm_embedding(self):
        self.fm_trained = True

    def train_fnn(self):
        self.fm_trained = False
