import torch
import torch.nn as nn
from model.basic.mlp import MLP
from model.basic.output_layer import OutputLayer


class NFM(nn.Module):
    def __init__(self, emb_dim, feat_dim, num_fields, fc_dims=None, dropout=None, batch_norm=None, out_type='binary'):
        super(NFM, self).__init__()
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.num_fields = num_fields
        self.emb_layer = nn.Embedding(num_embeddings=feat_dim, embedding_dim=emb_dim)
        self.bi_intaraction_layer = BiInteractionLayer()
        if not fc_dims:
            fc_dims = [32, 32]
        self.fc_dims = fc_dims
        self.fc_layers = MLP(emb_dim, fc_dims, dropout, batch_norm)
        self.output_layer = OutputLayer(in_dim=fc_dims[-1], out_type=out_type)

    def forward(self, feat_index, feat_value):
        feat_emb = self.emb_layer(feat_index)  # N * num_fields * emb_dim
        feat_value = feat_value.unsqueeze(dim=2)  # N * num_fields * 1
        feat_emb_value = torch.mul(feat_emb, feat_value)  # N * num_fields * emb_dim
        bi = self.bi_intaraction_layer(feat_emb_value)  # N * emb_dim

        fc_out = self.fc_layers(bi)
        out = self.output_layer(fc_out)
        return out


class BiInteractionLayer(nn.Module):
    def __init__(self):
        super(BiInteractionLayer, self).__init__()

    def forward(self, feat_emb_value):
        square_of_sum = torch.sum(feat_emb_value, dim=1)  # N * emb_dim
        square_of_sum = torch.mul(square_of_sum, square_of_sum)  # N * emb_dim

        sum_of_square = torch.mul(feat_emb_value, feat_emb_value)  # N * num_fields * emb_dim
        sum_of_square = torch.sum(sum_of_square, dim=1)  # N * emb_dim

        bi_out = square_of_sum - sum_of_square
        bi_out = bi_out / 2
        return bi_out
