import torch
import torch.nn as nn
from model.basic.mlp import MLP
from model.basic.output_layer import OutputLayer
from model.basic.functional import build_cross


class PNN(nn.Module):

    def __init__(self, emb_dim, feat_dim, num_fields, fc_dims=None, dropout=None, batch_norm=None,
                 product_type='inner', out_type='binary'):
        super(PNN, self).__init__()
        # embedding layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.num_fields = num_fields
        self.emb_layer = nn.Embedding(num_embeddings=self.feat_dim,
                                      embedding_dim=self.emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        # linear signal layer, named l_z
        if not fc_dims:
            fc_dims = [32, 32]
        self.d1 = d1 = fc_dims[0]
        self.product_type = product_type
        if product_type == '*':
            d1 *= 2
        self.linear_signal_weights = nn.Linear(in_features=num_fields * emb_dim, out_features=d1)
        nn.init.xavier_uniform_(self.linear_signal_weights.weight)

        # product layer, named l_p
        if product_type == 'inner':
            self.product_layer = InnerProductLayer(num_fields, d1)
        elif product_type == 'outer':
            self.product_layer = OuterProductLayer(emb_dim, num_fields, d1)
        else:
            self.product_layer = HybridProductLayer(emb_dim, num_fields, d1)

        # fc layers
        # l_1=relu(l_z+l_p_b_1)
        self.l1_layer = nn.ReLU()
        self.l1_bias = nn.Parameter(torch.randn(d1))
        # l_2 to l_n
        self.fc_dims = fc_dims
        self.fc_layers = MLP(d1, self.fc_dims, dropout, batch_norm)

        # output layer
        self.output_layer = OutputLayer(fc_dims[-1], out_type)

    def forward(self, feat_index):
        # feat_index: N * num_fields
        feat_emb = self.emb_layer(feat_index)  # N * num_fields * emb_dim

        # compute linear signal l_z
        concat_z = feat_emb.reshape(-1, self.emb_dim * self.num_fields)
        linear_signal = self.linear_signal_weights(concat_z)

        # product_layer
        product_out = self.product_layer(feat_emb)

        # fc layers from l_2 to l_n
        # l_1=relu(l_z+l_p_b_1)
        l1_in = torch.add(linear_signal, self.l1_bias)
        l1_in = torch.add(l1_in, product_out)
        l1_out = self.l1_layer(l1_in)
        y = self.fc_layers(l1_out)
        y = self.output_layer(y)
        return y


class InnerProductLayer(nn.Module):
    def __init__(self, num_fields, d1):
        super(InnerProductLayer, self).__init__()
        self.num_fields = num_fields
        self.d1 = d1
        self.num_pairs = int(num_fields * (num_fields - 1) / 2)
        # theta_i^n
        self.product_layer_weights = nn.Linear(in_features=self.num_pairs, out_features=d1)
        nn.init.xavier_uniform_(self.product_layer_weights.weight)

    def forward(self, feat_emb):
        # feat_emb: N * num_fields * emb_dim

        # p_ij=<f_i,f_j>
        # p is symmetric matrix, so only upper triangular matrix needs calculation (without diagonal)
        p, q = build_cross(self.num_fields, feat_emb)
        pij = p * q  # N * num_pairs * emb_dim
        pij = torch.sum(pij, dim=2)  # N * num_pairs

        # l_p
        lp = self.product_layer_weights(pij)
        return lp


class OuterProductLayer(nn.Module):
    def __init__(self, emb_dim, num_fields, d1, kernel_type='mat'):
        super(OuterProductLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_fields = num_fields
        self.d1 = d1
        self.num_pairs = num_fields * (num_fields - 1) / 2
        self.kernel_type = kernel_type
        if kernel_type == 'vec':
            kernel_shape = (self.num_pairs, emb_dim)
        elif kernel_type == 'num':
            kernel_shape = (self.num_pairs, 1)
        else:  # by default mat
            kernel_shape = (emb_dim, self.num_pairs, emb_dim)
        self.kernel_shape = kernel_shape
        self.kernel = nn.Parameter(torch.zeros(kernel_shape))
        nn.init.xavier_uniform_(self.kernel.data)
        self.num_pairs = num_fields * (num_fields - 1) / 2
        self.product_layer_weights = nn.Linear(in_features=num_fields, out_features=d1)
        nn.init.xavier_uniform_(self.product_layer_weights.weight)

    def forward(self, feat_emb):
        p, q = build_cross(self.num_fields, feat_emb)  # p, q: N * num_pairs * emb_dim

        if self.kernel_type == 'mat':
            # self.kernel: emb_dim * num_pairs * emb_dim
            p = p.unsqueeze(1)  # N * 1 * num_pairs * emb_dim
            p = p * self.kernel  # N * emb_dim * num_pairs * emb_dim
            kp = torch.sum(p, dim=-1)  # N * emb_dim * num_pairs
            kp = kp.permute(0, 2, 1)  # N * num_pairs * emb_dim
            pij = torch.sum(kp * q, -1)  # N * num_pairs
        else:
            # self.kernel: num_pairs * emb_dim/1
            kernel = self.kernel.unsqueeze(1)  # 1 * num_pairs * emb_dim/1
            pij = p * q  # N * num_pairs * emb_dim
            pij = pij * kernel  # N * num_pairs * emb_dim
            pij = torch.sum(pij, -1)  # N * num_pairs

        # l_p
        lp = self.product_layer_weights(pij)
        return lp


class HybridProductLayer(nn.Module):
    def __init__(self, emb_dim, num_fields, d1):
        super(HybridProductLayer, self).__init__()
        self.num_fields = num_fields
        self.d1 = d1 / 2
        self.inner_product_layer = InnerProductLayer(num_fields, d1)
        self.outer_product_layer = OuterProductLayer(emb_dim, num_fields, d1)

    def forward(self, feat_emb):
        inner_product_out = self.inner_product_layer(feat_emb)
        outer_product_out = self.outer_product_layer(feat_emb)
        lp = torch.cat([inner_product_out, outer_product_out], dim=1)
        return lp



