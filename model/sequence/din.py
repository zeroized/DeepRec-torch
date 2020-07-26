import torch
import torch.nn as nn
from model.basic import OutputLayer, Dice, MLP


class DIN(nn.Module):
    def __init__(self, u_emb_dim, c_emb_dim, g_emb_dim, fc_dims=None, activation_linear_dim=36, activation='dice',
                 dropout=None, out_type='binary'):
        super(DIN, self).__init__()
        self.activation_unit = ActivationUnit(g_emb_dim, activation_linear_dim, activation)
        if not fc_dims:
            fc_dims = [200, 80]
        self.fc_dims = fc_dims

        if activation == 'dice':
            self.activation = Dice()
        else:
            self.activation = nn.PReLU()
        self.fc_layers = MLP(u_emb_dim + c_emb_dim + 2 * g_emb_dim, fc_dims, dropout, None, self.activation)
        self.output_layer = OutputLayer(fc_dims[-1], out_type)

    def forward(self, history_feats, candidate_feat, user_profile_feat, context_feat):
        # history_feats: N * seq_length * g_emb_dim
        # candidate_feat: N * g_emb_dim
        # user_profile_feat: N * u_emb_dim
        # context_feat: N * c_emb_dim
        histories = torch.split(history_feats, 1, dim=1)  # [N * g_emb_dim] * seq_length
        att_signals = [
            self.activation_unit(history_feat.squeeze(), candidate_feat)  # N * 1
            for history_feat in histories  # N * g_emb_dim
        ]
        att_signal = torch.cat(att_signals, dim=1)  # N * seq_length
        att_signal = att_signal.unsqueeze(dim=2)  # N * seq_length * 1
        weighted = torch.mul(att_signal, history_feats)  # N * seq_length * g_emb_dim
        weighted_pooling = torch.sum(weighted, dim=1)  # N * g_emb_dim
        fc_in = torch.cat([user_profile_feat, weighted_pooling, candidate_feat, context_feat],dim=1)
        fc_out = self.fc_layers(fc_in)
        output = self.output_layer(fc_out)
        return output


class ActivationUnit(nn.Module):
    def __init__(self, g_emb_dim, linear_dim=36, activation='dice'):
        super(ActivationUnit, self).__init__()
        self.g_emb_dim = g_emb_dim
        if activation == 'dice':
            self.activation = Dice()
        else:
            self.activation = nn.PReLU()
        self.linear = nn.Linear(in_features=3 * g_emb_dim, out_features=linear_dim)
        self.out = nn.Linear(in_features=linear_dim, out_features=1)

    def forward(self, history_feat, candidate_feat):
        # history_feat: N * g_emb_dim
        # candidate_feat: N * g_emb_dim

        # There is no definition for "out product" in the activation unit, so here we use K * Q instead as many
        # other implementations do.
        out_product = torch.mul(history_feat, candidate_feat)  # N * g_emb_dim
        linear_in = torch.cat([history_feat, out_product, candidate_feat], dim=1)  # N * (3 * g_emb_dim)
        linear_out = self.linear(linear_in)
        out = self.activation(linear_out)
        return self.out(out)  # N * 1
