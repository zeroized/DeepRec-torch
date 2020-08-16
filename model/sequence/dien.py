#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/7/27 9:50
# @Author : Zeroized
# @File : dien.py
# @desc : DIEN implementation

import torch
import torch.nn as nn

from model.basic import MLP, OutputLayer, Dice
from model.basic.attention import AttentionSignal

"""
Model: DIEN: Deep Interest Evolution Network
Version: arXiv [v4] Thu, 13 Sep 2018 04:37:06 UTC
Reference: Zhou, G., Mou, N., Fan, Y., Pi, Q., Bian, W., Zhou, C., ... & Gai, K. (2018). 
           Deep Interest Evolution Network for Click-Through Rate Prediction. 
           arXiv: Machine Learning.
"""


class DIEN(nn.Module):
    def __init__(self, u_emb_dim, c_emb_dim, g_emb_dim, fc_dims=None, ext_hidden_dim=32, evo_hidden_dim=32,
                 activation_linear_dim=36,
                 activation='dice',
                 dropout=None, out_type='binary'):
        super(DIEN, self).__init__()

        self.extractor_layer = nn.GRU(g_emb_dim, ext_hidden_dim, 1, batch_first=True)

        self.attention_layer = AttentionSignal(query_dim=g_emb_dim, key_dim=ext_hidden_dim, similarity='general')

        self.evolution_layer = EvolutionLayer(ext_hidden_dim, evo_hidden_dim)

        if not fc_dims:
            fc_dims = [200, 80]
        self.fc_dims = fc_dims

        if activation == 'dice':
            self.activation = Dice()
        else:
            self.activation = nn.PReLU()
        self.fc_layers = MLP(u_emb_dim + c_emb_dim + g_emb_dim + evo_hidden_dim,
                             fc_dims, dropout, None, self.activation)
        self.output_layer = OutputLayer(fc_dims[-1], out_type)

    def forward(self, history_feats, candidate_feat, user_profile_feat, context_feat):
        # history_feats: N * seq_length * g_emb_dim
        # candidate_feat: N * g_emb_dim
        # user_profile_feat: N * u_emb_dim
        # context_feat: N * c_emb_dim

        extracted_interest, _ = self.extractor_layer(history_feats)  # [batch_size * ext_hidden_dim] * seq_length
        # extracted_interest = torch.stack(extracted_interest,dim=1)
        att_signal = self.attention_layer(candidate_feat, extracted_interest)  # batch_size * seq_length

        evolved_interest = self.evolution_layer(extracted_interest, att_signal)  # batch_size *evo_hidden_dim

        fc_in = torch.cat([evolved_interest, candidate_feat, user_profile_feat, context_feat], dim=1)
        fc_out = self.fc_layers(fc_in)
        output = self.output_layer(fc_out)
        return output


class EvolutionLayer(nn.Module):
    def __init__(self, input_dim, cell_hidden_dim):
        super(EvolutionLayer, self).__init__()
        self.cell_hidden_dim = cell_hidden_dim
        self.cell = AUGRUCell(input_dim, cell_hidden_dim)

    def forward(self, extracted_interests: torch.Tensor, att_signal: torch.Tensor, hx=None):
        # extracted_interests: batch_size * seq_length * gru_hidden_dim
        # att_signal: batch_size * seq_length
        interests = extracted_interests.split(split_size=1, dim=1)  # [batch_size * 1 * gru_hidden_dim] * seq_length
        att_signals = att_signal.split(split_size=1, dim=1)  # [batch_size * 1] * seq_length

        if hx is None:
            hx = torch.zeros((extracted_interests.size(0),extracted_interests.size(2)),
                             dtype=extracted_interests.dtype,
                             device=extracted_interests.device)

        for interest, att in zip(interests, att_signals):
            hx = self.cell(interest.squeeze(), att, hx)
        return hx  # batch_size * cell_hidden_dim


class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AUGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_weights = nn.Linear(in_features=input_dim,
                                       out_features=3 * hidden_dim)  # concat[Wu,Wr,Wh]
        nn.init.xavier_uniform_(self.input_weights.weight)
        self.hidden_weights = nn.Linear(in_features=hidden_dim,
                                        out_features=3 * hidden_dim,
                                        bias=False)  # concat[Uu,Ur,Uh]
        nn.init.xavier_uniform_(self.hidden_weights.weight)

    def forward(self, ix: torch.Tensor, att_signal: torch.Tensor, hx: torch.Tensor = None):
        # ix: batch_size * input_dim
        # att_signal: batch_size * 1
        # hx: batch_size * hidden_dim
        if hx is None:
            hx = torch.zeros(self.hidden_dim, dtype=ix.dtype, device=ix.device)
        weighted_inputs = self.input_weights(ix)  # [batch_size * hidden_dim] * 3
        weighted_inputs = weighted_inputs.chunk(chunks=3, dim=1)
        weighted_hiddens = self.hidden_weights(hx)
        weighted_hiddens = weighted_hiddens.chunk(chunks=3, dim=1)  # [1 * hidden_dim] * 3

        update_gate = weighted_inputs[0] + weighted_hiddens[0]  # batch_size * hidden_dim
        update_gate = torch.sigmoid(update_gate)
        update_gate = torch.mul(att_signal, update_gate)

        reset_gate = weighted_hiddens[1] + weighted_hiddens[1]  # batch_size * hidden_dim
        reset_gate = torch.sigmoid(reset_gate)

        hat_hidden = torch.mul(weighted_hiddens[2], reset_gate)
        hat_hidden = hat_hidden + weighted_inputs[2]
        hat_hidden = torch.tanh(hat_hidden)  # batch_size * hidden_dim

        hidden = torch.mul((1 - update_gate), hx) + torch.mul(update_gate, hat_hidden)
        return hidden
