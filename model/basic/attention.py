import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basic.functional import inner_product_attention_signal


class LocationBasedAttention(nn.Module):
    def __init__(self, emb_dim, att_weight_dim):
        super(LocationBasedAttention, self).__init__()
        self.weights = nn.Parameter(torch.zeros(emb_dim, att_weight_dim))
        nn.init.xavier_uniform_(self.weights.data)
        self.bias = nn.Parameter(torch.randn(att_weight_dim))
        self.h = nn.Parameter(torch.randn(att_weight_dim))

    def forward(self, values):
        # values: N * num * emb_dim
        att_signal = torch.matmul(values, self.weights)  # N * num * att_weight_dim
        att_signal = att_signal + self.bias  # N * num * att_weight_dim
        att_signal = F.relu(att_signal)
        att_signal = torch.mul(att_signal, self.h)  # N * num * att_weight_dim
        att_signal = torch.sum(att_signal, dim=2)  # N * num
        att_signal = F.softmax(att_signal, dim=1)  # N * num
        return att_signal


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, num_heads, dim, project_dim):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.num_heads = num_heads
#         self.dim = dim
#         self.project_dim = project_dim
#         # W_query^h
#         self.query_projection_weights = nn.Parameter(torch.zeros(dim, project_dim * num_heads))
#         # emb_dim * (d` * num_heads)
#         nn.init.xavier_uniform_(self.query_projection_weights.data)
#
#         # W_key^h
#         self.key_projection_weights = nn.Parameter(torch.zeros(dim, project_dim * num_heads))
#         # emb_dim * (d` * num_heads)
#         nn.init.xavier_uniform_(self.key_projection_weights.data)
#
#         # W_value^h
#         self.value_projection_weights = nn.Parameter(torch.zeros(dim, project_dim * num_heads))
#         # emb_dim * (d` * num_heads)
#         nn.init.xavier_uniform_(self.value_projection_weights.data)
#
#     def forward(self, feat_emb):
#         # feat_emb: N * num_feats * emb_dim
#         # (N * num_feats * emb_dim) * (emb_dim * (d` * num_heads)) = (N * num_feats * (d` * num_heads))
#         queries = torch.matmul(feat_emb, self.query_projection_weights)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.dim = dim
        self.attention_signal = AttentionSignal(dim, 'inner-product', True)

    def forward(self, query, keys, values):
        # query: N * emb_dim
        # keys: N * num_keys * emb_dim
        # values: N * num_keys * emb_dim
        att_signal = self.attention_signal(query, keys)  # N * num_keys
        att_signal = att_signal.unsqueeze(dim=2)  # N * num_keys * 1
        weighted_values = torch.mul(att_signal, values)
        return weighted_values  # N * num_keys * emb_dim


class AttentionSignal(nn.Module):
    def __init__(self, dim, similarity='inner-product', scale=False, activation='relu'):
        super(AttentionSignal, self).__init__()
        self.dim = dim
        self.similarity = similarity
        self.scale = scale
        self.activation = activation
        if similarity == 'inner-product':  # a_i = query^T * keys_i
            pass

        elif self.similarity == 'concat':  # a_i = v^T * ReLU(W_q * query + W_k * keys_i)
            # v
            self.v_a = nn.Parameter(torch.zeros(dim))
            nn.init.xavier_uniform_(self.v_a.data)
            # W_q
            self.weights_q = nn.Parameter(torch.zeros((dim, dim)))
            nn.init.xavier_uniform_(self.weights_q.data)
            # W_k
            self.weights_k = nn.Parameter(torch.zeros((dim, dim)))
            nn.init.xavier_uniform_(self.weights_k.data)

        else:  # general, a_i = query^T * W * keys_i
            self.weights_a = nn.Parameter(torch.zeros((dim, dim)))
            nn.init.xavier_uniform_(self.weights_a.data)

    def forward(self, query, keys):
        # query: N * emb_dim
        # keys: N * num_keys * emb_dim

        if self.similarity == 'inner-product':
            att = inner_product_attention_signal(query, keys, 'softmax')

        elif self.similarity == 'concat':
            query = query.unsqueeze(dim=1)  # N * 1 * emb_dim
            weighted_q = torch.matmul(query, self.weights_q)  # N * 1 * emb_dim
            weighted_k = torch.matmul(keys, self.weights_k)  # N * num_keys * emb_dim
            weighted_kq = torch.add(weighted_q, weighted_k)  # N * num_keys * emb_dim
            if not self.activation:
                pass
            elif self.activation == 'relu':
                weighted_kq = F.relu(weighted_kq)
            elif self.activation == 'tanh':
                weighted_kq = F.tanh(weighted_kq)
            elif self.activation == 'sigmoid':
                weighted_kq = F.sigmoid(weighted_kq)
            att = torch.mul(weighted_kq, self.v_a)  # N * num_keys * emb_dim
            att = torch.sum(att, dim=2)  # N * num_keys

        else:
            query = query.unsqueeze(dim=1)  # N * 1 * emb_dim
            qw = torch.matmul(query, self.weights_a)  # (N * 1 * emb_dim) * (emb_dim * emb_dim) = N * 1 * emb_dim
            qw = qw.transpose(1, 2)  # N * emb_dim * 1
            att = torch.bmm(keys, qw)  # (N * num_keys * emb_dim) * (N * emb_dim * 1) = N * num_keys * 1
            att = att.squeeze()  # N * num_keys
        if self.scale:
            att = att / torch.sqrt(self.dim)
        return F.softmax(att)
