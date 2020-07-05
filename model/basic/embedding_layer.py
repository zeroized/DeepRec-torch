import math
import torch
import torch.nn as nn
from feature import FeatureMeta


class UniformDimensionEmbedding(nn.Module):
    """ An embedding layer using embedding vector with a uniform dimension for all features. Continuous features are
        embedded through 'emb_mul", which is the multiplication of the embedding vector and the value.
    """

    def __init__(self, feat_meta: FeatureMeta, emb_dim):
        super(UniformDimensionEmbedding, self).__init__()
        self.num_feats = feat_meta.get_num_feats()
        self.emb_dim = emb_dim

        self.emb_layer = nn.Embedding(num_embeddings=self.num_feats, embedding_dim=emb_dim)
        nn.init.xavier_uniform_(self.emb_layer.weight)

        cont_fields = [
            feat_meta.continuous_feats[feat_name].start_idx
            for feat_name in feat_meta.continuous_feats
        ]

        self.cont_idx = torch.LongTensor(cont_fields)

    def forward(self, continuous_value, universal_category_index):
        # continuous_value: N * num_cont_fields
        # category_index: N * num_cate_fields
        cont_emb = self.emb_layer(self.cont_idx)  # num_cont_fields * emb_dim
        continuous_value = continuous_value.unsqueeze(dim=2)  # N * num_cont_fields * 1
        cont_emb = torch.mul(continuous_value, cont_emb)  # N * num_cont_fields * emb_dim

        cate_emb = self.emb_layer(universal_category_index)  # N * num_cate_fields * emb_dim
        emb = torch.cat([cont_emb, cate_emb], dim=1)  # N * num_fields * emb_dim
        return emb


class FeatureEmbedding(nn.Module):
    def __init__(self, feat_meta: FeatureMeta, uniform_dim=None, proc_continuous='concat', continuous_emb_dim=None):
        super(FeatureEmbedding, self).__init__()

        if proc_continuous not in ['concat', 'emb_mul']:
            proc_continuous = 'concat'
        self.proc_continuous = proc_continuous

        if not isinstance(uniform_dim, int):
            uniform_dim = 'auto'
        else:
            continuous_emb_dim = uniform_dim

        if proc_continuous == 'emb_mul' and (
                not isinstance(uniform_dim, int)) and (
                not isinstance(continuous_emb_dim, int)):
            raise Exception('No dim designated for embeddings of continuous feature! '
                            'Check param \'uniform_dim\' or \'continuous_emb_dim\'.')

        self.uniform_dim = uniform_dim
        self.continuous_emb_dim = continuous_emb_dim
        self.feat_meta = feat_meta

        self.num_cont_fields = len(feat_meta.continuous_feats.keys())
        continuous_emb_list = [
            ContinuousEmbedding(proc_continuous, continuous_emb_dim)
            for _ in feat_meta.continuous_feats.keys()
        ]
        self.continuous_embeddings = nn.ModuleList(continuous_emb_list)

        self.num_cate_fields = len(feat_meta.categorical_feats.keys())
        categorical_feat_dict = feat_meta.categorical_feats
        categorical_emb_list = [
            CategoricalEmbedding(categorical_feat_dict[feat_name].dim)
            for feat_name in categorical_feat_dict.keys()
        ]
        self.categorical_embeddings = nn.ModuleList(categorical_emb_list)

    def forward(self, continuous_value, category_index):
        continuous_value = torch.split(continuous_value, 1, 1)
        cont_embs = [
            self.continuous_embeddings[i](continuous_value[i])  # N * 1/emb_dim
            for i in range(len(continuous_value))
        ]
        cont_emb = torch.cat(cont_embs, dim=1)

        category_index = torch.split(category_index, 1, 1)
        cate_embs = [
            self.categorical_embeddings[i](category_index[i])  # N * emb_dim
            for i in range(len(category_index))
        ]
        cate_emb = torch.cat(cate_embs, dim=1)

        embedding = torch.cat([cont_emb, cate_emb], dim=1)
        return embedding


class ContinuousEmbedding(nn.Module):
    def __init__(self, proc='concat', dim=1):
        super(ContinuousEmbedding, self).__init__()
        if proc not in ['concat', 'emb_mul']:
            proc = 'concat'
        self.proc = proc
        self.dim = dim
        if proc == 'emb_mul':
            self.emb_layer = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        if self.proc == 'concat':
            return x  # N * 1
        else:  # emb_mul
            return self.emb_layer * x  # N * emb_dim


class CategoricalEmbedding(nn.Module):
    def __init__(self, num_classes, emb_dim='auto'):
        super(CategoricalEmbedding, self).__init__()

        if emb_dim == 'auto' or not isinstance(emb_dim, int):
            emb_dim = get_auto_embedding_dim(num_classes)

        self.emb_dim = emb_dim
        self.num_classes = num_classes

        self.emb_layer = nn.Embedding(num_embeddings=num_classes, embedding_dim=emb_dim)

    def forward(self, x):
        return self.emb_layer(x)  # N * emb_dim


def get_auto_embedding_dim(num_classes):
    """ Calculate the dim of embedding vector according to number of classes in the category
    emb_dim = [6 * (num_classes)^(1/4)]

    ref: Ruoxi Wang, Bin Fu, Gang Fu, and Mingliang Wang. 2017. Deep & Cross Network for Ad Click Predictions.
    In Proceedings of the ADKDD’17 (ADKDD’17). Association for Computing Machinery, New York, NY, USA, Article 12, 1–7.
    DOI:https://doi.org/10.1145/3124749.3124754


    :param num_classes: number of classes in the category
    :return: the dim of embedding vector
    """
    return math.floor(6 * math.pow(num_classes, 0.26))
