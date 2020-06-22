from feature.feature import *


class FeatureMeta:
    def __init__(self):
        super().__init__()
        self.continuous_feats = {}
        self.categorical_feats = {}
        self.multi_category_feats = {}
        self.feat_dict = {}

    def add_continuous_feat(self, name, transformation=None, discretize=False, discretize_bin=10):
        self.delete_feat(name)
        self.continuous_feats[name] = ContinuousFeature(name, transformation, discretize, discretize_bin)
        self.feat_dict[name] = 'continuous'

    def add_categorical_feat(self, name, all_categories=None):
        self.delete_feat(name)
        self.categorical_feats[name] = CategoricalFeature(name, all_categories)
        self.feat_dict[name] = 'categorical'

    def add_multi_category_feat(self, name, all_categories=None):
        self.delete_feat(name)
        self.multi_category_feats[name] = MultiCategoryFeature(name, all_categories)
        self.feat_dict[name] = 'multi_category'

    def delete_feat(self, name):
        if name in self.feat_dict:
            feat_type = self.feat_dict[name]
            if feat_type == 'continuous':
                del self.continuous_feats[name]
            elif feat_type == 'categorical':
                del self.categorical_feats[name]
            elif feat_type == 'multi_category':
                del self.multi_category_feats[name]

    def get_num_feats(self):
        total_dim = 0
        total_dim += len(self.continuous_feats)
        for key in self.categorical_feats:
            feat = self.categorical_feats[key]
            total_dim += feat.dim

        for key in self.multi_category_feats:
            feat = self.multi_category_feats[key]
            total_dim += feat.dim
        return total_dim

    def get_num_fields(self):
        return len(self.feat_dict.keys())

    def __str__(self):
        feats_list = [self.continuous_feats, self.categorical_feats, self.multi_category_feats]
        info_strs = []
        for feats in feats_list:
            info_str = ''
            for key in feats:
                feat = feats[key]
                info_str += str(feat)
                info_str += '\n'
            info_strs.append(info_str)
        return 'Continuous Features:\n{}Categorical Features:\n{}Multi-Category Features:\n{}'.format(*info_strs)
