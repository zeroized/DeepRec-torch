import torch.nn as nn

from model.basic.gbdt import GBDT
from model.ctr.lr import LR

"""
Model: GBDT+LR: Gradient Boosting Decision Tree with Logistic Regression
Version: arXiv [v1] Mon, 11 Jan 2016 10:04:40 UTC
Reference: He, X., Pan, J., Jin, O., Xu, T., Liu, B., Xu, T., ... & Candela, J. Q. (2014). 
           Practical Lessons from Predicting Clicks on Ads at Facebook. 
           International workshop on Data Mining for Online Advertising.
"""

class GBDTLR(nn.Module):
    def __init__(self, num_leaves=31, max_depth=-1, n_estimators=100, min_data_in_leaf=20,
                 learning_rate=0.1, objective='binary'):
        super(GBDTLR, self).__init__()
        self.gbdt = GBDT(num_leaves, max_depth, n_estimators, min_data_in_leaf, learning_rate, objective)
        self.gbdt_trained = False
        self.logistic_layer = LR(num_leaves * n_estimators, out_type=objective)

    def forward(self, data):
        pred_y, feat_index = self.gbdt.pred(data)
        y = self.logistic_layer(feat_index)
        return y

    def train_gbdt(self, data, y):
        self.gbdt.train(data, y)
        self.gbdt_trained = True

    def get_gbdt_trained(self):
        return self.gbdt_trained
