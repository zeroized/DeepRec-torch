#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/7/27 9:50
# @Author : Zeroized
# @File : dien.py
# @desc : DIEN implementation

import torch
import torch.nn as nn

"""
Model: DIEN: Deep Interest Evolution Network
Version: arXiv [v4] Thu, 13 Sep 2018 04:37:06 UTC
Reference: Zhou, G., Mou, N., Fan, Y., Pi, Q., Bian, W., Zhou, C., ... & Gai, K. (2018). 
           Deep Interest Evolution Network for Click-Through Rate Prediction. 
           arXiv: Machine Learning.
"""


class DIEN(nn.Module):
    def __init__(self):
        super(DIEN, self).__init__()

    def forward(self, history_feats, candidate_feat, user_feat, context_feat):
        pass
