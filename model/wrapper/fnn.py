import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from feature import FeatureMeta
from model.wrapper import BaseModel
from model.ctr import FNN
from util.train import train_model, split_dataset


class FNNModel(BaseModel):
    def __init__(self, feat_meta: FeatureMeta, emb_dim, fc_dims=None, dropout=None, batch_norm=None, out_type='binary',
                 device_name='cuda:0'):
        super(FNNModel, self).__init__()
        self.model = FNN(emb_dim, feat_meta.get_num_feats(), feat_meta.get_num_fields(), fc_dims, dropout,
                         batch_norm, out_type, True)
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.job_name = 'FNN-' + out_type

    def train(self, feat_index, y, batch_size=32, epochs=2, shuffle=True, val_size=0.2):
        """ train the FNN model with hold-out model selection method

        :param feat_index: ndarray-like, should be shape of (n_samples,num_fields)
        :param y: ndarray-like, should be shape of (n_samples,1) for binary cls/regression
               or (n_samples,num_classes) for multi-class cls
        :param batch_size: the size of samples in a batch
        :param epochs: the epochs to train
        :param shuffle: whether the order of samples is shuffled before each epoch, default true
        :param val_size: whether to use validation set in the training, default 0.2;
               set to >=1 means n_samples; 0 to 1 (0 and 1 not included) means ratio;
               0 means not to use validation set
        """
        self.model.train()
        feat_index_tensor = torch.LongTensor(feat_index).to(self.device)
        y_tensor = torch.Tensor(y).to(self.device)

        dataset = TensorDataset(feat_index_tensor, y_tensor)

        if val_size == 0:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            val_loader = None
        else:
            train_set, val_set = split_dataset(dataset, val_size)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
            val_loader = DataLoader(val_set, batch_size=batch_size)

        loss_func = nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)

        # train FM
        self.model.train_fm_embedding()
        train_model(self.model, train_loader, loss_func, optimizer, val_loader, epochs)

        # train FNN
        self.model.train_fm_embedding()
        train_model(self.model, train_loader, loss_func, optimizer, val_loader, epochs,
                    self.logger, self.tb_writer, self.ckpt_dir, self.ckpt_interval, self.model_path)

    def eval(self, feat_index):
        self.model.eval()
        return self.model(feat_index)
