import torch
from torch.utils.data import DataLoader

from util.log_util import create_file_console_logger
from util.train import config_path, split_dataset, train_model
from torch.utils.tensorboard import SummaryWriter


class BaseModel:
    def __init__(self):
        self.loader_args = None
        self.model = None
        self.job_name = ''
        self.device = torch.device('cpu')
        self.logger = None,
        self.tb_writer = None,
        self.ckpt_dir = None
        self.log_path = None,
        self.model_path = None
        self.ckpt_interval = -1

    def config_training(self, write_log_file=True, log_path=None,
                        save_ckpt=True, ckpt_dir=None, ckpt_interval=None,
                        save_model=True, model_path=None,
                        write_tb=True, tb_dir=None):
        self.logger, self.tb_writer, self.ckpt_dir, self.log_path, self.model_path = \
            config_path(self.job_name, self.device, write_log_file, log_path, save_ckpt, ckpt_dir, save_model,
                        model_path, write_tb, tb_dir)
        if save_ckpt:
            self.ckpt_interval = ckpt_interval

    def config_tensorboard(self, write_tb=False, tb_dir=None):
        if write_tb:
            self.tb_writer = SummaryWriter(log_dir=tb_dir)

    def config_logger(self, write_log_file=False, log_path=None):
        if write_log_file:
            self.logger = create_file_console_logger(log_path, name=self.job_name)

    def config_ckpt(self, save_ckpt=False, ckpt_dir=None, ckpt_interval=None):
        if save_ckpt:
            self.ckpt_dir = ckpt_dir
            self.ckpt_interval = ckpt_interval

    def config_model_saving(self, save_model=False, model_path=None):
        if save_model:
            self.model_path = model_path

    def config_loader_meta(self, **kwargs):
        self.loader_args = kwargs

    def _train(self, dataset, loss_func, optimizer=None, epochs=2, val_size=0):
        if not optimizer:
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=1e-3)
        if val_size <= 0:
            train_loader = DataLoader(dataset, **self.loader_args)
            val_loader = None
        else:
            train_set, val_set = split_dataset(dataset, val_size)
            train_loader = DataLoader(train_set, **self.loader_args)
            val_loader = DataLoader(val_set, batch_size=self.loader_args['batch_size'])
        self.model.train()
        train_model(self.model, train_loader, loss_func, optimizer, val_loader, epochs,
                    self.logger, self.tb_writer, self.ckpt_dir, self.ckpt_interval, self.model_path)

    def train(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError
