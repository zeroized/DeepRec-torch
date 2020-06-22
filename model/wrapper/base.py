import torch
from util.train import config_path


class BaseModel:
    def __init__(self):
        self.model = None
        self.job_name = ''
        self.device = torch.device('cpu')
        self.logger = None,
        self.tb_writer = None,
        self.ckpt_dir = None
        self.log_path = None,
        self.model_path = None
        self.ckpt_interval = -1

    def config_training_logs(self, write_log_file=True, log_path=None,
                             save_ckpt=True, ckpt_dir=None, ckpt_interval=None,
                             save_model=True, model_path=None,
                             write_tb=True, tb_dir=None):
        self.logger, self.tb_writer, self.ckpt_dir, self.log_path, self.model_path = \
            config_path(self.job_name, self.device, write_log_file, log_path, save_ckpt, ckpt_dir, save_model,
                        model_path, write_tb, tb_dir)
        if save_ckpt:
            self.ckpt_interval = ckpt_interval

    def train(self, **kwargs):
        raise NotImplementedError

    def eval(self, **kwargs):
        raise NotImplementedError
