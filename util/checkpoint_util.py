import torch
import torch.nn as nn
import os
from util.filedir_util import get_file_dir_and_name


def load_to_eval(model: nn.Module, optimizer, path):
    load_checkpoint(model, optimizer, path)
    model.eval()


def load_to_train(model: nn.Module, optimizer, path):
    epoch, loss = load_checkpoint(model, optimizer, path)
    model.eval()
    return epoch, loss


def load_checkpoint(model: nn.Module, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def save_checkpoint(ckpt_path, model, optimizer, loss, epoch):
    ckpt_dir, ckpt_file = get_file_dir_and_name(ckpt_path)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, ckpt_path)
    return ckpt_path


def save_trained_model(model_path, model):
    model_dir, model_file = get_file_dir_and_name(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(model, model_path)
    return model_path


def load_trained_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model
