from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset

from util.checkpoint_util import *
from util.log_util import *


def train_model_hold_out(job_name, device,
                         model: nn.Module, dataset: Dataset,
                         loss_func, optimizer, val_size=0.2,
                         batch_size=32, epochs=2, shuffle=True,
                         write_log_file=True, log_path=None,
                         save_ckpt=True, ckpt_dir=None, ckpt_interval=None,
                         save_model=True, model_path=None,
                         write_tb=False, tb_dir=None,
                         load_ckpt=None):
    train_set, val_set = split_dataset(dataset, val_size)
    train_config_model(job_name, device, model, train_set, loss_func, optimizer, val_set, batch_size, epochs,
                       shuffle, write_log_file, log_path, save_ckpt, ckpt_dir, ckpt_interval, save_model, model_path,
                       write_tb, tb_dir, load_ckpt)


def train_config_model(job_name, device,
                       model: nn.Module, train_set: Dataset,
                       loss_func, optimizer, val_set: Dataset = None,
                       batch_size=32, epochs=2, shuffle=True,
                       write_log_file=True, log_path=None,
                       save_ckpt=True, ckpt_dir=None, ckpt_interval=None,
                       save_model=True, model_path=None,
                       write_tb=False, tb_dir=None,
                       load_ckpt=None):
    # before training
    # config logger, tensorboard writer, checkpoint file directory and trained model saving path
    logger, writer, ckpt_dir, log_path, model_path = config_path(job_name, device, write_log_file, log_path,
                                                                 save_ckpt, ckpt_dir, save_model, model_path,
                                                                 write_tb, tb_dir)
    # config training meta
    if not save_ckpt:
        ckpt_interval = epochs + 1
    first_epoch = 1
    # if load_ckpt is not None, load the checkpoint and continue training
    if load_ckpt:
        curr_epoch, curr_loss = load_to_train(model, optimizer, load_ckpt)
        write_info_log(logger, 'model loaded from {}'.format(load_ckpt))
        write_info_log(logger, 'epochs trained:{}, current loss:{:.5f}'.format(curr_epoch, curr_loss))
        first_epoch = curr_epoch + 1
        model.to(device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    if val_set:
        val_loader = DataLoader(val_set, batch_size=batch_size)
    else:
        val_loader = None

    # write training meta log
    write_model_meta(logger, job_name, device, model, loss_func, optimizer, epochs, batch_size, shuffle)

    train_model(model, train_loader, loss_func, optimizer, val_loader, epochs, logger, writer,
                ckpt_dir, ckpt_interval, model_path, first_epoch)


def train_model(model: nn.Module, train_loader,
                loss_func, optimizer, val_loader=None, epochs=2,
                logger=None, writer=None,
                ckpt_dir=None, ckpt_interval=None,
                model_path=None, first_epoch=1):
    # start training
    write_info_log(logger, 'training started')
    train_epochs(model, epochs, train_loader, loss_func, optimizer, ckpt_interval, logger, val_loader=val_loader,
                 tensorboard_writer=writer, ckpt_dir=ckpt_dir, first_epoch_idx=first_epoch)

    # finish training
    if model_path:
        saved_model_path = save_trained_model(model_path, model)
        write_info_log(logger, 'model saved:{}'.format(saved_model_path))
    if writer:
        writer.flush()
        writer.close()
    write_info_log(logger, 'training_finished')
    close_logger(logger)


def config_path(job_name, device, write_log_file=True, log_path=None,
                save_ckpt=True, ckpt_dir=None,
                save_model=True, model_path=None,
                write_tb=True, tb_dir=None):
    # config job directory
    job_timestamp = time.time()
    job_timestamp_str = time.strftime("%b%d-%H-%M-%S", time.localtime(job_timestamp))
    device_str = '{}-{}'.format(device.type, device.index)

    job_dir = '/job/{}_{}_{}'.format(job_name, device_str, job_timestamp_str)
    job_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + job_dir
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    # config checkpoint path
    if save_ckpt:
        if not ckpt_dir:
            ckpt_dir = job_dir
    else:
        ckpt_dir = None

    # config log path and load logger
    logger_name = '{}_{}'.format(job_name, device_str)
    if write_log_file:
        if not log_path:
            log_path = job_dir + '/train_log.log'
        logger = create_file_console_logger(log_path, name=logger_name)
    else:
        logger = create_console_logger(name=logger_name)
        log_path = None

    # config model saving path
    if save_model:
        if not model_path:
            model_path = job_dir + '/model.pt'
    else:
        model_path = None

    # config tensorboard
    if write_tb:
        if not tb_dir:
            tb_dir = job_dir + '/tb'
        writer = SummaryWriter(log_dir=tb_dir)
    else:
        writer = None
        tb_dir = None

    write_training_file_meta(logger, ckpt_dir, log_path, model_path, tb_dir)
    return logger, writer, ckpt_dir, log_path, model_path


def train_epochs(model, epochs, train_loader, loss_func, optimizer, ckpt_interval,
                 logger, val_loader=None, tensorboard_writer=None, ckpt_dir=None,
                 first_epoch_idx=1):
    # config checkpoint interval epoch
    if not ckpt_interval:
        if epochs > 50:
            ckpt_interval = 10
        elif epochs > 10:
            ckpt_interval = 5
        else:
            ckpt_interval = 2

    save_ckpt_flag = ckpt_interval
    loss = 0
    max_epoch = epochs + first_epoch_idx - 1
    for epoch in range(first_epoch_idx, max_epoch + 1):
        # train the epoch
        model.train()
        for step, tensors in enumerate(train_loader):
            y = tensors[-1]
            X = tensors[:-1]
            pred_y = model(*X)
            loss = loss_func(pred_y, y)

            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train", loss, epoch)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate average loss during the epoch and write log
        write_training_log(logger, epoch, max_epoch, loss)

        # run validation set
        if val_loader:
            model.eval()
            acc_epoch_loss = 0
            batch_nums = 0
            for step, tensors in enumerate(val_loader):
                y = tensors[-1]
                X = tensors[:-1]
                pred_y = model(*X)
                loss = loss_func(pred_y, y)

                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/validation", loss, epoch)

                acc_epoch_loss += loss
                batch_nums += 1
            avg_epoch_loss = acc_epoch_loss / batch_nums
            write_training_log(logger, epoch, max_epoch, avg_epoch_loss, loss_type='validation')

        # save model to checkpoint file
        save_ckpt_flag -= 1
        if save_ckpt_flag == 0:
            saved_ckpt_path = save_checkpoint('{}/epoch_{}.pt'.format(ckpt_dir, epoch), model, optimizer, loss, epoch)
            write_info_log(logger, 'checkpoint saved:{}'.format(saved_ckpt_path))
            save_ckpt_flag = ckpt_interval


def split_dataset(dataset, val_size):
    data_size = len(dataset)
    if 0 < val_size < 1:
        val_size = int(data_size * val_size)
    train_size = data_size - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_set, val_set
