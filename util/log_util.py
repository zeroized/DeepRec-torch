import logging
import os
import time
from util.filedir_util import get_file_dir_and_name

LEVEL_DICT = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
DEFAULT_FORMATTER = logging.Formatter(
    # "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    "[%(asctime)s][%(levelname)s] %(message)s"
)


def create_console_logger(formatter=DEFAULT_FORMATTER, verbosity=1, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(LEVEL_DICT[verbosity])

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def create_file_console_logger(log_path, formatter=DEFAULT_FORMATTER, verbosity=1, name=None):
    log_dir, log_file = get_file_dir_and_name(log_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(LEVEL_DICT[verbosity])

    fh = logging.FileHandler(log_path, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def write_training_file_meta(logger, ckpt_path=None, log_path=None, model_path=None, tb_dir=None):
    if ckpt_path:
        logger.info('checkpoint file path:{}'.format(ckpt_path))
    else:
        logger.info('no checkpoint will be saved')

    if log_path:
        logger.info('log file path:{}'.format(log_path))
    else:
        logger.info('only logs on console')

    if model_path:
        logger.info('trained model path:{}'.format(model_path))
    else:
        logger.info('trained model will not be saved')

    if tb_dir:
        logger.info('tensorboard log directory:{}'.format(tb_dir))
    else:
        logger.info('no logs will be shown on tensorboard')


def write_model_meta(logger,
                     job_name, device,
                     model, loss_func, optimizer,
                     epochs, batch_size, shuffle):
    logger.info('job:{},device:{}'.format(job_name, device))
    logger.info('model structure')
    logger.info(model)
    logger.info('loss function:{}'.format(loss_func))
    logger.info('optimizer information')
    logger.info(optimizer)
    logger.info('training meta')
    logger.info('epochs:{},batch size:{},shuffle:{}'.format(epochs, batch_size, shuffle))


def write_training_log(logger, epoch, epochs, loss, loss_type='training'):
    log_info = 'epoch:[{}/{}], {} loss:{:.5f}'.format(epoch, epochs, loss_type, loss)
    logger.info(log_info)


def write_info_log(logger, msg):
    logger.info(msg)


def close_logger(logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
