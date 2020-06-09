import scipy.stats as stats
import numpy as np


def box_cox_transform(x, lmbda=None):
    if not lmbda:
        lmbda = stats.boxcox(x)
    return stats.boxcox(x, lmbda)


def log_transform(x):
    return np.log(x)
