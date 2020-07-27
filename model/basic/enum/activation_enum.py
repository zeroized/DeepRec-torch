from enum import Enum
import torch.nn as nn


class ActivationEnum(Enum):
    RELU = nn.ReLU()
    SIGMOID = nn.Sigmoid()
    TANH = nn.Tanh()
    PRELU = nn.PReLU()
