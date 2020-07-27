import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor):
        # x: N * num_neurons

        avg = x.mean(dim=1)  # N
        avg = avg.unsqueeze(dim=1)  # N * 1
        var = torch.pow(x - avg, 2) + self.epsilon  # N * num_neurons
        var = var.sum(dim=1).unsqueeze(dim=1)  # N * 1

        ps = (x - avg) / torch.sqrt(var)  # N * 1

        ps = nn.Sigmoid()(ps)  # N * 1
        return ps * x + (1 - ps) * self.alpha * x
