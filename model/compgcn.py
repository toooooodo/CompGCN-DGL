import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from model.layer import CompGCNCov


class CompGCN(nn.Module):
    def __init__(self, ):
        super(CompGCN, self).__init__()
        self.act = torch.tanh
        self.loss = nn.BCELoss()

    def calc_loss(self, pred, label):
        return self.loss(pred, label)
