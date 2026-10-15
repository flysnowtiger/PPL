import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import numpy as np
import torchvision


class AttentionRegularization(nn.Module):
    def __init__(self, num_part=8):
        super(AttentionRegularization, self).__init__()
        self.num_part = num_part

    def forward(self, mask):
        # mask: bt* num_part*h*w
        bt = mask.size(0)
        num_part = mask.size(1)
        mask = mask.view(bt, num_part, -1)
        mask = F.normalize(mask, p=1, dim=2)
        mask = F.normalize(mask, p=2, dim=-1)
        temp = torch.matmul(mask, mask.transpose(2, 1)).view(bt, -1)  # bt*num_part*num_part.to(mask.device)
        I = torch.eye(num_part).unsqueeze(dim=0).expand(bt, num_part, num_part).to(mask.device).reshape(bt, -1)
        orthogonal_loss = (temp - I).pow(2).sum(dim=1).pow(0.5).mean()

        return orthogonal_loss
