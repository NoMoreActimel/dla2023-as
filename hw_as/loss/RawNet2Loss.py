import torch
from torch import nn

class RawNet2Loss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    def forward(self, predict, target, **kwargs):
        return self.loss(predict, target)