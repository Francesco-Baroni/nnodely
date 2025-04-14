import torch.nn as nn
import torch
from nnodely.support.utils import check

available_losses = ['mse', 'rmse', 'mae', 'cross_entropy']

# class CustomRMSE(nn.Module):
#     def __init__(self):
#         super(CustomRMSE, self).__init__()
#         self.mse = nn.MSELoss()
#
#     def forward(self, inA, inB):
#         #assert predictions.keys() == labels.keys(), "Keys of predictions and labels must match"
#         #loss = torch.sqrt(self.mse(inA, inB))
#         return self.mse(inA, inB)

class CustomLoss(nn.Module):
    def __init__(self, loss_type='mse', **kwargs):
        super(CustomLoss, self).__init__()
        check(loss_type in available_losses, TypeError, f'The \"{loss_type}\" loss is not available. Possible losses are: {available_losses}.')
        self.loss_type = loss_type
        self.loss = nn.MSELoss(**kwargs)
        if callable(loss_type):
            self.loss = loss_type
        elif self.loss_type == 'mae':
            self.loss = nn.L1Loss(**kwargs)
        elif self.loss_type == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss(**kwargs)
        
    def forward(self, inA, inB):
        if self.loss_type == 'cross_entropy':
            inB = inB.squeeze().float() if inA.shape == inB.shape else inB.squeeze().long()
            inA = inA.squeeze()
        res = self.loss(inA,inB)
        if self.loss_type == 'rmse':
            res = torch.sqrt(res)
        return res