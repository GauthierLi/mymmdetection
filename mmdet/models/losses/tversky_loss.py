import torch
import torch.nn as nn

from ..builder import LOSSES

eps=1e-5

def tversky(pred, 
            target, 
            alpha=0.7):
    input = pred.flatten(1)
    target = target.flatten(1).float()

    tp = torch.sum(input * target, 1)
    fn = torch.sum((1-input) * target, 1)
    fp = torch.sum(input * (1-target), 1)

    return (tp + eps) / (tp + alpha * fn + (1 - alpha) * fp + eps)

def tversky_loss(pred, target, alpha=0.7):
    return 1 - tversky(pred, target, alpha)

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.75):
    pt = tversky_loss(pred, target, alpha)
    return torch.pow(pt, beta)

@LOSSES.register_module()
class FocalTverskyLoss(nn.Module):

    def __init__(self, alpha=0.7, beta=0.75, loss_weight=1.0, activate=True):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=True)
        self.beta = torch.tensor(beta, dtype=torch.float32, requires_grad=True)
        self.alpha = nn.Parameter(self.alpha)
        self.beta = nn.Parameter(self.beta)
        self.loss_weight = loss_weight
        self.activate = activate
    
    def forward(self, pred, target, avg_factor=None):
        if self.activate:
            pred = pred.sigmoid()
        
        loss = self.loss_weight * focal_tversky_loss(pred, 
                                                     target, 
                                                     torch.sigmoid(self.alpha), 
                                                     self.beta)
        return loss