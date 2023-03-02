import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class AutoLoss(nn.Module):
    def __init__(self, activate=True, **args):
        super().__init__()
        self.activate=activate
        self.conv = conv = nn.Sequential(*[nn.Linear(2, 64, bias=False),
                                        nn.LayerNorm(64),
                                        nn.ReLU(),
                                        nn.Linear(64, 128, bias=False),
                                        nn.LayerNorm(128)])
    
    def forward(self, pred, gt):
        b = pred.size(0)
        pred = pred.view(b, -1)
        gt = gt.view(b, -1)
        comb = torch.stack([pred, gt], dim=1)
        loss = self.conv(comb)
        return loss
