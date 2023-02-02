import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ..builder import LOSSES
from .utils import weight_reduce_loss

def gau_dice_loss(pred,
              target,
              weight=None,
              eps=1e-4,
              reduction='mean',
              naive_dice=False,
              avg_factor=None,
              mask2former_enabled=False):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    kld_loss = KLD(pred, target, eps)
    # loca_loss = location_loss(pred, target)
    if not mask2former_enabled:
        input = pred.flatten(1)
        gt = target.flatten(1).float()

        a = torch.sum(input * gt, 1)
        if naive_dice:
            b = torch.sum(input, 1)
            c = torch.sum(gt, 1)
            d = (2 * a + eps) / (b + c + eps)
        else:
            b = torch.sum(input * input, 1) + eps
            c = torch.sum(gt * gt, 1) + eps
            d = (2 * a) / (b + c)
        dice_loss = 1 - d

        loss = dice_loss + kld_loss #+ loca_loss
    else:
        loss = kld_loss #+ loca_loss
    if weight is not None:
        assert weight.ndim == dice_loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

    return loss

def edge_distribution(mask):
    """
    input:
        mask: b,h,w
    retrun:
        h_proj: b,h
        w_proj: b,w
    """
    b,h,w = mask.shape
    h, w = float(h), float(w)
    mask=mask.type(torch.float32)
    h_projection = mask.sum(dim=2) / w
    w_projection = mask.sum(dim=1) / h
    return h_projection, w_projection

def KLD(pred, gt, eps=1e-4):
    """
    input:
        pred: b,h,w
        gt: b,h,w
    return:
        kl: b
    """
    h_proj_pred, w_proj_pred = edge_distribution(pred)
    h_proj_gt, w_proj_gt = edge_distribution(gt)

    # import pdb; pdb.set_trace()
    kl_h = nn.BCELoss()(h_proj_pred, h_proj_gt)
    kl_w = nn.BCELoss()(w_proj_pred, w_proj_gt)
    # kl_h = (h_proj_pred*torch.log((h_proj_pred+eps)/(h_proj_gt+eps))).sum(dim=1,keepdims=True)
    # kl_w = (w_proj_pred*torch.log((w_proj_pred+eps)/(w_proj_gt+eps))).sum(dim=1,keepdims=True)

    return kl_h + kl_w

def gravity(mask):
    b, x_h, y_w = mask.shape
    x_weight = torch.from_numpy(np.array([[i for i in range(x_h)] for j in range(y_w)])).permute(-1,-2)
    y_weight = torch.from_numpy(np.array([[i for i in range(y_w)] for j in range(x_h)])) 
    x_weight, y_weight = [x_weight for i in range(b)],[y_weight for i in range(b)]
    x_weight, y_weight = torch.stack(x_weight, dim=0), torch.stack(y_weight, dim=0)
    x_weight = x_weight.to(mask.device)
    y_weight = y_weight.to(mask.device)

    mask_sum = mask.sum(dim=(-1,-2))
    gravity_x = (mask * x_weight).sum(dim=(-1,-2)) / mask_sum
    gravity_y = (mask * y_weight).sum(dim=(-1,-2)) / mask_sum
    gravity = torch.stack([gravity_x, gravity_y], dim=1)

    return gravity

def location_loss(mask, gt):
    mask_gravity, gt_gravity = gravity(mask), gravity(gt)
    rho = torch.sqrt(torch.pow(mask_gravity - gt_gravity, 2).sum(dim=1))

    total_mask = (mask + gt) / 2.
    h_lenth = (total_mask.sum(dim=2).clamp(max=0.1) * 10).sum(dim=1)
    w_lenth = (total_mask.sum(dim=1).clamp(max=0.1) * 10).sum(dim=1)
    c_ = torch.sqrt(torch.pow(h_lenth, 2) + torch.pow(w_lenth, 2))

    loss = rho / c_
    return loss

@LOSSES.register_module()
class GauDiceLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-4,
                 mask2former_enabled=False):
        super(GauDiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.mask2former_enabled = mask2former_enabled
    
    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * gau_dice_loss(pred,
                                                target,
                                                weight,
                                                eps=self.eps,
                                                reduction=reduction,
                                                naive_dice=self.naive_dice,
                                                avg_factor=avg_factor,
                                                mask2former_enabled=self.mask2former_enabled)
        return loss

if __name__ == '__main__':
    # !generate fake pred and gt
    pred_mask, gt_mask = np.zeros((224, 448)), np.zeros((224, 448))
    pred_mask[30:161, 60:191] = 1
    gt_mask[33:164, 63:194] = 1

    # plt.imshow(pred_mask)
    # plt.show()

    pred_mask = torch.from_numpy(pred_mask)
    pred_mask = pred_mask.type(torch.float32)
    pred_mask.requires_grad = True
    pred_mask.sum()

    gt_mask = torch.from_numpy(gt_mask)
    gt_mask = gt_mask.type(torch.float32)
    gt_mask.requires_grad = False

    # !given pred_mask, gt_mask
    pred_mask, gt_mask = torch.stack([pred_mask,pred_mask], dim=0), torch.stack([gt_mask,gt_mask], dim=0)
    # pred_mask = torch.rand_like(gt_mask)
    print("mask shape:", pred_mask.shape, gt_mask.shape)
    
    # # dice_loss
    # print("dice:", dice_loss(pred_mask, gt_mask))

    # !gevin projection
    h_proj, w_proj = edge_distribution(pred_mask)
    # print("projection:", h_proj, w_proj)

    # !kld loss
    print("kld:", KLD(pred_mask, gt_mask))

    # !gravity 
    print("gravity", gravity(pred_mask))

    # !location 
    print("location:", location_loss(pred_mask, gt_mask))

    # !gau_dice_loss
    gau = GauDiceLoss()
    print("gau_dice:", gau(pred_mask, gt_mask))