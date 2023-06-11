# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weight_reduce_loss


def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
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

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                boundary=False,
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

        if boundary:
            thr = 0.8
            pred_boundary = generate_boundary(pred) > thr
            target_boundary = generate_boundary(target) > thr
            boundary_mask_area = pred_boundary + target_boundary
            pred = torch.masked_select(pred, boundary_mask_area).unsqueeze(dim=0)
            target = torch.masked_select(target, boundary_mask_area).unsqueeze(dim=0)
            # from mmdet.utils import show_feature
            # show_feature(pred_boundary.to(torch.float), "pred", mode='all', save_dir='/home/gauthierli/code/mmdetection/workdir/boundary_dice_v1/boundary')
            # show_feature(target_boundary.to(torch.float), "target", mode='all', save_dir='/home/gauthierli/code/mmdetection/workdir/boundary_dice_v1/boundary')
            # show_feature(boundary_mask_area.to(torch.float), "total", mode='all', save_dir='/home/gauthierli/code/mmdetection/workdir/boundary_dice_v1/boundary')
            # import pdb; pdb.set_trace()

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)

        return loss

def generate_boundary(mask, kernel_size=11):
  assert (kernel_size - 1) % 2 == 0, "kernel size must be even number !"
  max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2))
  tensor_dilate = max_pool(mask.detach())
  tensor_erode = -max_pool(-mask.detach())
  ten_reduce = tensor_dilate - tensor_erode
  return ten_reduce