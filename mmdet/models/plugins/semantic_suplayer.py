import torch
from einops import rearrange
from mmcv.cnn import PLUGIN_LAYERS
from mmcv.runner import BaseModule
from mmdet.utils import upsample_like
# from mmdet.core import multi_apply

@PLUGIN_LAYERS.register_module()
class SemanticSup(BaseModule):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, features, gt_semantics, img_metas):
        b = len(gt_semantics)
        devices = [features[0].device for i in range(b)]

        num_layer = len(features)
        gt_semantic = [self.process_gt(gt_semantics[i], img_metas[i], devices[i]) for i in range(b)]
        gt_semantic_list = [gt_semantic for _ in range(num_layer)]

        loss = self.multi_apply(self.single_semantic_loss, features, gt_semantic_list)
        loss = torch.stack(loss).mean()
        return {"semanticSup": loss}
    
    def process_gt(self, gt_semantic, img_meta, device):
        return gt_semantic.pad(img_meta['pad_shape'][:2], pad_val=0)\
        .to_tensor(dtype=torch.bool, device=device).float()
    
    def single_semantic_loss(self, feature, gt_semantic):
        feature = upsample_like(feature, gt_semantic[0])
        feature = rearrange(feature, 'b c h w -> b c (h w)')
        single_loss = 0.
        for i, gt_sem in enumerate(gt_semantic):
            feat = feature[i]
            gt_sem = rearrange(gt_sem, 'b c h->b (c h)')
            coefficient = torch.softmax((torch.einsum('ce, me->cm', feat.detach(), gt_sem.detach()) + self.eps) \
                                        /(gt_sem.detach().sum(dim=1) + self.eps), 
                                        dim=1)
            loss = self.weight_subdice_loss(feat, gt_sem, coefficient)
            single_loss += loss
        return single_loss

    def multi_apply(self, func, *args):
        n_list = len(args[0])
        result = []
        arg_list = []
        for i in range(n_list):
            tmp_arg = []
            for arg in args:
                tmp_arg.append(arg[i])
            arg_list.append(tmp_arg)
        for arg in arg_list:
            result.append(func(*arg))
        return result

    def weight_subdice_loss(self, feat, gt, weight):
        tp = torch.einsum('ce, ge->cg', feat, gt) + self.eps
        fp = torch.einsum('ce, ge->cg', feat, 1 - gt) + self.eps
        subdice = (tp + self.eps) / (tp + fp)
        return ((1 - subdice) * weight).mean()
