# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage_instance_seg import SingleStageInstanceSegmentor


@DETECTORS.register_module()
class SOLOv2(SingleStageInstanceSegmentor):
    """`SOLOv2: Dynamic and Fast Instance Segmentation
    <https://arxiv.org/abs/2003.10152>`_

    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 mask_head=None,
                 semsup_on=False,
                 semantic_sup=dict(type='SemanticSup'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            semsup_on=semsup_on,
            semantic_sup=semantic_sup,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
