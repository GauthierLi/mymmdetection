import os
import sys
import cv2
import torch

import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from .monitors import buffer

def multi_apply(func, *args):
    args_length = len(args)
    para_length = len(args[0])
    results = []
    for i in range(para_length):
        paramters = list()
        for arg in args:
            paramters += [arg[i]]
        result = func(*paramters)
        results.append(result)
    return tuple(results)        
        

class visual_feature:
    def __init__(self, mode="mean",save_dir="featuremap", show=False):
        assert mode in ["mean", "top", "all"]
        self.mode = mode
        self.save_dir = save_dir
        if not (os.path.exists(self.save_dir) and show):
            os.makedirs(self.save_dir)
        self.show = show
    
    def _vis_tensor(self, feature:torch.Tensor)->tuple:
        shape = feature.shape
        shape_length = len(shape)
        assert shape_length == 4 or shape_length == 3 or shape_length == 2, f"Tensor only support dimention 2 or 3, not implental for {shape_length}!"
        if shape_length == 4:
            bs = shape[0]
            assert bs == 1, f"Warning: Shape:{shape}, only support bs 1, unless only show the first batch!"
            feature = feature.squeeze(dim=0)
            feature = feature.cpu().numpy()
            result = self._vis_ndarry(feature)
        else:
            feature = feature.cpu().numpy()
            result = self._vis_ndarry(feature)
        return result
            

    def _vis_ndarray(self, feature:np.ndarray)->tuple:
        shape = feature.shape
        shape_length = len(shape)
        assert shape_length == 3 or shape_length == 2, f"Ndarray only support dimention 2 or 3, not implental for {shape_length}!"
        if shape_length == 3:
            if self.mode == "mean":
                result = [np.mean(feature, axis=0)]
            elif self.mode == "top":
                result = [feature[0]]
            elif self.mode == "all":
                result = [feature[i] for i in range(shape[0])]
        elif shape_length == 2:
            result = [feature]
        return result
    
    def __call__(self, features, spetial_name=None):
        if isinstance(features, torch.Tensor):
            show = self._vis_tensor(features)
        elif isinstance(features, np.ndarray):
            show = self._vis_ndarray(features)
        else:
            assert isinstance(features, list), "Feature should be represent as torch.ndarry, ndarray or list!"
            features = np.array(features)
            show = self._vis_ndarray(features)
        
        N = len(show)
        for i, feature in enumerate(show):
            assert len(feature.shape)==2, "Only show heatmap in 2 dimension."
            if spetial_name is not None:
                fea_name = spetial_name + "_" + f'feature_{i}.jpg'
            else:
                fea_name = f'feature_{i}.jpg'
            fea_name = osp.join(self.save_dir, fea_name)
            if self.show:
                plt.imshow(feature)
                plt.show()
            else:
                plt.imsave(fea_name, feature)
        # print(f"=>Done! file saved at {os.path.join(os.getcwd(), self.save_dir)}")

def show_feature(features, spetial_name=None, mode="mean",save_dir="featuremap", show=False):
    """
    sptial_name: spetial name for feature map
    mode: included mean, all, top
    """
    vis = visual_feature(mode=mode, save_dir=save_dir, show=show)
    if isinstance(features, tuple):
        N = len(features)
        spetial_name_list = [spetial_name+f"{i}^th" for i in range(N)]
        multi_apply(vis, features, spetial_name_list)
    else:
        vis(features=features, spetial_name=spetial_name)

class batch_feature_vuer(buffer):
    def __init__(self, spetial_dir:str, mode='mean'):
        super(batch_feature_vuer, self).__init__()
        self.monitor = visual_feature(save_dir=spetial_dir)
        self.save_dir = spetial_dir
    
    def save(self):
        for key in self.buffer_dict:
            img_name = key + ".jpg"
            self.monitor(features=self.buffer_dict[key], spetial_name=self.save_dir + "/" + img_name)
    