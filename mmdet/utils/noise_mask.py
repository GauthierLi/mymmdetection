import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def shape_noise_generate(shape, stages, rate):
    """
    Args:
        shape (tuple): shape of noise
        stages (int): prepare multi-stage nosie for step by step denoise
        rate (Tensor): the rate of greatest noise

    Returns:
        List(Tensor): noise of each stage, nosie shape N,H,W
    """
    if isinstance(rate, float):
        rate = torch.tensor([rate])
        rate = rate.repeat(shape)
    noise = torch.rand(shape)
    thr_list = [rate * (i+1) / float(stages) for i in range(stages)]
    noise_list = []
    for thr in thr_list:
        tmp_noise = (noise>thr).float()
        noise_list.append(tmp_noise)
    return noise_list[::-1]

def step_add_noise(mask, noise_list):
    """
    Args:
        mask (Tensor): Tensor with N,H,W
        noise_list (List(Tensor)): list of noise of each stage

    Returns:
        List(Tensor): list of gt mask with noise
    """
    stage_noise_mask = []
    for noise in noise_list:
        noise = noise.to(mask.device)
        noise_mask = 2 * mask * noise + 1 - noise - mask
        stage_noise_mask.append(noise_mask)
        torch.cuda.empty_cache()

    return stage_noise_mask

def generate_random_mask(gt_masks_list, gt_labels_list, number_of_query, shape):
    """_summary_

    Args:
        number_of_query (_type_): extra query for denoising
        gt_masks_list (_type_): list of batch masks, cause each instances number are not same 
        gt_labels_list (_type_): list of  batch labels

    Returns:
        _type_: _description_
    """
    random_masks = []
    corresponding_label = []
    for gt_mask, gt_label in tuple(zip(gt_masks_list, gt_labels_list)):
        # import pdb; pdb.set_trace()
        gt_mask = F.interpolate(gt_mask.unsqueeze(dim=0).float(), shape, mode='nearest').squeeze(dim=0)
        index = np.random.randint(0, gt_label.shape[0], size=number_of_query)
        gt_mask = gt_mask[index]
        gt_label = gt_label[index]
        random_masks.append(gt_mask)
        corresponding_label.append(gt_label)
    random_masks = torch.stack(random_masks, dim=0)
    corresponding_label = torch.stack(corresponding_label,dim=0)
    return random_masks, corresponding_label
