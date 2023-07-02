import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from mmcv.cnn import PLUGIN_LAYERS
from mmdet.utils import show_feature


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 定义查询、键、值的线性变换
        self.query = torch.nn.Linear(dim, dim, bias=False)
        self.key = torch.nn.Linear(dim, dim, bias=False)
        self.value = torch.nn.Linear(dim, dim, bias=False)
        
        # 定义输出的线性变换
        self.out = torch.nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        # 将输入张量按照头的数量拆分，并进行线性变换
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 将拆分后的张量转置，以便在乘法中将头和序列维度进行匹配
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # 使用批量矩阵乘法计算注意力分数，并除以根号下维度
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.head_dim)
        # 对注意力分数进行softmax，以获得注意力权重
        weights = torch.softmax(scores, dim=-1)
        # 使用注意力权重对值进行加权求和
        attention = torch.matmul(weights, values)
        # 将头的维度合并回原始的维度，并进行输出的线性变换
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        x = self.out(attention)
        return x

@PLUGIN_LAYERS.register_module()
class spatialBias_v1(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, **kwargs):
        super().__init__()
        self.dim_reduction = nn.Sequential(*[nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                                            nn.BatchNorm2d(hidden_dim),
                                            nn.ReLU()])
        self.apt_pool = nn.AdaptiveAvgPool2d((16,16))
        self.att = SelfAttention(hidden_dim, 1)
        self.position_compress = nn.Sequential(*[nn.Conv2d(hidden_dim, 1, kernel_size=1),
                                                nn.ReLU()])
        self.fuse = nn.Sequential(*[nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),
                                    nn.BatchNorm2d(in_channels)])
    
    def forward(self, x):
        x_red = self.dim_reduction(x)
        x_red_reshape = self.apt_pool(x_red)
        b_,c_,h_,w_ = x_red_reshape.shape
        x_seq = rearrange(x_red_reshape, 'b c h w -> b (h w) c')
        x_seq_embedding = self.att(x_seq)
        x_seq_embedding_ = rearrange(x_seq_embedding, 'b (h w) c -> b c h w', h=h_)
        spatial_bias = self.position_compress(x_seq_embedding_)
        spatial_bias_ = F.interpolate(spatial_bias, size=(x.shape[-2:]),mode='bilinear')
        x_ = torch.cat([x,spatial_bias_], dim=1)
        out = self.fuse(x_)
        return out

@PLUGIN_LAYERS.register_module()
class spatialBias_v2(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, **kwargs):
        super().__init__()
        self.dim_reduction = nn.Sequential(*[nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                                            nn.BatchNorm2d(hidden_dim),
                                            nn.ReLU()])
        self.apt_pool = nn.AdaptiveAvgPool2d((16,16))
        self.att = SelfAttention(hidden_dim, 1)
        self.position_compress = nn.Sequential(*[nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
                                                nn.BatchNorm2d(in_channels),
                                                nn.ReLU()])
        # self.fuse = nn.Sequential(*[nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),
        #                             nn.BatchNorm2d(in_channels)])
    
    def forward(self, x):
        x_red = self.dim_reduction(x)
        x_red_reshape = self.apt_pool(x_red)
        b_,c_,h_,w_ = x_red_reshape.shape
        x_seq = rearrange(x_red_reshape, 'b c h w -> b (h w) c')
        x_seq_embedding = self.att(x_seq)
        x_seq_embedding_ = rearrange(x_seq_embedding, 'b (h w) c -> b c h w', h=h_)
        spatial_bias = self.position_compress(x_seq_embedding_)
        spatial_bias_ = F.interpolate(spatial_bias, size=(x.shape[-2:]),mode='bilinear')
        # x_ = torch.cat([x,spatial_bias_], dim=1)
        # out = self.fuse(x_)
        return x + spatial_bias_

@PLUGIN_LAYERS.register_module()
class spatialBias_v3(nn.Module):
    """first 18 experiments used"""
    def __init__(self, in_channels, hidden_dim=256*4, **kwargs):
        super().__init__()
        self.dim_reduction = nn.Sequential(*[nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                                            nn.BatchNorm2d(hidden_dim),
                                            nn.ReLU()])
        self.apt_pool = nn.AdaptiveAvgPool2d((32,32))
        self.att = SelfAttention(hidden_dim, 8)
        self.position_compress = nn.Sequential(*[nn.Conv2d(hidden_dim, 512, kernel_size=1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(512, in_channels, kernel_size=1)])
        # self.fuse = nn.Sequential(*[nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),
        #                             nn.BatchNorm2d(in_channels)])
    
    def forward(self, x):
        x_red = self.dim_reduction(x)
        x_red_reshape = self.apt_pool(x_red)
        b_,c_,h_,w_ = x_red_reshape.shape
        x_seq = rearrange(x_red_reshape, 'b c h w -> b (h w) c')
        x_seq_embedding = self.att(x_seq)
        x_seq_embedding_ = rearrange(x_seq_embedding, 'b (h w) c -> b c h w', h=h_)
        spatial_bias = self.position_compress(x_seq_embedding_)
        spatial_bias_ = F.interpolate(spatial_bias, size=(x.shape[-2:]),mode='bilinear')
        # x_ = torch.cat([x,spatial_bias_], dim=1)
        # out = self.fuse(x_)
        return x + spatial_bias_

@PLUGIN_LAYERS.register_module()
class spatialBias(nn.Module):
    """shared ATTENTION"""
    def __init__(self, in_channels, hidden_dim=256*4, **kwargs):
        super().__init__()
        self.dim_reduction = nn.Sequential(*[nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                                            nn.BatchNorm2d(hidden_dim),
                                            nn.ReLU()])
        self.apt_pool = nn.AdaptiveAvgPool2d((32,32))
        self.position_compress = nn.Sequential(*[nn.Conv2d(hidden_dim, 512, kernel_size=1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(512, 512, kernel_size=1),
                                                 nn.ReLU(),
                                                 nn.Conv2d(512, in_channels, kernel_size=1)])
        # self.fuse = nn.Sequential(*[nn.Conv2d(in_channels + 1, in_channels, kernel_size=1),
        #                             nn.BatchNorm2d(in_channels)])
    
    def forward(self, x, attn_layer):
        x_red = self.dim_reduction(x)
        x_red_reshape = self.apt_pool(x_red)
        b_,c_,h_,w_ = x_red_reshape.shape
        x_seq = rearrange(x_red_reshape, 'b c h w -> b (h w) c')
        x_seq_embedding = attn_layer(x_seq)
        x_seq_embedding_ = rearrange(x_seq_embedding, 'b (h w) c -> b c h w', h=h_)
        spatial_bias = self.position_compress(x_seq_embedding_)
        spatial_bias_ = F.interpolate(spatial_bias, size=(x.shape[-2:]),mode='bilinear')
        # x_ = torch.cat([x,spatial_bias_], dim=1)
        # out = self.fuse(x_)
        return x + spatial_bias_

# if __name__ == '__main__':
#     model = spatialBias(256)
#     a = torch.rand((2,256,17,17))
#     print(model(a).shape)
