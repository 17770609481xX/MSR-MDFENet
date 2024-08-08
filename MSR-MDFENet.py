import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import torch.fft as fft
from collections import OrderedDict
from torch import Tensor

# 旨在通过显式地建模通道间的依赖关系来提升网络的表示能力
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class FeatureEnhancedAttention(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        # 使用一个线性层来学习特征的重要性权重
        self.feature_weights = nn.Linear(emb_size, emb_size)
        # 使用sigmoid函数确保权重是正的且范围在0到1之间
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取输入张量的形状
        batch_size, emb_size, time_steps, spatial_dim = x.size()

        # 将输入张量的形状调整为(batch_size * time_steps * spatial_dim, emb_size)
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size * time_steps * spatial_dim, emb_size)

        # 计算每个特征的重要性权重
        weights = self.sigmoid(self.feature_weights(x))

        # 将权重的形状调整回(batch_size, time_steps, spatial_dim, emb_size)
        weights = weights.view(batch_size, time_steps, spatial_dim, emb_size)

        # 将输入张量的形状调整为(batch_size * time_steps * spatial_dim, emb_size)
        x = x.view(batch_size, time_steps, spatial_dim, emb_size)

        # 对输入特征进行加权
        enhanced_features = x * weights

        # 将加权后的特征的形状调整回(batch_size, emb_size, time_steps, spatial_dim)
        enhanced_features = enhanced_features.permute(0, 3, 1, 2)
        # 这个特征增强注意力机制是针对于输入张量的所有维度的，它的作用是根据输入张量的每个特征的重要性权重，
        # 对整个输入张量进行加权。在该实现中，首先通过一个线性层学习每个特征的重要性权重，然后使用 sigmoid
        # 函数将权重映射到 (0, 1) 区间，确保权重是正的。接下来，将输入张量的形状调整为
        # (batch_size * time_steps * spatial_dim, emb_size)，并将权重和特征进行相应的维度调整和扩展，
        # 最后对特征进行加权并将加权后的特征的形状调整回原始形状。因此，该特征增强注意力机制是全局性的，
        # 可以同时作用于时间维度和空间维度。不作用于通道维度
        return enhanced_features


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=16):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        # self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.conv1 = DepthwiseSeparableConv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = DepthwiseSeparableConv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = DepthwiseSeparableConv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(mip)
        # self.relu = h_swish()
        self.relu = Swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, out_channel=64):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=(1, 5), stride=(1, 2)),
            nn.BatchNorm2d(out_channel),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(0.5),

            MultiScaleResidualBlockDWSK(out_channel, out_channel*2),  # 自定义残差块
            nn.BatchNorm2d(out_channel*2),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(0.5),

            MultiScaleResidualBlockDWSK(out_channel*2, out_channel*4),  # 自定义残差块
            nn.BatchNorm2d(out_channel*4),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(0.3),
        )

        self.projection = nn.Sequential(
            # nn.Conv2d(out_channel*4, out_channel*4, kernel_size=(1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.reduction = Reduce('b n e -> b e', reduction='mean')
        self.norm = nn.LayerNorm(emb_size)
        self.fc = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = self.reduction(x)
        x = self.norm(x)
        x = self.fc(x)
        return x

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        # self.depthwise_bn = nn.BatchNorm1d(in_channels)  # 增加批量归一化  这是错误用法
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.pointwise_bn = nn.BatchNorm1d(out_channels)  # 增加批量归一化
        self.dropout = nn.Dropout(dropout_rate)
        self.Mish = Mish()


    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = F.relu(x)
        # x = self.Mish(x)
        x = self.pointwise_bn(x)
        x = self.dropout(x)
        # 应用批量归一化
        return x

class MSFRNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, out_channel=256, out_channel1=256): #官方源代码是那个？
        super().__init__()
        self.noise_layer = NoiseLayer()
        self.patch_embedding = PatchEmbedding(in_channels, out_channel=out_channel)
        self.depthwise_separable_conv = DepthwiseSeparableConvolution(4*out_channel1, 8*out_channel1)  # 确保维度一致
        self.classification_head = ClassificationHead(8*out_channel1, n_classes)
        # self.encode = TransformerEncoder(emb_size=8*out_channel1, depth=2, num_heads=8)
        # self.feature_enhanced_attention = FeatureEnhancedAttention1d(8*out_channel1)  # 特征增强注意力
    # 主模型在这
    def forward(self, x):
        # x = self.noise_layer(x)
        x = self.patch_embedding(x)  #
        x = x.permute(0, 2, 1)
        x = self.depthwise_separable_conv(x)
        x = x.permute(0, 2, 1)
        # x = self.feature_enhanced_attention(x)
        x = self.classification_head(x)
        return x
