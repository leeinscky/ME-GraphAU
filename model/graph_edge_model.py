import torch
import torch.nn as nn
import math


class CrossAttn(nn.Module):
    """ cross attention Module"""
    def __init__(self, in_channels):
        super(CrossAttn, self).__init__()
        self.in_channels = in_channels
        self.linear_q = nn.Linear(in_channels, in_channels // 2)
        self.linear_k = nn.Linear(in_channels, in_channels // 2)
        self.linear_v = nn.Linear(in_channels, in_channels)
        self.scale = (self.in_channels // 2) ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.linear_k.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_q.weight.data.normal_(0, math.sqrt(2. / (in_channels // 2)))
        self.linear_v.weight.data.normal_(0, math.sqrt(2. / in_channels))

    def forward(self, y, x):
        query = self.linear_q(y)
        key = self.linear_k(x)
        value = self.linear_v(x)
        dots = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, value)
        return out


class GEM(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GEM, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.FAM = CrossAttn(self.in_channels)
        self.ARM = CrossAttn(self.in_channels)
        self.edge_proj = nn.Linear(in_channels, in_channels)
        self.bn = nn.BatchNorm2d(self.num_classes * self.num_classes)

        self.edge_proj.weight.data.normal_(0, math.sqrt(2. / in_channels))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, class_feature, global_feature):
        print('[graph_edge_model.py] class_feature.shape: ', class_feature.shape, ', global_feature.shape: ', global_feature.shape) # class_feature.shape: [bs, 12, 49, 512] , global_feature.shape: [bs, 49, 512]
        B, N, D, C = class_feature.shape
        global_feature = global_feature.repeat(1, N, 1).view(B, N, D, C)
        feat = self.FAM(class_feature, global_feature)
        print('[graph_edge_model.py] after FAM, feat.shape: ', feat.shape) # feat.shape: [bs, 12, 49, 512]
        feat_end = feat.repeat(1, 1, N, 1).view(B, -1, D, C)
        feat_start = feat.repeat(1, N, 1, 1).view(B, -1, D, C)
        feat = self.ARM(feat_start, feat_end)
        print('[graph_edge_model.py] after ARM, feat.shape: ', feat.shape, ', input param is feat_start.shape:', feat_start.shape, ', feat_end.shape:', feat_end.shape) # feat.shape: [bs, 144, 49, 512], input param is feat_start.shape: [bs, 144, 49, 512], feat_end.shape: [bs, 144, 49, 512]
        edge = self.bn(self.edge_proj(feat))
        print('[graph_edge_model.py] after edge_proj, final edge.shape: ', edge.shape) # edge.shape: [bs, 144, 49, 512]
        return edge



