import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from .swin_transformer import swin_transformer_tiny, swin_transformer_small, swin_transformer_base
from .resnet import resnet18, resnet50, resnet101
from .graph import normalize_digraph
from .basic_block import *


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # in_channels: dim of node feature
        # num_classes: num of nodes
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.
        # metric: metric for assessing node similarity. Used in FGG module to build a dynamical graph
        # X' = ReLU(X + BN(V(X) + A x U(X)) )

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels,self.in_channels)
        self.V = nn.Linear(self.in_channels,self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        print(f"\n[ANFL.py]GNN.forward(), 输入参数 x.shape: {x.shape}")  # x.shape: [bs, 12, 512]
        b, n, c = x.shape # b: batch size, n: num of nodes, c: dim of node feature
        print(f"[ANFL.py]GNN.forward(), 执行b, n, c = x.shape 后 b: {b}, n: {n}, c: {c}") # 执行b, n, c = x.shape 后 b: bs, n: 12, c: 512
        # build dynamical graph
        if self.metric == 'dots': # dots 全称是 dot product
            si = x.detach() # [bs, 12, 512] si: similarity matrix
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2)) # [bs,12,512] x [bs,512,12] = [bs,12,12] einsum 全称是 Einstein summation，输入参数为公式 b i j , b j k -> b i k，表示 b i j 与 b j k 相乘，结果为 b i k，si 为输入参数, si.transpose(1, 2) 为转置
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1) # 沿着最后一个维度取出每个节点的k个最近邻，并将它们的得分储存在一个形状为[batch_size, node_num, 1]的张量threshold中。
            print(f"[ANFL.py]GNN.forward(), si.shape:{si.shape} threshold.shape: {threshold.shape}") # si.shape:[bs, 12, 12] threshold.shape: torch.Size([bs, 12, 1]) 
            # si.topk(k=self.neighbor_num, dim=-1, largest=True)：这部分使用 PyTorch 中的 topk 函数，取出最大的 k 个元素。其中，k 的值是通过参数 self.neighbor_num 传入的。dim=-1 表示在最后一个维度上执行操作（也就是矩阵的列方向）；largest=True 表示取最大的 k 个元素。 [0][:, :, -1]：这部分先使用索引 [0] 取出前面 topk 函数返回的元组中的第一个元素，也就是最大的 k 个元素；然后使用 [:, :, -1] 进一步对结果进行索引，取出每个元素在它所在维度的最后一个位置上的值。这样得到的结果是一个形状为 (batch_size, num_points) 的张量，其中每个元素是最大的 k 个元素中的最小值，也就是阈值。
            adj = (si >= threshold).float() # 将si与threshold比较，得到一个形状为[batch_size, node_num, node_num]的邻接矩阵adj。如果si中某个元素的值大于等于threshold中对应位置的值，那么adj对应位置上的值就是1.0，否则就是0.0。
            print(f"[ANFL.py]GNN.forward(), 执行adj = (si >= threshold).float() 后 adj.shape: {adj.shape}") #  adj.shape: torch.Size([bs, 12, 12])
            
            # 整体步骤：在GNN中，领接矩阵(adjacency matrix)表示节点之间的连接关系。在ANFL中，通过计算每个节点和其它节点的相似度(si)，然后将相似度大于一定阈值(threshold)的节点之间连接起来，生成领接矩阵。
            # 具体地，计算相似度(si)时，首先通过Einsum函数计算出所有节点之间的点积，得到一个大小为[b, n, n]的矩阵(si)。其中b表示batch size，n表示节点数。然后在每一行上，按从大到小排序，取前k个最大值作为阈值(threshold)，其它小于阈值的元素设为0，大于等于阈值的元素设为1，最终得到大小为[b, n, n]的领接矩阵(adj)。
            # 在ANFL中，通过取点积的方式计算相似度，如果相似度大于等于阈值，则认为这两个节点之间有连接关系。这样得到的领接矩阵可以反映节点之间的连接关系，从而在GNN中进行消息传递和信息聚合。
        elif self.metric == 'cosine': # cosine 全称是 cosine similarity
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()
        elif self.metric == 'l1': # l1 全称是 l1 distance
            si = x.detach().repeat(1, n, 1).view(b, n, n, c)
            si = torch.abs(si.transpose(1, 2) - si)
            si = si.sum(dim=-1)
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=False)[0][:, :, -1].view(b, n, 1)
            adj = (si <= threshold).float()
        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)  # normalize_digraph 作用是归一化，输入参数为 adj，表示邻接矩阵. 即: 对邻接矩阵进行归一化处理，得到归一化邻接矩阵 A
        print(f"[ANFL.py]GNN.forward(), A.shape: {A.shape}, self.V(x).shape:{self.V(x).shape}") # A.shape:[bs, 12, 12] self.V(x).shape: [bs, 12, 512]
        
        # 将 A 和特征矩阵 x 通过 Einsum 函数进行矩阵乘法，得到聚合矩阵（aggregate）
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x)) # self.V 是一个线性层
        print(f"[ANFL.py]GNN.forward(), aggregate.shape: {aggregate.shape}") # aggregate.shape: [bs, 12, 512]
        
        # 将聚合矩阵和特征矩阵 x 通过批归一化（BatchNorm）和 ReLU 激活函数进行加权和并激活，得到输出特征矩阵（x）。
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        print(f"[ANFL.py]GNN.forward(), x.shape: {x.shape}") # x.shape: [bs, 12, 512]
        return x


class Head(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.in_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.gnn = GNN(self.in_channels, self.num_classes,neighbor_num=neighbor_num,metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self, x):
        print(f"\n[ANFL.py]Head.forward(), input x.shape: {x.shape}") # x: [bs, 49, 512]
        
        # AU-specific Feature Generator (AFG)
        f_u = [] # f_u 全称 feature_u 用于存储每个类别的特征
        for i, layer in enumerate(self.class_linears): # ModuleList，里面存储了 12 个 LinearBlock
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1) # 将每个类别的特征拼接在一起 [bs, 12, 49, 512]
        f_v = f_u.mean(dim=-2)  # f_v 全称 feature_v 用于存储每个类别的平均特征 mean(dim=-2) 指将 f_u 的第二维度求平均 [bs, 12, 512]
        print(f"[ANFL.py]Head.forward(), f_u.shape: {f_u.shape} f_v shape: {f_v.shape}") # f_u.shape: torch.Size([bs, 12, 49, 512]) f_v shape: torch.Size([bs, 12, 512])
        
        # Facial Graph Generator (FGG)
        f_v = self.gnn(f_v)
        print(f"[ANFL.py]Head.forward() 执行 f_v = self.gnn(f_v) 后， f_v shape: {f_v.shape}") # f_v shape: torch.Size([bs, 12, 512])
        b, n, c = f_v.shape
        print(f"[ANFL.py]Head.forward() 执行 b, n, c = f_v.shape后， b: {b} n: {n} c: {c}") # b: bs n: 12 c: 512
        sc = self.sc
        sc = self.relu(sc)
        print(f"[ANFL.py]Head.forward() 执行 sc = self.relu(sc) 后，sc.shape: {sc.shape}") # sc.shape: torch.Size([12, 512])
        sc = F.normalize(sc, p=2, dim=-1) # sc 全称 scale
        print(f"[ANFL.py]Head.forward() 执行 sc = F.normalize(sc, p=2, dim=-1) 后，sc.shape: {sc.shape}") # sc.shape: torch.Size([12, 512])
        cl = F.normalize(f_v, p=2, dim=-1) # cl 全称 class_label
        print(f"[ANFL.py]Head.forward() 执行 cl = F.normalize(f_v, p=2, dim=-1) 后，cl.shape: {cl.shape}") # cl.shape: torch.Size([bs, 12, 512])
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        print(f"[ANFL.py]Head.forward() 执行 cl = (cl * sc.view(1, n, c)).sum(dim=-1) 后，cl.shape: {cl.shape}") # cl.shape: torch.Size([bs, 12])
        return cl


class MEFARG(nn.Module):
    def __init__(self, num_classes=12, backbone='swin_transformer_base', neighbor_num=4, metric='dots'):
        super(MEFARG, self).__init__()
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()
            self.in_channels = self.backbone.num_features
            self.out_channels = self.in_channels // 2
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)
        # print(f'[ANFL.py]class MEFARG 正在执行__init__函数 构造的模型的参数为 self.in_channels={self.in_channels}  self.out_channels={self.out_channels} num_classes={num_classes} neighbor_num={neighbor_num}  metric={metric}')
        # self.in_channels=2048  self.out_channels=512 num_classes=12 neighbor_num=4  metric=dots
        # print(f'[ANFL.py]class MEFARG 正在执行__init__函数，self.backbone: {self.backbone}  self.global_linear: {self.global_linear}  self.head: {self.head}')
        """ self.backbone: ResNet(
        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (layer1): Sequential(
            (0): Bottleneck(
            (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
                (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            )
            (1): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
        )
        (layer2): Sequential(
            (0): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
                (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            )
            (1): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
        )
        (layer3): Sequential(
            (0): Bottleneck(
            (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
                (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            )
            (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (5): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
        )
        (layer4): Sequential(
            (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
                (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
                (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            )
            (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
            (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            )
        )
        (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        (fc): None
        )  
        
        self.global_linear: LinearBlock(
        (fc): Linear(in_features=2048, out_features=512, bias=True)
        (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (drop): Dropout(p=0.0, inplace=False)
        )  
        
        self.head: Head(
        (class_linears): ModuleList(
            (0): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (1): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (2): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (3): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (4): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (5): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (6): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (7): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (8): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (9): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (10): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
            (11): LinearBlock(
            (fc): Linear(in_features=512, out_features=512, bias=True)
            (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (drop): Dropout(p=0.0, inplace=False)
            )
        )
        (gnn): GNN(
            (relu): ReLU()
            (U): Linear(in_features=512, out_features=512, bias=True)
            (V): Linear(in_features=512, out_features=512, bias=True)
            (bnv): BatchNorm1d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (relu): ReLU()
        ) """
        

    def forward(self, x):
        # x: b d c
        print(f'[ANFL.py]class MEFARG 正在执行forward函数 初始输入参数 x.shape: {x.shape}') # torch.Size([bs, 3, 224, 224]) bs张图片, 每张图片3个通道, 每个通道的图片尺寸是224*224
        x = self.backbone(x)
        print(f'[AFL.py]class MEFARG 正在执行forward函数 执行x = self.backbone(x)后， x.shape: {x.shape}') # x.shape: torch.Size([bs, 49, 2048]) bs张图片，每张图片49个特征点，每个特征点2048维
        x = self.global_linear(x)
        print(f'[AFL.py]class MEFARG 正在执行forward函数 执行x = self.global_linear(x)后， x.shape: {x.shape}') # x.shape: torch.Size([bs, 49, 512]) bs张图片，每张图片49个特征点，每个特征点512维
        cl = self.head(x)
        print(f'[AFL.py]class MEFARG 正在执行forward函数 执行cl = self.head(x)后，最终返回的cl.shape: {cl.shape}')  # torch.Size([bs, 12])
        return cl
