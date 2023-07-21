import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable


def SemanticKNN(x, k, label=None):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    if not label is None:
        pairwise_distance[label != label.transpose(2, 1)] -= 5.0
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.k = k

    def forward(self, x):
        # 2 12 12000
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def get_graph_feature(coor, nor, k=10, label=None):
    batch_size, num_dims, num_points = coor.shape
    coor = coor.view(batch_size, -1, num_points)
    if label is None:
        idx = SemanticKNN(coor, k=k)
    else:
        idx = SemanticKNN(coor, k=k, label=label)
    index = idx
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()
    coor = coor.transpose(2, 1).contiguous()
    nor = nor.transpose(2, 1).contiguous()
    # coordinate
    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)
    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()
    return coor_feature, nor_feature, index


class GraphTransformerBlock(nn.Module):
    def __init__(self, feature_dim, out_dim, K):
        super(GraphTransformerBlock, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.sa = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=2)
        self.K = K

    def forward(self, Graph_index, x, feature):
        B, C, N = x.shape
        # bsz cell_num feature_dim
        x = x.contiguous().view(B, N, C)
        # bsz cell_num k out_dim
        feature = feature.permute(0, 2, 3, 1)
        # bsz cell_num k feature_dim
        neighbor_feature = index_points(x, Graph_index).view(B, -1, C)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C).reshape(B, -1, C)
        neighbor_feature, _ = self.sa(neighbor_feature, centre-neighbor_feature, neighbor_feature)
        neighbor_feature = neighbor_feature.view(B, N, self.K, C).permute(0, 3, 2, 1)
        e = self.conv(neighbor_feature).permute(0, 3, 2, 1)
        attention = F.softmax(e, dim=2)
        graph_feature = torch.sum(torch.mul(attention, feature), dim=2).permute(0, 2, 1)
        return graph_feature


class SemanticsPrediction(nn.Module):
    def __init__(self, feature_dim=1024, output_channels=15):
        super(SemanticsPrediction, self).__init__()
        self.dropout = 0.6
        self.num_classes = output_channels
        self.bn = nn.BatchNorm1d(feature_dim)
        self.pred1 = nn.Sequential(nn.Conv1d(feature_dim, int(feature_dim / 4), kernel_size=1, bias=False),
                                   nn.BatchNorm1d(int(feature_dim / 4)),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Conv1d(int(feature_dim / 4), output_channels, kernel_size=1, bias=False))
        self.dp1 = nn.Dropout(p=0.6)

    def forward(self, x):
        # x batch_size:feature_dim:cell_num
        x = self.bn(x).permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        x = self.pred1(x)
        x = self.pred4(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        x = x.max(dim=2, keepdim=True)[1]
        return x


class SGTNet(nn.Module):
    def __init__(self, k=16, pt_num=12000, in_channels=12, output_channels=15):
        super(SGTNet, self).__init__()
        self.k = k
        ''' coordinate stream '''
        self.bn1_c = nn.BatchNorm2d(64)
        self.bn2_c = nn.BatchNorm2d(128)
        self.bn3_c = nn.BatchNorm2d(256)
        self.bn4_c = nn.BatchNorm1d(512)
        self.conv1_c = nn.Sequential(nn.Conv2d(in_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1_c,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv2_c = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn2_c,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.conv3_c = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn3_c,
                                   nn.LeakyReLU(negative_slope=0.2))



        self.conv4_c = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.graph_transformer_block1 = GraphTransformerBlock(feature_dim=12, out_dim=64, K=self.k)
        self.graph_transformer_block2 = GraphTransformerBlock(feature_dim=64, out_dim=128, K=self.k)
        self.graph_transformer_block3 = GraphTransformerBlock(feature_dim=128, out_dim=256, K=self.k)
        self.FTM_c1 = STNkd(k=12)
        ''' normal stream '''
        self.bn1_n = nn.BatchNorm2d(64)
        self.bn2_n = nn.BatchNorm2d(128)
        self.bn3_n = nn.BatchNorm2d(256)
        self.bn4_n = nn.BatchNorm1d(512)
        self.conv1_n = nn.Sequential(nn.Conv2d((in_channels)*2, 64, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv2_n = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))


        self.conv3_n = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))



        self.conv4_n = nn.Sequential(nn.Conv1d(448, 512, kernel_size=1, bias=False),
                                     self.bn4_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        self.global_graph_attention1 = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.global_graph_attention2 = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.FTM_n1 = STNkd(k=12)

        self.fa = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.2))

        ''' feature fusion '''
        self.pred1 = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Conv1d(256, output_channels, kernel_size=1, bias=False))
        self.dp1 = nn.Dropout(p=0.6)
        self.dp2 = nn.Dropout(p=0.6)
        self.dp3 = nn.Dropout(p=0.6)
        coor_weight = (torch.ones([1, 1, pt_num]) + torch.zeros([1, 1, pt_num])) / 2
        self.coor_weight = nn.Parameter(coor_weight, requires_grad=True)
        nor_weight = (torch.ones([1, 1, pt_num]) + torch.zeros([1, 1, pt_num])) / 2
        self.nor_weight = nn.Parameter(nor_weight, requires_grad=True)
        self.semantics_prediction0 = SemanticsPrediction(feature_dim=24)
        self.semantics_prediction1 = SemanticsPrediction(feature_dim=128)
        self.semantics_prediction2 = SemanticsPrediction(feature_dim=256)

    def forward(self, x):
        # 2 24 12000
        coor = x[:, :12, :]
        nor = x[:, 12:24, :]
        # transform
        trans_c = self.FTM_c1(coor)
        # 2 12000 24
        coor = coor.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)
        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)
        feature0 = torch.cat((coor * self.coor_weight, nor * self.nor_weight), dim=1)
        label0 = self.semantics_prediction0(feature0)
        coor1, nor1, index = get_graph_feature(coor, nor, k=self.k, label=label0)
        coor1 = self.conv1_c(coor1)
        nor1 = self.conv1_n(nor1)
        coor1 = self.graph_transformer_block1(index, coor, coor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]
        feature1 = torch.cat((coor1 * self.coor_weight, nor1 * self.nor_weight), dim=1)
        label1 = self.semantics_prediction1(feature1)
        coor2, nor2, index = get_graph_feature(coor1, nor1, k=self.k, label=label1)
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.graph_transformer_block2(index, coor1, coor2)
        nor2 = nor2.max(dim=-1, keepdim=False)[0]
        feature2 = torch.cat((coor2 * self.coor_weight, nor2 * self.nor_weight), dim=1)
        label2 = self.semantics_prediction2(feature2)
        coor3, nor3, index = get_graph_feature(coor2, nor2, k=self.k, label=label2)
        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.graph_transformer_block3(index, coor2, coor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]
        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor = self.conv4_c(coor)
        coor = coor.permute(2, 0, 1)
        coor, _ = self.global_graph_attention1(coor, coor, coor)
        coor = coor.permute(1, 2, 0)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor = self.conv4_n(nor)
        nor = nor.permute(2, 0, 1)
        nor, _ = self.global_graph_attention2(nor, nor, nor)
        nor = nor.permute(1, 2, 0)
        x = torch.cat((coor * self.coor_weight, nor * self.nor_weight), dim=1)
        weight = self.fa(x)
        x = weight * x
        x = self.pred1(x)
        score = self.pred4(x)
        score = F.log_softmax(score, dim=1)
        score = score.permute(0, 2, 1)
        return score


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input size: [batch_size, C, N], where C is the number of feature dimension, N is the number of cells.
    x = torch.rand(1, 24, 12000)
    x = x.cuda()
    model = SGTNet(in_channels=12, output_channels=15, k=32)
    model = model.cuda()
    y = model(x)
    print(y.shape)
