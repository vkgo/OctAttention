# -*- coding: utf-8 -*-
# ---
# @Software: PyCharm
# @File: FPS.py
# @Author: vkgo
# @E-mail: hwjho@qq.com, csvk@mail.scut.edu.cn
# @Time: May 11, 2023

# ---
import torch
import torch.nn as nn

def farthest_point_sampling(point_cloud, num_samples):
    """
    对点云进行最远点采样（FPS）的函数。

    参数：
    point_cloud (torch.Tensor): 输入的点云，形状为 (总点数，batch_size，父节点层数，6)，最后三维为 xyz 坐标。
    num_samples (int): 需要采样的点的数量。

    返回：
    torch.Tensor: 下采样后的点云，形状为 (num_samples, batch_size, 父节点层数, 6)。
    """

    total_points, batch_size, _, _ = point_cloud.shape

    # 如果需要的采样点数大于或等于总点数，直接返回输入的点云
    if num_samples >= total_points:
        return point_cloud

    # 提取 xyz 坐标，只取最后一个维度为4的切片
    xyz = point_cloud[:, :, -1, -3:].float()

    # 初始化采样点的索引列表
    sampled_indices = torch.zeros((num_samples, batch_size), dtype=torch.long)

    # 对每个 batch 进行处理
    for i in range(batch_size):
        # 随机选择一个初始点
        sampled_indices[0, i] = torch.randint(total_points, (1,))

        for j in range(1, num_samples):
            dist = torch.cdist(xyz[:, i], xyz[sampled_indices[:j, i], i])
            farthest_point_idx = torch.argmax(torch.min(dist, dim=1)[0])
            sampled_indices[j, i] = farthest_point_idx

    # 使用采样的索引获取点云数据
    sampled_point_cloud = point_cloud[sampled_indices, torch.arange(batch_size).unsqueeze(0), -1]

    return sampled_point_cloud


class PointNetGlobalFeatureExtractor(nn.Module):
    """
    PointNet 全局特征提取层，将整个点云的特征提取为 255 维。
    """

    def __init__(self):
        super(PointNetGlobalFeatureExtractor, self).__init__()

        self.featureExtractor_conv1 = nn.Conv1d(6, 64, )
        self.featureExtractor_conv2 = nn.Conv1d(64, 128, 1)
        self.featureExtractor_conv3 = nn.Conv1d(128, 255, 1)

        self.featureExtractor_bn1 = nn.BatchNorm1d(64)
        self.featureExtractor_bn2 = nn.BatchNorm1d(128)
        self.featureExtractor_bn3 = nn.BatchNorm1d(255)

    def forward(self, x):
        """
        前向传播函数。

        参数：
        x (torch.Tensor): 输入的点云数据，形状为 (num_points, batch_size, 6)。

        返回：
        torch.Tensor: 提取的全局特征，形状为 (batch_size, 255)。
        """
        x = x.transpose(0, 1).transpose(1, 2).float()
        x = self.featureExtractor_bn1(self.featureExtractor_conv1(x))
        x = self.featureExtractor_bn2(self.featureExtractor_conv2(x))
        x = self.featureExtractor_bn3(self.featureExtractor_conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(-1)

        return x