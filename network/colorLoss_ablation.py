import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

class colorLossAbl(nn.Module):
    def __init__(self, device='cuda'):  # 移除tau参数（不再需要温度系数）
        super().__init__()
        # 定义11种颜色类别（与用户指定的类别列表一致）
        self.class_names = ["Red", "Green", "Blue", "Black", "White", "Gray", "Pink", "Orange", "Purple", "Yellow", "Brown"]
        # 建立颜色名称到类别索引的映射（0-10）
        self.name_to_index = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)  # 类别数量固定为11
        self.device = device

    # 移除原有的infoNCELoss和infoNCELoss_fast方法（不再使用embedding约束）
    
    def forward(self, x: torch.Tensor, x_names: tuple):
        # 将颜色名称转换为类别索引（目标标签）
        targets = torch.tensor(
            [self.name_to_index[name] for name in x_names],
            dtype=torch.long,
            device=self.device
        )
        # 直接计算交叉熵损失（x是b×11的logits向量）
        total_loss = F.cross_entropy(x, targets)
        return total_loss

    # 保留classification方法用于评估（如需），但修改为直接使用类别索引
    def classification(self, x: torch.Tensor, x_names: tuple):
        """返回预测类别索引和真实类别索引"""
        # x是b×11的logits，直接取最大值索引作为预测
        _, class_index = torch.max(x, dim=1, keepdim=True)  # Nx1
        # 真实类别索引
        class_index_gt = torch.tensor(
            [self.name_to_index[name] for name in x_names],
            dtype=torch.long,
            device=x.device
        ).unsqueeze(1)  # Nx1
        return class_index, class_index_gt

    # 保留get_logits方法（如需），输出softmax概率分布
    def get_logits(self, x: torch.Tensor):
        """返回类别概率分布"""
        logits = F.softmax(x, dim=1)  # b×11
        return logits

if __name__ == '__main__':
    # 测试用例：x是2×11的logits向量，x_names是对应的颜色名称
    criteria = colorLoss(device='cpu')
    # 随机生成logits（实际使用时应替换为模型输出）
    x = torch.randn(2, 11).float()  # 2个样本，11个类别
    colors = ('Blue', 'Blue')    # 真实标签
    loss = criteria(x, colors)
    print('loss B-B', loss)
    colors = ('Red', 'Blue')     # 真实标签
    loss = criteria(x, colors)
    print('loss B-R', loss)
