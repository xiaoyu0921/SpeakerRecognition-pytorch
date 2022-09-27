import torch
import torch.nn as nn
from utils_ import myconfig


def get_triplet_loss(anchor, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    # 实现了上篇论文中定义的三元损失函数
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    # 使用余弦相似度
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + myconfig.TRIPLET_ALPHA,
        torch.tensor(0.0))


def get_triplet_loss_from_batch_output(batch_output, batch_size):
    """Triplet loss from N*(a|p|n) batch output."""
    # 计算一个数据批次的输出对应的损失函数
    batch_output_reshaped = torch.reshape(
        batch_output, (batch_size, 3, batch_output.shape[1]))
    # print(batch_output_reshaped.shape)

    batch_loss = get_triplet_loss(
        batch_output_reshaped[:, 0, :],
        batch_output_reshaped[:, 1, :],
        batch_output_reshaped[:, 2, :])
    loss = torch.mean(batch_loss)
    return loss