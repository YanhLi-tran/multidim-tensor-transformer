"""
轴向结构化 Dropout：在 reshape 后的多维张量上做 dropout。

与普通 Dropout 的关键区别：
- 普通 Dropout：每个标量元素独立丢弃
- 轴向 Dropout：沿某个轴随机丢弃一整"片"（chunk）

这种结构化噪声迫使模型在多个轴上产生信息冗余，可能引导出解耦的语义子空间。
"""

import torch
import torch.nn as nn


class AxialDropout(nn.Module):
    """
    在 reshape 后的多维张量上做轴向 dropout。

    假设 h 被 reshape 为 (B, L, d1, d2, d3)，如 (B, L, 12, 8, 8)
    对 axis=2（即 d1=12 维）以概率 p 随机丢弃整个 chunk。

    训练时随机丢弃，推理时关闭（和标准 dropout 行为一致）。
    """

    def __init__(self, p=0.1, axis=2):
        """
        Args:
            p: 丢弃概率
            axis: 要丢弃的轴（在 reshape 后的张量中的位置索引）
                  0=batch, 1=seq, 2=d1, 3=d2, 4=d3
        """
        super().__init__()
        self.p = p
        self.axis = axis

    def forward(self, x):
        """
        Args:
            x: (B, L, d1, d2, d3) — reshape 后的多维张量
        Returns:
            x_dropped: 同 shape，某些轴被整片丢弃
        """
        if not self.training or self.p <= 0:
            return x

        # 在 axis 维度上生成 mask
        # mask shape: 和 x 的指定 axis 维度对齐
        mask_shape = [1] * x.dim()
        mask_shape[self.axis] = x.shape[self.axis]
        mask = torch.bernoulli(
            torch.full(mask_shape, 1 - self.p, device=x.device, dtype=x.dtype)
        )
        mask = mask / (1 - self.p)  # 尺度校准，保持期望不变

        return x * mask


class MultiAxisDropout(nn.Module):
    """
    在多个轴上独立做轴向 Dropout。

    每个轴独立进行 dropout，ensemble 效果更强——模型被迫在所有轴上都有冗余。
    """

    def __init__(self, p_per_axis=0.1, axes=(2, 3)):
        """
        Args:
            p_per_axis: 每个轴的独立丢弃概率
            axes: 要做 dropout 的轴列表
        """
        super().__init__()
        self.p = p_per_axis
        self.axes = axes

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x

        result = x
        for axis in self.axes:
            mask_shape = [1] * x.dim()
            mask_shape[axis] = x.shape[axis]
            mask = torch.bernoulli(
                torch.full(mask_shape, 1 - self.p, device=x.device, dtype=x.dtype)
            )
            mask = mask / (1 - self.p)
            result = result * mask

        return result
