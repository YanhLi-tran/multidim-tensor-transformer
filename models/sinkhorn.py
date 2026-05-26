"""
Sinkhorn 归一化：将任意矩阵转换为双随机矩阵（近似置换矩阵）。

用于学习 hidden 维度之间的最优重排序——让语义相关的维度在 reshaped 张量中相邻。
"""

import torch
import torch.nn as nn


class SinkhornPermutation(nn.Module):
    """
    可学习的维度置换模块。

    原理：
    1. 学习一个 raw 矩阵 S ∈ R^{d×d}（通过小参数矩阵的外积构造，节省参数）
    2. 通过 Sinkhorn 迭代将 S 归一化为双随机矩阵 P
    3. 用温度参数 τ 控制硬/软程度：
       - 训练早期 τ 大 → 软（梯度流好）
       - 训练后期 τ 小 → 硬（接近真正的置换）
    4. h_permuted = P @ h，即对 hidden 维度做线性重排

    参数量：d×d ≈ 768² ≈ 0.59M（如果直接存），可以用低秩近似进一步压缩。
    """

    def __init__(self, d_model, num_sinkhorn_iters=20, tau=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_iters = num_sinkhorn_iters
        self.tau = tau

        # 用低秩近似减少参数：S = A @ B^T, A,B ∈ R^{d×r}
        self.rank = min(64, d_model // 4)
        self.A = nn.Parameter(torch.randn(d_model, self.rank) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, self.rank) * 0.01)

    def sinkhorn(self, S, eps=1e-6):
        """Sinkhorn-Knopp 迭代，将非负矩阵归一化为双随机矩阵。"""
        for _ in range(self.num_iters):
            # 行归一化
            row_sum = S.sum(dim=-1, keepdim=True)
            S = S / (row_sum + eps)
            # 列归一化
            col_sum = S.sum(dim=-2, keepdim=True)
            S = S / (col_sum + eps)
        return S

    def forward(self, h, temperature=None):
        """
        Args:
            h: (B, L, d_model) — hidden states
            temperature: 温度参数，None 则用 self.tau
        Returns:
            h_permuted: (B, L, d_model) — 重排后的 hidden states
            P: (d_model, d_model) — 置换矩阵（双随机）
        """
        tau = temperature if temperature is not None else self.tau

        # 构造评分矩阵: S = A @ B^T
        raw_S = self.A @ self.B.T  # (d, d)

        # 通过 softmax + 温度 → 非负矩阵
        S_positive = torch.exp(raw_S / tau)

        # Sinkhorn 归一化 → 双随机
        P = self.sinkhorn(S_positive.clone())  # (d, d)

        # 对 hidden 维度做线性重排
        batch_size, seq_len, _ = h.shape
        h_permuted = h @ P.T  # (B, L, d) × (d, d)^T = (B, L, d)

        return h_permuted, P

    def set_temperature(self, tau):
        """动态调整温度（训练中逐步降低）。"""
        self.tau = tau
