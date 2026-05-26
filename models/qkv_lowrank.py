"""
低秩 QKV 投影对照组。

用简单的 A @ B^T 分解替代全连接 Wq ∈ R^{768×768}：

  Q = h @ (A_q @ B_q^T)

其中 A_q ∈ R^{768×r}, B_q ∈ R^{768×r}, r 取使参数量等于 Tucker 分解的值。

目的：验证 Tucker 的多维结构是否比简单的低秩分解更有优势。
"""

import torch
import torch.nn as nn


class LowRankProj(nn.Module):
    """低秩投影：A @ B^T, A,B ∈ R^{d×r}"""

    def __init__(self, d_model=768, rank=51):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(d_model, rank) * 0.02)

    def forward(self, h):
        # h: (B, L, d) → W = A @ B^T
        # Q = h @ (A @ B^T) = (h @ A) @ B^T
        # 先算 hA: (B, L, r) 再右乘 B^T: (B, L, d)
        hA = h @ self.A              # (B, L, r)
        return hA @ self.B.T         # (B, L, d)

    @property
    def num_params(self):
        return self.A.numel() + self.B.numel()


class LowRankQKV(nn.Module):
    """三个独立的低秩投影 → Q, K, V"""

    def __init__(self, d_model=768, rank=51):
        super().__init__()
        self.Wq = LowRankProj(d_model, rank)
        self.Wk = LowRankProj(d_model, rank)
        self.Wv = LowRankProj(d_model, rank)

    def forward(self, h):
        return self.Wq(h), self.Wk(h), self.Wv(h)
