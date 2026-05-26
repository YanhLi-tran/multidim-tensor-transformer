"""
Tucker 分解的 QKV 投影模块。

将传统的 h @ Wq ∈ R^{d×d} 替换为 Tucker-4 分解的多轴收缩积：

  h ∈ R^{B, L, d₁, d₂, d₃} = R^{B, L, 12, 8, 8}

  Q = TuckerProj(h, G, A, B, C, D)

  Step 1: 在三个轴上分别做低维投影
    h_a = contract(h, A, dims=([2], [0]))    → (B, L, r₁, d₂, d₃)
    h_b = contract(h_a, B, dims=([3], [0]))  → (B, L, r₁, r₂, d₃)
    h_c = contract(h_b, C, dims=([4], [0]))  → (B, L, r₁, r₂, r₃)

  Step 2: 过 core tensor
    h_core = contract(h_c, G, dims=([2,3,4], [0,1,2]))  → (B, L, r₄)

  Step 3: 投影到原始维度
    Q = h_core @ D^T  → (B, L, 768)

参数量对比（per projection，rank=16）:
  Tucker:  ~78K  (vs 全连接 589K)
  12层×3:  ~2.8M (vs 全连接 ~21M)
"""

import torch
import torch.nn as nn


class TuckerProj(nn.Module):
    """
    单个 Tucker 投影：多维张量 → 输出向量。

    输入: (B, L, 12, 8, 8)  — reshape 后的 embedding
    输出: (B, L, d_out)     — Q / K / V
    """

    def __init__(self, d_out=768, rank=16, in_dims=(12, 8, 8)):
        """
        Args:
            d_out: 输出维度（通常是 768）
            rank: Tucker decomposition 的 rank (r₁=r₂=r₃=r₄)
            in_dims: 输入的多维 shape (d₁, d₂, d₃)
        """
        super().__init__()
        d1, d2, d3 = in_dims
        r = rank

        # Mode factors
        self.A = nn.Parameter(torch.randn(d1, r) * 0.02)
        self.B = nn.Parameter(torch.randn(d2, r) * 0.02)
        self.C = nn.Parameter(torch.randn(d3, r) * 0.02)

        # Core tensor
        self.G = nn.Parameter(torch.randn(r, r, r, r) * 0.02)

        # Output projection
        self.D = nn.Parameter(torch.randn(d_out, r) * 0.02)

        self.d_out = d_out
        self.rank = r

    def forward(self, h_multidim):
        """
        Args:
            h_multidim: (B, L, d1, d2, d3) = (B, L, 12, 8, 8)
        Returns:
            output: (B, L, d_out) = (B, L, 768)
        """
        B, L, d1, d2, d3 = h_multidim.shape
        r = self.rank

        # Step 1: 沿三个轴分别做低维投影
        # h: (B, L, d1, d2, d3) × A: (d1, r) → (B, L, r, d2, d3)
        h_a = torch.einsum('b l i j k, i r -> b l r j k', h_multidim, self.A)

        # h_a: (B, L, r, d2, d3) × B: (d2, r) → (B, L, r, r, d3)
        h_b = torch.einsum('b l r j k, j s -> b l r s k', h_a, self.B)

        # h_b: (B, L, r, r, d3) × C: (d3, r) → (B, L, r, r, r)
        h_c = torch.einsum('b l r s k, k t -> b l r s t', h_b, self.C)

        # Step 2: 过 core tensor
        # h_c: (B, L, r, r, r) × G: (r, r, r, r) → (B, L, r)
        h_core = torch.einsum('b l r s t, r s t u -> b l u', h_c, self.G)

        # Step 3: 投影到输出维度
        # h_core: (B, L, r) × D: (d_out, r)^T → (B, L, d_out)
        output = h_core @ self.D.T

        return output

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class TuckerQKV(nn.Module):
    """
    三个独立的 Tucker 投影 → Q, K, V。

    每个投影有自己的 Tucker 分解参数（A, B, C, G, D）。
    """

    def __init__(self, d_model=768, rank=16, in_dims=(12, 8, 8), num_heads=12):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 64

        # 三个独立的 Tucker 投影
        self.Wq = TuckerProj(d_out=d_model, rank=rank, in_dims=in_dims)
        self.Wk = TuckerProj(d_out=d_model, rank=rank, in_dims=in_dims)
        self.Wv = TuckerProj(d_out=d_model, rank=rank, in_dims=in_dims)

    def forward(self, h_multidim):
        """
        Args:
            h_multidim: (B, L, 12, 8, 8)
        Returns:
            Q, K, V: each (B, L, 768)
        """
        Q = self.Wq(h_multidim)
        K = self.Wk(h_multidim)
        V = self.Wv(h_multidim)
        return Q, K, V

    @property
    def num_params(self):
        return self.Wq.num_params + self.Wk.num_params + self.Wv.num_params
