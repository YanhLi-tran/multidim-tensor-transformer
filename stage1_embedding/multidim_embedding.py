"""
多维 Embedding 包装器。

将原始 GPT-2 embedding → 维度置换 → 多维 reshape → 轴向 Dropout → 还原 → 送入 Transformer。

训练和推理时行为不同：
- 训练：置换 + reshape + 轴向 Dropout
- 推理：置换 + reshape，不做 Dropout
"""

import torch
import torch.nn as nn

from .sinkhorn import SinkhornPermutation
from .axial_dropout import AxialDropout, MultiAxisDropout


class MultidimEmbedding(nn.Module):
    """
    多维结构化 Embedding 包装器。

    Pipeline:
      (B, L, 768)
        → SinkhornPermutation: 学习最优维度排序
        → Reshape: (B, L, 12, 8, 8)
        → AxialDropout: 沿某轴丢弃整块（仅训练）
        → Reshape back: (B, L, 768)
        → 送入原始 Transformer
    """

    def __init__(
        self,
        d_model=768,
        reshape_dims=(12, 8, 8),
        dropout_axis=2,        # 在 reshape 后的哪个轴做 dropout
        dropout_p=0.1,
        sinkhorn_iters=20,
        sinkhorn_tau=1.0,
        use_multi_axis=False,  # 是否在多个轴上做 dropout
    ):
        super().__init__()
        self.d_model = d_model
        self.reshape_dims = reshape_dims
        assert d_model == reshape_dims[0] * reshape_dims[1] * reshape_dims[2], (
            f"reshape_dims ({reshape_dims}) 的乘积必须等于 d_model ({d_model})"
        )

        # 可学习的维度置换
        self.sinkhorn = SinkhornPermutation(
            d_model=d_model,
            num_sinkhorn_iters=sinkhorn_iters,
            tau=sinkhorn_tau,
        )

        # 轴向 Dropout
        if use_multi_axis:
            self.dropout = MultiAxisDropout(p_per_axis=dropout_p, axes=(dropout_axis,))
        else:
            self.dropout = AxialDropout(p=dropout_p, axis=dropout_axis)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, L, d_model) — GPT-2 embedding 输出
        Returns:
            hidden_states: (B, L, d_model) — 处理后，保持原始 shape
            P: (d_model, d_model) — 置换矩阵（用于分析）
        """
        B, L, D = hidden_states.shape

        # Step 1: 学习维度置换
        h_permuted, P = self.sinkhorn(hidden_states)

        # Step 2: Reshape 为多维张量
        d1, d2, d3 = self.reshape_dims
        h_multidim = h_permuted.reshape(B, L, d1, d2, d3)

        # Step 3: 轴向 Dropout（仅训练时）
        h_multidim = self.dropout(h_multidim)

        # Step 4: Reshape 回原始 shape
        h_flat = h_multidim.reshape(B, L, D)

        return h_flat, P
