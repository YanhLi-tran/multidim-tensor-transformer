"""
Tucker Attention Layer：用 Tucker QKV 替换全连接 QKV，其余组件借用 BERT。

每层的计算流程：
1. Tucker QKV → Q, K, V
2. 标准 Multi-Head Scaled Dot-Product Attention
3. BERT 原始 attention output dense + dropout + residual + LayerNorm
4. BERT 原始 FFN (intermediate + output) + residual + LayerNorm

仅 QKV 投影被替换，其他（output dense, FFN, LayerNorm）全部来自原始 BERT 权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tucker_qkv import TuckerQKV


class TuckerBertLayer(nn.Module):
    """
    单层 BERT encoder layer，QKV 投影用 Tucker 分解。

    使用原始 BERT 的 output dense、FFN、LayerNorm（冻结），
    仅训练 TuckerQKV 参数。
    """

    def __init__(
        self,
        bert_layer,          # 原始 BERT 的 BertLayer
        d_model=768,
        rank=16,
        in_dims=(12, 8, 8),
        num_heads=12,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # === Tucker QKV 替换（可训练）===
        self.tucker_qkv = TuckerQKV(
            d_model=d_model,
            rank=rank,
            in_dims=in_dims,
            num_heads=num_heads,
        )

        # === 从原始 BERT layer 借用的组件（冻结）===
        self.attention_output_dense = bert_layer.attention.output.dense
        self.attention_output_dropout = bert_layer.attention.output.dropout
        self.attention_output_LayerNorm = bert_layer.attention.output.LayerNorm

        self.intermediate_dense = bert_layer.intermediate.dense
        self.intermediate_act = bert_layer.intermediate.intermediate_act_fn
        self.output_dense = bert_layer.output.dense
        self.output_dropout = bert_layer.output.dropout
        self.output_LayerNorm = bert_layer.output.LayerNorm

        # 不冻结借用组件（由外层 BERTTuckerVariant 统一管理 requires_grad）
        # FFN 需要参与训练，attention output 保持冻结

    def _transpose_for_scores(self, x):
        """将 (B, L, d) reshape 为 (B, num_heads, L, head_dim)"""
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, L, 768)
            attention_mask: (B, 1, 1, L) — BERT 风格的扩展 mask
        Returns:
            hidden_states: (B, L, 768)
        """
        # ====== 1. 多维 Reshape + Tucker QKV ======
        B, L, D = hidden_states.shape
        h_multidim = hidden_states.reshape(B, L, 12, 8, 8)
        Q, K, V = self.tucker_qkv(h_multidim)  # each (B, L, 768)

        # ====== 2. Multi-Head Attention ======
        Q = self._transpose_for_scores(Q)  # (B, n_heads, L, head_dim)
        K = self._transpose_for_scores(K)
        V = self._transpose_for_scores(V)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attention_mask,
            dropout_p=0.0,  # 原始 BERT 的 attention dropout
            is_causal=False,
        )

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)

        # ====== 3. BERT Attention Output ======
        attn_output = self.attention_output_dense(attn_output)
        attn_output = self.attention_output_dropout(attn_output)
        hidden_states = self.attention_output_LayerNorm(
            hidden_states + attn_output
        )

        # ====== 4. BERT FFN ======
        ffn_output = self.intermediate_dense(hidden_states)
        ffn_output = self.intermediate_act(ffn_output)
        ffn_output = self.output_dense(ffn_output)
        ffn_output = self.output_dropout(ffn_output)
        hidden_states = self.output_LayerNorm(
            hidden_states + ffn_output
        )

        return hidden_states
