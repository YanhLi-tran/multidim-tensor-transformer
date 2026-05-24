"""
Variant: BERT-base-chinese + 多维结构化 Embedding + 轴向 Dropout。

Forward 流程：
1. BERT Embedding (word + position + token_type) → (B, L, 768)
2. **MultidimEmbedding** → 维度置换 + reshape (12,8,8) + 轴向 Dropout → reshape back (768)
3. 通过 inputs_embeds 送入原始 BERT（跳过内部 embedding）
4. MLM Head 预测（仅 masked 位置计算 loss）
"""

import torch
import torch.nn as nn
from transformers import BertForMaskedLM

from .multidim_embedding import MultidimEmbedding


class BERTMultidimVariant(nn.Module):
    """
    BERT-base-chinese + 多维 Embedding 变体。

    核心改动：在 embedding 后插入 MultidimEmbedding 模块，
    然后通过 inputs_embeds 传给 BERT，跳过其内部 embedding 层。
    BERT 主体全部冻结，仅训练 multidim_embedding 模块。
    """

    def __init__(
        self,
        model_path="E:/BaiduNetdiskDownload/bert-base-chinese",
        reshape_dims=(12, 8, 8),
        dropout_axis=2,
        dropout_p=0.1,
        sinkhorn_tau=1.0,
        use_multi_axis=False,
    ):
        super().__init__()

        self.bert = BertForMaskedLM.from_pretrained(model_path)
        self.config = self.bert.config

        # 冻结 BERT 主体
        for param in self.bert.parameters():
            param.requires_grad = False

        # 多维 Embedding 模块（可训练）
        self.multidim_embedding = MultidimEmbedding(
            d_model=self.config.hidden_size,
            reshape_dims=reshape_dims,
            dropout_axis=dropout_axis,
            dropout_p=dropout_p,
            sinkhorn_iters=20,
            sinkhorn_tau=sinkhorn_tau,
            use_multi_axis=use_multi_axis,
        )

    def get_embeddings(self, input_ids, token_type_ids=None):
        """计算 BERT embeddings (word + position + token_type + LN + dropout)"""
        emb = self.bert.bert.embeddings

        word_embeds = emb.word_embeddings(input_ids)

        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand_as(input_ids)
        position_embeds = emb.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = emb.token_type_embeddings(token_type_ids)

        hidden_states = word_embeds + position_embeds + token_type_embeds
        hidden_states = emb.LayerNorm(hidden_states)
        hidden_states = emb.dropout(hidden_states)

        return hidden_states

    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        token_type_ids=None,
        output_P=False,
    ):
        """
        Args:
            input_ids: (B, L), 部分位置已被 mask
            labels: (B, L), -100 表示不计算 loss
            attention_mask: (B, L)
            token_type_ids: (B, L), 可选
            output_P: 是否输出置换矩阵 P
        """
        # Step 1: 获取 BERT embedding
        hidden_states = self.get_embeddings(input_ids, token_type_ids)

        # Step 2: === 核心改动：多维结构化处理 ===
        hidden_states, P = self.multidim_embedding(hidden_states)
        # =======================================

        # Step 3: 通过 inputs_embeds 送入 BERT（跳过内部 embedding 层）
        outputs = self.bert(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        # transformers 5.x 返回 tuple (loss, logits)
        if isinstance(outputs, tuple):
            result = {"loss": outputs[0], "logits": outputs[1]}
        else:
            result = {"loss": outputs.loss, "logits": outputs.logits}

        if output_P:
            result["P"] = P

        return result

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.bert.eval()
        return self

    def get_trainable_params(self):
        return list(self.multidim_embedding.parameters())
