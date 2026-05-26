"""
Tucker Variant: 完整的多维 Embedding + Tucker QKV + BERT FFN。

Pipeline:
1. BERT Embedding（word + position + token_type）→ (B, L, 768)
2. Sinkhorn 维度置换 → reshape (12,8,8)
3. 轴向 Dropout（训练时）
4. reshape back → (B, L, 768)
5. N 层 TuckerBertLayer（每层: Tucker QKV → 标准 Attention → BERT FFN）
6. MLM Head 预测
"""

import torch
import torch.nn as nn
import os
from transformers import BertForMaskedLM, BertConfig

from .multidim_embedding import MultidimEmbedding
from .tucker_attention import TuckerBertLayer


class BERTTuckerVariant(nn.Module):
    """
    BERT + 多维 Embedding + Tucker QKV 全替换。

    训练时行为：
      - multidim_embedding 和 Tucker QKV 参数可训练
      - BERT FFN / LayerNorm / Output dense 冻结（除非 from_scratch）

    如果 from_scratch=True：
      - 整个模型随机初始化（包括 Embedding、FFN、LayerNorm）
      - 唯一变量：QKV 计算方式（全连接 vs Tucker）
    """

    def __init__(
        self,
        model_path="E:/BaiduNetdiskDownload/bert-base-chinese",
        rank=16,
        dropout_p=0.1,
        sinkhorn_tau=1.0,
        from_scratch=False,
        num_layers=12,
    ):
        super().__init__()

        # 加载 BERT：from_scratch 时用 config 随机初始化
        if from_scratch:
            if os.path.exists(model_path):
                config = BertConfig.from_pretrained(model_path)
            else:
                config = BertConfig(
                    vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072,
                )
            config.num_hidden_layers = num_layers
            self.bert = BertForMaskedLM(config)
        else:
            self.bert = BertForMaskedLM.from_pretrained(model_path)
            # 如果指定了 num_layers 且不等于 12，需要截断或扩充
            if num_layers != self.bert.config.num_hidden_layers:
                self._adjust_num_layers(num_layers)
            config = self.bert.config

        self.config = self.bert.config
        self.from_scratch = from_scratch
        self.num_layers = num_layers

        # 全冻结 BERT，然后解冻 FFN
        for p in self.bert.parameters():
            p.requires_grad = False

        for name, param in self.bert.named_parameters():
            if "encoder.layer" in name:
                is_ffn = (
                    "intermediate" in name or
                    ("output.dense" in name and "attention.output" not in name) or
                    ("output.LayerNorm" in name and "attention.output" not in name)
                )
                if is_ffn:
                    param.requires_grad = True

        # 多维 Embedding 模块
        self.multidim_embedding = MultidimEmbedding(
            d_model=self.config.hidden_size,
            reshape_dims=(12, 8, 8),
            dropout_axis=2,
            dropout_p=dropout_p,
            sinkhorn_iters=20,
            sinkhorn_tau=sinkhorn_tau,
        )

        # num_layers 层 Tucker Attention
        self.tucker_layers = nn.ModuleList([
            TuckerBertLayer(
                bert_layer=self.bert.bert.encoder.layer[i],
                d_model=self.config.hidden_size,
                rank=rank,
                in_dims=(12, 8, 8),
                num_heads=self.config.num_attention_heads,
            )
            for i in range(self.num_layers)
        ])

    def _adjust_num_layers(self, target_layers):
        """调整 BERT 的层数（截断或报错）。"""
        current = self.bert.config.num_hidden_layers
        if target_layers >= current:
            raise ValueError(
                f"num_layers={target_layers} > pretrained layers={current}，"
                f"请用 from_scratch=True 或从预训练模型初始化后手动扩充。"
            )
        # 截断：只保留前 target_layers 层
        self.bert.bert.encoder.layer = nn.ModuleList(
            self.bert.bert.encoder.layer[:target_layers]
        )
        self.config.num_hidden_layers = target_layers

    def get_embeddings(self, input_ids, token_type_ids=None):
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
        # Step 1: BERT Embedding
        hidden_states = self.get_embeddings(input_ids, token_type_ids)

        # Step 2: 多维结构化处理
        hidden_states, P = self.multidim_embedding(hidden_states)

        # Step 3: 构造 attention mask
        extended_attention_mask = self.bert.bert.get_extended_attention_mask(
            attention_mask, (int(input_ids.shape[0]), int(input_ids.shape[1]))
        )

        # Step 4: 12 层 Tucker Attention
        for layer in self.tucker_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
            )

        # Step 5: MLM Head
        prediction_scores = self.bert.cls(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        result = {"loss": loss, "logits": prediction_scores}
        if output_P:
            result["P"] = P
        return result

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.bert.eval()
        return self

    def get_trainable_params(self):
        """返回所有可训练参数（multidim_embedding + TuckerQKV + FFN）。"""
        params = []
        params.extend(self.multidim_embedding.parameters())
        for layer in self.tucker_layers:
            params.extend(layer.tucker_qkv.parameters())
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                params.append(param)
        seen = set()
        unique = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique
