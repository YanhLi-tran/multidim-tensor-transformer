"""
QKV-Only 对照组：用简单低秩矩阵替换 BERT 全部 QKV 投影。

对比逻辑：
- Tucker variant: 多维 (12,8,8) + 张量收缩 → 你的方案
- QKV-Only:      二维 h @ (A @ B^T) 低秩 → 对照组（等参数量）
- Baseline:      二维 h @ W (全连接) → 上界

若 Tucker PPL 明显低于 QKV-Only，则多维结构有额外价值。
若两者 PPL 接近，则说明收益主要来自低秩而非多维。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import BertForMaskedLM, BertConfig

from .qkv_lowrank import LowRankQKV


class LowRankBertLayer(nn.Module):
    """单层 BERT encoder layer，QKV 用低秩投影，FFN 不冻结。"""

    def __init__(self, bert_layer, d_model=768, rank=51, num_heads=12,
                 from_scratch=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.lowrank_qkv = LowRankQKV(d_model=d_model, rank=rank)

        # 借用 BERT FFN/LN（不冻结，参与训练）
        self.attention_output_dense = bert_layer.attention.output.dense
        self.attention_output_dropout = bert_layer.attention.output.dropout
        self.attention_output_LayerNorm = bert_layer.attention.output.LayerNorm

        self.intermediate_dense = bert_layer.intermediate.dense
        self.intermediate_act = bert_layer.intermediate.intermediate_act_fn
        self.output_dense = bert_layer.output.dense
        self.output_dropout = bert_layer.output.dropout
        self.output_LayerNorm = bert_layer.output.LayerNorm

        # 不冻结 FFN（与 BaselineQKV 保持一致）

    def _transpose(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, attention_mask=None):
        B, L, D = hidden_states.shape

        # 低秩 QKV
        Q, K, V = self.lowrank_qkv(hidden_states)

        # Multi-head attention
        Q = self._transpose(Q)
        K = self._transpose(K)
        V = self._transpose(V)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)

        # BERT attention output
        attn_output = self.attention_output_dense(attn_output)
        attn_output = self.attention_output_dropout(attn_output)
        hidden_states = self.attention_output_LayerNorm(hidden_states + attn_output)

        # BERT FFN
        ffn = self.intermediate_dense(hidden_states)
        ffn = self.intermediate_act(ffn)
        ffn = self.output_dense(ffn)
        ffn = self.output_dropout(ffn)
        hidden_states = self.output_LayerNorm(hidden_states + ffn)

        return hidden_states


class BERTQKVOnly(nn.Module):
    """
    BERT + 低秩 QKV 对照组。
    支持 from_scratch 随机初始化和 num_layers 控制层数。
    """

    def __init__(self, model_path="E:/BaiduNetdiskDownload/bert-base-chinese",
                 rank=51, from_scratch=False, num_layers=12):
        super().__init__()

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
            if num_layers != self.bert.config.num_hidden_layers:
                self._adjust_num_layers(num_layers)

        self.config = self.bert.config
        self.from_scratch = from_scratch
        self.num_layers = num_layers

        # 全冻结，然后解冻 FFN
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

        self.layers = nn.ModuleList([
            LowRankBertLayer(
                bert_layer=self.bert.bert.encoder.layer[i],
                d_model=self.config.hidden_size,
                rank=rank,
                num_heads=self.config.num_attention_heads,
                from_scratch=from_scratch,
            )
            for i in range(self.num_layers)
        ])

    def _adjust_num_layers(self, target_layers):
        """截断 BERT 层数（只支持减少）。"""
        current = self.bert.config.num_hidden_layers
        if target_layers > current:
            raise ValueError(
                f"num_layers={target_layers} > pretrained layers={current}，"
                f"请用 from_scratch=True。"
            )
        self.bert.bert.encoder.layer = nn.ModuleList(
            self.bert.bert.encoder.layer[:target_layers]
        )
        self.bert.config.num_hidden_layers = target_layers

    def forward(self, input_ids, labels=None, attention_mask=None):
        emb = self.bert.bert.embeddings
        hidden_states = emb(input_ids)

        extended_mask = self.bert.bert.get_extended_attention_mask(
            attention_mask, (int(input_ids.shape[0]), int(input_ids.shape[1]))
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=extended_mask)

        logits = self.bert.cls(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        return {"loss": loss, "logits": logits}

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.bert.eval()
        return self

    def get_trainable_params(self):
        """返回所有可训练参数（低秩 QKV + FFN）。"""
        params = []
        for layer in self.layers:
            params.extend(layer.lowrank_qkv.parameters())
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                params.append(param)
        # 用 id() 去重
        seen = set()
        unique = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique.append(p)
        return unique
