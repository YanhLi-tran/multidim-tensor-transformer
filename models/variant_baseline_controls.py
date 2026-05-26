"""
Baseline 对照组：

1. BaselineFrozen: 完全冻结 BERT（QKV + FFN + Embedding），下界
2. BaselineQKV: 仅训练全连接 QKV（冻结 FFN/Embedding），上界 / 天花板
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import BertForMaskedLM, BertConfig


# ============================================================
# BaselineFrozen — 下界
# ============================================================

class BERTBaselineFrozen(nn.Module):
    """BERT 完全冻结，不训练任何参数。用于验证随机猜测的 PPL 下界。"""

    def __init__(self, model_path="E:/BaiduNetdiskDownload/bert-base-chinese", from_scratch=False):
        super().__init__()
        if from_scratch:
            config = BertConfig.from_pretrained(model_path) if isinstance(model_path, str) and os.path.exists(model_path) else BertConfig(
                vocab_size=21128, hidden_size=768, num_hidden_layers=12,
                num_attention_heads=12, intermediate_size=3072,
            )
            self.bert = BertForMaskedLM(config)
        else:
            self.bert = BertForMaskedLM.from_pretrained(model_path)
        self.config = self.bert.config
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, input_ids, labels=None, attention_mask=None):
        out = self.bert(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        if isinstance(out, tuple):
            return {"loss": out[0], "logits": out[1]}
        return {"loss": out.loss, "logits": out.logits}

    def get_trainable_params(self):
        return []


# ============================================================
# BaselineQKV — 上界（全连接 QKV 天花板）
# ============================================================


class FullRankBertLayer(nn.Module):
    """单层 BERT，QKV 用可训练全连接，FFN 不冻结（参与训练）。

    支持 from_scratch：bert_layer 是随机初始化的 BertLayer。
    """

    def __init__(self, bert_layer, d_model=768, num_heads=12, from_scratch=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # from_scratch: 随机初始化 QKV，否则从 bert_layer 复制预训练权重
        if from_scratch:
            self.Wq = nn.Linear(d_model, d_model, bias=True)
            self.Wk = nn.Linear(d_model, d_model, bias=True)
            self.Wv = nn.Linear(d_model, d_model, bias=True)
        else:
            self.Wq = nn.Linear(d_model, d_model, bias=True)
            self.Wk = nn.Linear(d_model, d_model, bias=True)
            self.Wv = nn.Linear(d_model, d_model, bias=True)
            with torch.no_grad():
                self.Wq.weight.copy_(bert_layer.attention.self.query.weight)
                self.Wq.bias.copy_(bert_layer.attention.self.query.bias)
                self.Wk.weight.copy_(bert_layer.attention.self.key.weight)
                self.Wk.bias.copy_(bert_layer.attention.self.key.bias)
                self.Wv.weight.copy_(bert_layer.attention.self.value.weight)
                self.Wv.bias.copy_(bert_layer.attention.self.value.bias)

        # 借用 BERT FFN/LN（不冻结，参与训练）
        self.attn_output_dense = bert_layer.attention.output.dense
        self.attn_output_dropout = bert_layer.attention.output.dropout
        self.attn_output_LN = bert_layer.attention.output.LayerNorm
        self.inter_dense = bert_layer.intermediate.dense
        self.inter_act = bert_layer.intermediate.intermediate_act_fn
        self.out_dense = bert_layer.output.dense
        self.out_dropout = bert_layer.output.dropout
        self.out_LN = bert_layer.output.LayerNorm

        # 不设置 requires_grad = False → FFN 默认可训练

    def _transpose(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, h, attention_mask=None):
        B, L, D = h.shape
        Q = self.Wq(h)
        K = self.Wk(h)
        V = self.Wv(h)

        Q = self._transpose(Q)
        K = self._transpose(K)
        V = self._transpose(V)

        attn = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)

        attn = self.attn_output_dense(attn)
        attn = self.attn_output_dropout(attn)
        h = self.attn_output_LN(h + attn)

        ffn = self.inter_dense(h)
        ffn = self.inter_act(ffn)
        ffn = self.out_dense(ffn)
        ffn = self.out_dropout(ffn)
        h = self.out_LN(h + ffn)

        return h


class BERTBaselineQKV(nn.Module):
    """
    BERT + 可训练全连接 QKV（支持 from_scratch 随机初始化）+ FFN 不冻结。

    冻结：Embedding、Pooler、cls_head
    不冻结：QKV（Wq/Wk/Wv）+ FFN（intermediate + output.dense + output.LayerNorm）
    支持 num_layers 控制 Transformer 层数。
    """

    def __init__(self, model_path="E:/BaiduNetdiskDownload/bert-base-chinese",
                 from_scratch=False, num_layers=12):
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
            config = self.bert.config

        self.config = self.bert.config
        self.from_scratch = from_scratch
        self.num_layers = num_layers

        # 全冻结，然后单独解冻 FFN + QKV 会用到的参数
        for p in self.bert.parameters():
            p.requires_grad = False

        # 解冻 FFN（每层的 intermediate + output.dense + output.LayerNorm）
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
            FullRankBertLayer(
                bert_layer=self.bert.bert.encoder.layer[i],
                d_model=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                from_scratch=from_scratch,
            )
            for i in range(self.num_layers)
        ])

    def _adjust_num_layers(self, target_layers):
        """截断 BERT 层数（只支持减少，不支持增加）。"""
        current = self.bert.config.num_hidden_layers
        if target_layers > current:
            raise ValueError(
                f"num_layers={target_layers} > pretrained layers={current}，"
                f"请用 from_scratch=True 或从预训练模型初始化后手动扩充。"
            )
        self.bert.bert.encoder.layer = nn.ModuleList(
            self.bert.bert.encoder.layer[:target_layers]
        )
        self.bert.config.num_hidden_layers = target_layers

    def forward(self, input_ids, labels=None, attention_mask=None):
        emb = self.bert.bert.embeddings
        h = emb(input_ids)

        ext_mask = self.bert.bert.get_extended_attention_mask(
            attention_mask, (int(input_ids.shape[0]), int(input_ids.shape[1]))
        )

        for layer in self.layers:
            h = layer(h, attention_mask=ext_mask)

        logits = self.bert.cls(h)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        return {"loss": loss, "logits": logits}

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.bert.eval()
        return self

    def get_trainable_params(self):
        """返回所有可训练参数（QKV Wq/Wk/Wv + FFN）。"""
        params = []
        for layer in self.layers:
            params.extend(layer.Wq.parameters())
            params.extend(layer.Wk.parameters())
            params.extend(layer.Wv.parameters())
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
