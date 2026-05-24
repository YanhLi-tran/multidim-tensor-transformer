"""
Baseline: 纯 BERT-base-chinese，不做任何修改。

任务：Masked Language Modeling (MLM)
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM


class BERTBaseline(nn.Module):
    """标准 BERT-base-chinese 包装器。"""

    def __init__(self, model_path="E:/BaiduNetdiskDownload/bert-base-chinese"):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(model_path)
        self.config = self.bert.config

    def forward(self, input_ids, labels=None, attention_mask=None):
        """
        Args:
            input_ids: (B, L), 部分位置已被 mask
            labels: (B, L), -100 表示不计算 loss
            attention_mask: (B, L)
        Returns:
            dict with keys: loss, logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        # transformers 5.x 返回 tuple (loss, logits); 兼容 dict
        if isinstance(outputs, tuple):
            return {"loss": outputs[0], "logits": outputs[1]}
        return {"loss": outputs.loss, "logits": outputs.logits}

    @property
    def device(self):
        return next(self.parameters()).device
