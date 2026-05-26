import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, '.')

import torch
from models.variant_baseline_controls import BERTBaselineQKV
from models.variant_tucker import BERTTuckerVariant

print("=== 参数数量检查 ===")

m1 = BERTBaselineQKV(model_path='E:/BaiduNetdiskDownload/bert-base-chinese')
p1 = sum(p.numel() for p in m1.get_trainable_params())
t1 = sum(p.numel() for p in m1.parameters())
print(f"BERTBaselineQKV: 可训练={p1:,}, 总参数={t1:,}")

m2 = BERTTuckerVariant(model_path='E:/BaiduNetdiskDownload/bert-base-chinese', rank=16)
p2 = sum(p.numel() for p in m2.get_trainable_params())
t2 = sum(p.numel() for p in m2.parameters())
print(f"BERTTuckerVariant: 可训练={p2:,}, 总参数={t2:,}")

# 验证 FFN 确实可训练
print("\n=== FFN 参数 requires_grad 检查 (BaselineQKV layer 0) ===")
layer0 = m1.layers[0]
for name, p in layer0.named_parameters():
    if 'inter' in name or 'out_dense' in name:
        print(f"  {name}: requires_grad={p.requires_grad}")

print("\n=== FFN 参数 requires_grad 检查 (Tucker layer 0) ===")
tucker_layer0 = m2.tucker_layers[0]
for name, p in tucker_layer0.named_parameters():
    pass  # TuckerBertLayer 结构不同，检查 bert 主模型
for name, p in m2.bert.named_parameters():
    if 'intermediate' in name or 'output.dense' in name:
        print(f"  {name}: requires_grad={p.requires_grad}")

print("\nDone.")
