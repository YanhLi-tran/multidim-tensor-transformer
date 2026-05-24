# Stage 1: Embedding 层多维拆解

> 第一个验证步骤——只在 Embedding 层将 hidden_size 拆成多维张量

## 动机

传统 BERT 的 token embedding 是扁平的 768 维向量。我们在 embedding 出口加一个模块：

```
BERT Embedding (B, L, 768)
     ↓
Sinkhorn 可学习维度置换     ← 让模型自己学"哪 768 个维度应该相邻"
     ↓
reshape → (B, L, 12, 8, 8)  ← 拆成三维张量
     ↓
轴向 Dropout               ← 训练时随机丢弃某些 chunk
     ↓
reshape back → (B, L, 768)  ← 拍回二维
     ↓
送入原始 BERT Encoder
```

## 核心文件

| 文件 | 作用 |
|------|------|
| `sinkhorn.py` | 可学习维度置换矩阵（Sinkhorn 归一化） |
| `axial_dropout.py` | 轴向结构化 Dropout（按 chunk 丢弃） |
| `multidim_embedding.py` | 核心包装器：置换 → reshape → dropout → 还原 |
| `baseline_bert.py` | BERT baseline（不改动） |
| `variant_bert.py` | BERT + 多维 embedding 变体 |
| `train_bert.py` | MLM 训练脚本 |

## 结论

只改 embedding 层（约 98K 可训练参数），BERT 主体冻结时，效果微乎其微。因为 QKV 和 FFN 的全连接占据主导。

这引导我们进入 **Stage 2**——把多维分解延伸到 QKV 投影。
