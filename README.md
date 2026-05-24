# Multidim Tensor Transformer

> 将 Transformer hidden_size 从二维升级为多维结构化张量 — 概念验证

## 核心想法

传统 Transformer 中每个 token 表示为 `(seq_len, hidden_size)` 的二维张量。
我们将 `hidden_size` 在 embedding 层拆分为多维结构（如 `4×4×4×4`），
然后用 Tucker 张量分解替代 QKV 的全连接矩阵乘法。

## 状态

🚧 即将开源 — 代码整理中
