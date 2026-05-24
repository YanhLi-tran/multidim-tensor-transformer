# Multidim Tensor Transformer

> 将 Transformer hidden_size 从二维升级为多维结构化张量
>
> 从 embedding 层的多维拆解开始，到 Tucker 张量分解替代 QKV 全连接

---

## 实验路径

| 阶段 | 内容 | 发现 |
|------|------|------|
| **[Stage 1](stage1_embedding/)** | Embedding 层 Sinkhorn 置换 + reshape(12,8,8) + 轴向 Dropout | 只改 embedding 不够，需延伸到 QKV |
| **[Stage 2](stage2_tucker_qkv/)** | Tucker 分解替换 12 层 QKV + 四模型严格对比 | 冻结 FFN 时 QKV 微调贡献极小；Step 1 串行收缩存在优化空间 |
| **Stage 3** | Tucker Step 1+2 计算图优化：einsum 融合 + opt_einsum 最优路径 | 即将开源 |
| **Stage 4** | 自建 3/6/12 层 mini Decoder 从零训练，公平评估 Tucker vs 全连接 | 即将开源 |

---

## 核心思想

```
传统 Transformer：
  h ∈ R^{B, L, 768}          扁平向量
  Q = h @ Wq                 矩阵乘 (Wq ∈ R^{768×768})

我们的方法：
  h ∈ R^{B, L, 12, 8, 8}     多维张量 (768 = 12×8×8)
  Q = TuckerProj(h)           张量收缩积 (A,B,C,G,D)
                               参数 78K vs 全连接 589K (7.5x 压缩)
```

## 项目结构

```
stage1_embedding/      ← 第一阶段: Embedding 层多维拆解
stage2_tucker_qkv/     ← 第二阶段: Tucker QKV + 四模型对比
stage3_einsum_opt/     ← 第三阶段: Step 1+2 计算图优化 (即将添加)
stage4_mini_decoder/   ← 第四阶段: 从零训练 mini Decoder (即将添加)
```
