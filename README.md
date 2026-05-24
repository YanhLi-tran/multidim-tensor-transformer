# Multidim Tensor Transformer

> 将 Transformer hidden_size 从二维升级为多维结构化张量
>
> 从 embedding 层的多维拆解开始，到 Tucker 张量分解替代 QKV 全连接

---

## 实验路径

| 阶段 | 内容 | 发现 |
|------|------|------|
| **[Stage 1](stage1_embedding/)** | Embedding 层 Sinkhorn 置换 + reshape(12,8,8) + 轴向 Dropout | 只改 embedding 不够，需要动 QKV |
| **Stage 2** | Tucker 张量分解替换全部 QKV 投影 | 即将开源 |
| **Stage 3** | 自己搭建 mini Decoder 从零训练对比 | 即将开源 |

---

## 项目结构

```
stage1_embedding/    ← Embedding 层多维拆解
stage2_tucker_qkv/   ← Tucker 分解 QKV（待添加）
stage3_mini_decoder/ ← 从零训练对比（待添加）
```
