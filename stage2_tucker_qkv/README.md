# Stage 2: Tucker 张量分解替代 QKV 全连接

> 把多维分解从 embedding 层延伸到 BERT 的 12 层 QKV 投影

## 动机

Stage 1 证明：只在 embedding 层做多维拆解（仅 98K 参数），对冻结的 BERT 主体影响微乎其微。
自然的下一步：**把多维分解延伸到 Transformer 内部——替换 QKV 全连接投影**。

## 核心改动

```
传统 QKV：
  h ∈ R^{B, L, 768}           # 扁平向量
  Q = h @ Wq                  # Wq ∈ R^{768×768}
  K = h @ Wk
  V = h @ Wv

Tucker QKV（我们的方法）：
  h ∈ R^{B, L, 12, 8, 8}      # reshape 为三维张量 (768=12×8×8)
  Q = TuckerProj(h, Gq, Aq, Bq, Cq, Dq)   # 张量收缩积替代矩阵乘
  K = TuckerProj(h, Gk, Ak, Bk, Ck, Dk)
  V = TuckerProj(h, Gv, Av, Bv, Cv, Dv)
```

## 参数计算（rank=16, d_model=768, reshape=(12,8,8)）

### 单个 TuckerProj

```
A: (12, 16)          =      192    ← 第 1 轴投影
B: (8,  16)          =      128    ← 第 2 轴投影
C: (8,  16)          =      128    ← 第 3 轴投影
G: (16, 16, 16, 16)  =   65,536    ← 核张量 (轴间交互)
D: (768, 16)         =   12,288    ← 输出投影
────────────────────────────────
1 个 TuckerProj       =   78,272    (vs 全连接 768×768 = 589,824)
                                         压缩比: 7.5x
```

### 一层 TuckerQKV（Q + K + V）

```
Wq: 78,272
Wk: 78,272
Wv: 78,272
────────────────
1 层 TuckerQKV = 234,816         (vs 全连接 768×768×3 = 1,769,472)
                                         压缩比: 7.5x
```

### 12 层总计 + Sinkhorn embedding

```
TuckerQKV × 12 层    = 2,817,792
Sinkhorn embedding    =    98,304   ← 从 Stage 1 保留
───────────────────────────────────
Tucker 可训练参数      = 2,916,096
```

## Tucker 计算流程详解

输入形状 `(B, L, 12, 8, 8)` 经过 A/B/C/G/D 五个参数完成 QKV 投影：

### Step 1 — 三轴因子 A / B / C（逐轴线性压缩）

```python
# A: (12, 16) — 压缩第 1 轴 12 → rank
h_a = einsum('b l i j k, i r -> b l r j k', h, A)   # (B, L, 16, 8, 8)

# B: (8, 16)  — 压缩第 2 轴 8 → rank
h_b = einsum('b l r j k, j s -> b l r s k', h_a, B)  # (B, L, 16, 16, 8)

# C: (8, 16)  — 压缩第 3 轴 8 → rank
h_c = einsum('b l r s k, k t -> b l r s t', h_b, C)  # (B, L, 16, 16, 16)
```

A/B/C 各自只负责一个轴，互不干扰。

### Step 2 — 核张量 G（轴间交互）

```python
# G: (16, 16, 16, 16) — 三轴 rank 同时收缩，输出第四个 rank 轴
h_core = einsum('b l r s t, r s t u -> b l u', h_c, G)  # (B, L, 16)
```

G 是整个分解的"中枢"：A/B/C 各自完成轴内压缩后，G 决定三轴之间如何交互，
输出 16 维紧凑表示。参数最多（16⁴ = 65536），但仍远少于全连接的单轴参数。

### Step 3 — 输出矩阵 D（升维回目标空间）

```python
# D: (768, 16) — 从 rank 空间扩展回 768 维
Q = h_core @ D.T   # (B, L, 768)
```

后续与标准 BERT 一致：reshape 为 `(B, heads, L, head_dim=64)` 做多头注意力。

### 为什么优于直接 flatten 做全连接

传统全连接 `h_flat @ Wq`（768→768）在相乘前把三个轴打平，**结构信息全部丢失**。
Tucker 的做法：

- A/B/C 各在自己的语义轴上做压缩（保留轴内结构）
- G 捕捉跨轴的高阶交互
- D 把结果映射回注意力所需空间

代价是 G 有 16⁴ 项，但总参数仍只有全连接的 **1/7.5**。

## 模型严格对比

**共用组件**：BERT FFN / Embedding / LayerNorm — 全部冻结，完全相同  
**唯一变量**：12 层 QKV 的计算方式

| 模型 | QKV 方式 | 可训练参数 | 角色 |
|------|---------|:----:|------|
| **Baseline-Frozen** | 原始 Wq 冻结 | 0 | 随机下界 |
| **Baseline-QKV** | 全连接 Wq 微调 | 21,261,312 | 全量上界 |
| **QKV-Only** | 低秩 A@B^T | 2,820,096 | 等参数二维对照 |
| **Tucker** ★ | Tucker(12,8,8) | 2,916,096 | **我们的方案** |

## 核心文件

| 文件 | 作用 |
|------|------|
| `tucker_qkv.py` | TuckerProj（单投影）+ TuckerQKV（QKV 三路） |
| `tucker_attention.py` | TuckerBertLayer（替换 BERT 单层 attention QKV） |
| `variant_tucker.py` | BERTTuckerVariant（完整模型组装） |
| `qkv_lowrank.py` | LowRankQKV（二维低秩对照组） |
| `variant_qkv_only.py` | BERTQKVOnly（低秩 QKV 模型） |
| `variant_baseline_controls.py` | BERTBaselineFrozen + BERTBaselineQKV（上下界） |

## 实验结果（合成数据, 1 epoch, 5000 样本）

| 模型 | PPL |
|------|----:|
| Baseline-Frozen | — |
| QKV-Only | 263 |
| Tucker | 637 |
| Baseline-QKV | 3.25 |

> **关键发现**：BERT 冻结 FFN 时，QKV 微调对 MLM 帮助极微（上界 PPL 3.25 ≈ 下界 2.71）。
> QKV 分解的价值不能在"冻结 FFN"下评估，这推动了后续两个方向：计算图优化（Stage 3）和从零训练（Stage 4）。

## 结论

- Tucker 分解的 QKV 压缩比约 7.5 倍（2.8M vs 21M）
- 冻结 FFN 时，QKV 微调对 MLM 任务贡献极小
- Step 1 三轴串行收缩存在计算图优化空间：可融合为单条 einsum 或用 opt_einsum 搜索最优路径 → **Stage 3**
- 需要从零训练才能公平评估 QKV 分解的实际效果 → **Stage 4**
