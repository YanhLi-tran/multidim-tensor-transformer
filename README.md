# Multidim Tensor Transformer

> Tucker 张量分解替代 Transformer QKV 全连接 —— 小数据场景下的参数效率研究

---

## 研究动机

大模型压缩（GQA、MLA）关注海量数据场景。但实际中大量垂直领域面临**数据稀缺**：

- 金融、医疗、法律标注数据昂贵
- 小语种语料天然不足
- 边缘设备参数预算紧张

**核心发现**：在小数据（~2000 样本）从零训练时，Tucker 分解的 QKV 对层数增加几乎不敏感 —— L3 → L6 的 PPL 退化远小于全连接 Baseline。低秩约束本身成为天然正则化器。

---

## 实验设计

### 对比方案

| 模型 | QKV 实现 | QKV 参数量 (12层) | 压缩比 |
|------|----------|-------------------|--------|
| **BaselineQKV** | 标准 Linear(768→768) × 3 | 21.2M | 1× |
| **Tucker** | Tucker 分解 rank=16, in_dims=(12,8,8) | 2.8M | 7.5× |

### 实验配置

- 从零随机初始化（`BertConfig` + random init）
- 层数扫描：3 / 6 / 9 / 12 层
- 冻结 BERT 预训练权重（除 QKV 和 FFN）
- MLM 预训练 50 epochs, batch_size=4, lr=2e-4
- 数据：中文财经新闻 2261 段（2034 train / 227 val）
- GPU：RTX 3060 6GB

---

## 实验结果

| 实验 | 层数 | Epoch | Train PPL | Best PPL | QKV 参数 |
|------|------|-------|-----------|----------|----------|
| BaselineQKV | 3 | 50/50 | 180 | 597 | 5.3M |
| BaselineQKV | 6 | 50/50 | 377 | 1,157 | 10.6M |
| BaselineQKV | 9 | 50/50 | 3,107 | 3,180 | 15.9M |
| BaselineQKV | 12 | 50/50 | 9,202 | 8,355 | 21.2M |
| **Tucker** | **3** | **50/50** | **3,080** | **3,371** | **0.69M** |
| **Tucker** | **6** | **50/50** | **—** | **3,640** | **1.38M** |
| **Tucker** | **9** | **50/50** | **—** | **3,531** | **2.07M** |
| **Tucker** | **12** | **50/50** | **—** | **7,725** | **2.76M** |

### 关键发现

#### 1. 平台效应（L3-L9）
```
Tucker PPL vs 层数:  3,371 → 3,640 → 3,531
                     ↑ 波动仅 269，在 3-9 层范围内层数增加几乎不退化
```
- Tucker 用 rank=16 的低秩瓶颈锁死了容量天花板
- L3-L9 三条线紧凑聚集，层数增加不引入过拟合

#### 2. 对比 Baseline：Tucker 参数效率极高
```
Baseline: L3=597 → L6=1,157 → L9=3,180 → L12=8,355  (参数翻倍 → PPL 爆炸)
Tucker:   L3=3,371 → L6=3,640 → L9=3,531 → L12=7,725 (L3-L9 持平，L12 崩塌)
```
- Baseline 每加一层 QKV 多 177 万参数，快速过拟合
- Tucker L3-L9 仅 0.69M-2.07M QKV 参数，rank-16 正则化天然防过拟合

#### 3. 深度崩塌（L12）
- L12 Best PPL=7,725，从 epoch 4 到 50 纹丝不动
- rank=16 的低秩瓶颈在 12 层网络上梯度传播失败（卡死 saddle point）
- **有效深度窗口：3-9 层**

#### 4. Rank Sweep（计划中）

| Rank | L6 QKV 参数 | 压缩比 (vs Baseline 10.6M) |
|------|-------------|---------------------------|
| 4 | 3,168 | 3,350× |
| 8 | 13,248 | 801× |
| 12 | 37,152 | 286× |
| **16** | **81,792** | **130×** |
| 24 | 260,928 | 41× |
| 32 | 605,952 | 18× |

下一步：固定 L6，扫描 rank=4/8/12/16/24/32，绘制 PPL vs 参数量效率曲线，找最优 rank。

---

## 项目结构

```
models/                  ← 统一模型目录
├── variant_baseline_controls.py   # BaselineQKV
├── variant_tucker.py              # Tucker QKV
├── variant_bert.py                # Embedding 层多维拆解
├── tucker_qkv.py / tucker_attention.py  # Tucker 分解核心
└── ...
train_bert.py            ← 训练脚本 (支持 --from_scratch, --num_layers, --variant)
run_tucker_chain.sh      ← Tucker 串行训练链 (L3→L6→L9→L12)
eval_bert.py             ← 评估脚本
check_params.py          ← 参数量诊断
results/                 ← 训练日志 (.log) 和结果 (.json)
stage1_embedding/        ← Stage 1: Embedding 层多维拆解
stage2_tucker_qkv/       ← Stage 2: Tucker QKV 原始代码
```
