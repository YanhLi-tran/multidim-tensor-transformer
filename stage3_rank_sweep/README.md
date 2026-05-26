# Stage 3: Tucker Rank Sweep

## 研究问题

Tucker 分解的 rank 是决定模型容量的核心超参数。Stage 2 固定 rank=16 发现了**平台效应**（L3-L9 PPL 波动仅 3,371-3,640），说明 rank 是统一容量天花板。

**Stage 3 要回答**：最优 rank 是多少？PPL 随 rank 增长的效率曲线是怎样的？

## 实验设计

| 参数 | 值 |
|---|---|
| 变体 | `tucker` |
| 固定层数 | **L=6**（Stage 2 中 Tucker 表现最好的层数，PPL=3,640） |
| 扫描 rank | **4, 8, 12, 16, 24, 32** |
| 初始化 | `--from_scratch`（全部从零随机初始化） |
| Epochs | 50 |
| Batch size | 4 |
| Learning rate | 2e-4 |
| 数据 | `example.txt`（2261 段中文财经新闻） |

## 预期参数量范围

| Rank | QKV 参数量 |
|---|---|
| 4 | ~0.003M |
| 8 | ~0.024M |
| 12 | ~0.080M |
| 16 | ~0.189M |
| 24 | ~0.294M |
| 32 | ~0.606M |

对比：Baseline L6 的 QKV 全连接参数量 = **10.6M**

## 假设

1. **低 rank (4-8)**：容量不足，PPL 显著高于 Stage 2 rank=16
2. **中 rank (12-16)**：接近容量天花板，PPL 接近最优
3. **高 rank (24-32)**：容量过剩，PPL 应与 rank=16 持平（平台延伸），但参数量增加

## 预期产物

- PPL vs Rank 曲线图
- PPL vs 参数量效率曲线（标注 Baseline 参考线）
- 最优 rank 推荐（性价比拐点）

## 运行方式

```bash
cd stage3_rank_sweep
bash run_rank_sweep.sh
```

串行训练 rank=4 → 8 → 12 → 16 → 24 → 32，每次 50 epochs。
全部完成后运行 `plot_results.py`（需更新以包含 rank sweep 数据）生成图表。

## 文件命名

- Checkpoint: `results/bert_tucker_scratch_L6_rank{N}_checkpoint.pt`
- Best model: `results/bert_tucker_scratch_L6_rank{N}_best.pt`
- Log: `results/tucker_scratch_L6_rank{N}.log`
- Results JSON: `results/bert_tucker_scratch_L6_rank{N}_results.json`
