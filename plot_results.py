"""实验数据可视化脚本 - PPL vs 层数 & PPL vs 参数量"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── 中文字体设置 ──
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 实验数据 ──
layers = [3, 6, 9, 12]

baseline_ppl = [597, 1157, 3180, 8355]
tucker_ppl = [3371, 3640, 3531, 7725]

baseline_qkv_params = [5.31, 10.62, 15.93, 21.24]  # million
tucker_qkv_params = [0.69, 1.38, 2.07, 2.76]       # million

# ── Rank sweep 数据（计划） ──
ranks = [4, 8, 12, 16, 24, 32]
rank_params = [0.003, 0.013, 0.037, 0.082, 0.261, 0.606]  # million (L6)

output_dir = os.path.dirname(os.path.abspath(__file__)) + "/results"

# ═══════════════════════════════════════════
# 图 1: PPL vs 层数 (Baseline vs Tucker)
# ═══════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图: PPL vs 层数
ax1.plot(layers, baseline_ppl, 'o-', color='#E74C3C', linewidth=2.5, markersize=9,
         label='BaselineQKV (全连接)', zorder=3)
ax1.plot(layers, tucker_ppl, 's--', color='#2E86C1', linewidth=2.5, markersize=9,
         label='Tucker (rank=16)', zorder=3)

# L3-L9 平台区域标注
ax1.axhspan(3300, 3700, xmin=0.0, xmax=0.65, alpha=0.12, color='#2E86C1')
ax1.annotate('平台效应\nPPL ≈ 3,400-3,600',
             xy=(4.5, 3550), fontsize=10, color='#1A5276',
             ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#D6EAF8', edgecolor='none', alpha=0.8))

# L12 崩塌标注
ax1.annotate('深度崩塌\nL12=7,725',
             xy=(12, 7725), fontsize=10, color='#E74C3C',
             ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='none', alpha=0.8))

# 每个点标值（L9: 3180/3531 位置互换避免重叠）
for x, y in zip(layers, baseline_ppl):
    if x == 9:
        offset = -18  # Baseline L9=3,180 放点下方
    else:
        offset = -15 if y < 2000 else 15
    ax1.annotate(f'{y:,}', (x, y), textcoords="offset points",
                 xytext=(0, offset), ha='center', fontsize=8, color='#E74C3C')
for x, y in zip(layers, tucker_ppl):
    if x == 9:
        offset = 15   # Tucker L9=3,531 放点上方
    else:
        offset = 15 if y > 7000 else -18
    ax1.annotate(f'{y:,}', (x, y), textcoords="offset points",
                 xytext=(0, offset), ha='center', fontsize=8, color='#2E86C1')

ax1.set_xlabel('Transformer 层数', fontsize=12, fontweight='bold')
ax1.set_ylabel('最佳 PPL (越低越好)', fontsize=12, fontweight='bold')
ax1.set_title('PPL vs 层数对比', fontsize=14, fontweight='bold')
ax1.set_xticks(layers)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.grid(True, alpha=0.12, linestyle='--', which='minor')
ax1.set_yscale('linear')
ax1.set_ylim(500, 8500)
ax1.set_yticks(np.arange(1000, 9000, 1000))
ax1.minorticks_on()
ax1.tick_params(axis='y', which='minor', length=3)

# 右图: QKV 参数量 vs 层数
ax2.bar(np.array(layers) - 0.15, baseline_qkv_params, 0.3, color='#E74C3C', alpha=0.85,
        label='BaselineQKV', zorder=3)
ax2.bar(np.array(layers) + 0.15, tucker_qkv_params, 0.3, color='#2E86C1', alpha=0.85,
        label='Tucker (rank=16)', zorder=3)

for i, (x, v) in enumerate(zip(layers, baseline_qkv_params)):
    ax2.text(x - 0.15, v + 0.3, f'{v:.1f}M', ha='center', fontsize=8, color='#E74C3C', fontweight='bold')
for i, (x, v) in enumerate(zip(layers, tucker_qkv_params)):
    ax2.text(x + 0.15, v + 0.3, f'{v:.2f}M', ha='center', fontsize=8, color='#2E86C1', fontweight='bold')

ax2.set_xlabel('Transformer 层数', fontsize=12, fontweight='bold')
ax2.set_ylabel('QKV 参数量 (百万)', fontsize=12, fontweight='bold')
ax2.set_title('QKV 参数量 vs 层数 (压缩比 ~7.5×)', fontsize=14, fontweight='bold')
ax2.set_xticks(layers)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout(pad=2)
fig.savefig(output_dir + '/fig1_ppl_vs_layers.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✅ fig1_ppl_vs_layers.png 已保存")

# ═══════════════════════════════════════════
# 图 2: PPL vs QKV 参数量（效率曲线）
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

# 合并为 (param, ppl, label) 对
points = []
for i in range(4):
    points.append((baseline_qkv_params[i], baseline_ppl[i], f'Baseline L{layers[i]}', '#E74C3C'))
    points.append((tucker_qkv_params[i], tucker_ppl[i], f'Tucker L{layers[i]}', '#2E86C1'))

# 手工偏移表：避免 Tucker L3/L6/L9 参数值太近导致标注重叠
# x 为正=向右, 负=向左; y 为正=向上, 负=向下
_offset = {
    'Tucker L3':   (18, 30),
    'Tucker L6':   (18, -28),
    'Tucker L9':   (-40, 0),
    'Tucker L12':  (15, -15),
    'Baseline L3': (15, 20),
    'Baseline L6': (15, 20),
    'Baseline L9': (-35, -15),
    'Baseline L12': (15, -15),
}
# 被 Tucker L6 遮挡的 Tucker L9: ha='right' 向左展开

# 按参数量排序绘制
points.sort(key=lambda x: x[0])
for param, ppl, label, color in points:
    marker = 'o' if 'Baseline' in label else 's'
    ax.scatter(param, ppl, c=color, s=140, marker=marker, zorder=5, edgecolors='white', linewidth=1.5)
    ox, oy = _offset[label]
    ha = 'right' if ox < 0 else 'left'
    ax.annotate(f'{label}\n({param:.2f}M, PPL={ppl:,})',
                (param, ppl), textcoords="offset points",
                xytext=(ox, oy),
                ha=ha, fontsize=8, color=color, fontweight='bold')

# 画 Tucker 趋势线
ax.plot([tucker_qkv_params[0], tucker_qkv_params[2]],
        [tucker_ppl[0], tucker_ppl[2]],
        '--', color='#2E86C1', alpha=0.4, linewidth=1.5)
# Baseline 趋势线
ax.plot(baseline_qkv_params, baseline_ppl,
        '--', color='#E74C3C', alpha=0.4, linewidth=1.5)

ax.set_xlabel('QKV 参数量 (百万)', fontsize=12, fontweight='bold')
ax.set_ylabel('最佳 PPL (越低越好)', fontsize=12, fontweight='bold')
ax.set_title('参数效率对比：PPL vs QKV 参数量', fontsize=14, fontweight='bold')
ax.set_yscale('linear')
ax.set_ylim(0, 9000)
ax.grid(True, alpha=0.3, linestyle='--')

# 压缩比标注
ax.annotate('Tucker 用 1/7.5 的参数\nL3-L9 保持 PPL 稳定',
            xy=(0.5, 0.15), xycoords='axes fraction',
            fontsize=11, fontweight='bold', color='#1A5276',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#D6EAF8',
                      edgecolor='#2E86C1', alpha=0.9))

plt.tight_layout()
fig.savefig(output_dir + '/fig2_ppl_vs_params.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✅ fig2_ppl_vs_params.png 已保存")

# ═══════════════════════════════════════════
# 图 3: Tucker 层数 PPL (单独聚焦)
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))

colors = ['#2471A3', '#2E86C1', '#3498DB', '#E74C3C']
for i, (l, p) in enumerate(zip(layers, tucker_ppl)):
    is_outlier = (l == 12)
    ax.bar(l, p, 0.5, color=colors[i], alpha=0.85, zorder=3,
           edgecolor='white', linewidth=1.5)
    ax.text(l, p + 150, f'{p:,}', ha='center', fontsize=11,
            color=colors[i], fontweight='bold')

# 平台线
ax.axhline(y=3560, xmin=0.08, xmax=0.62, color='#2E86C1', linestyle='--',
           linewidth=1.5, alpha=0.5)
ax.text(4.5, 3660, '平台效应区域\nPPL ≈ 3,531±109', ha='center', fontsize=10,
        color='#1A5276', fontweight='bold')

ax.set_xlabel('Transformer 层数', fontsize=12, fontweight='bold')
ax.set_ylabel('最佳 PPL', fontsize=12, fontweight='bold')
ax.set_title('Tucker rank=16: 层数扩展性', fontsize=14, fontweight='bold')
ax.set_xticks(layers)
ax.set_ylim(0, 8500)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
fig.savefig(output_dir + '/fig3_tucker_layers_detail.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✅ fig3_tucker_layers_detail.png 已保存")

# ═══════════════════════════════════════════
# 图 4: Rank sweep 参数效率预览
# ═══════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))

# 当前已知：rank=16, PPL=3,640
known_rank = 16
known_ppl = 3640

bar_colors = ['#D4E6F1', '#A9CCE3', '#7FB3D8', '#2E86C1', '#1F6F9F', '#154360']
ax.bar(ranks, rank_params, 0.6, color=bar_colors, alpha=0.85,
       edgecolor='white', linewidth=1.5, zorder=3)

for r, c in zip(ranks, rank_params):
    ax.text(r, c + max(rank_params) * 0.06, f'{c:.3f}M', ha='center', fontsize=9,
            color='#1A5276', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7))

ax.set_xlabel('Tucker Rank', fontsize=12, fontweight='bold')
ax.set_ylabel('L6 QKV 参数量 (百万)', fontsize=12, fontweight='bold')
ax.set_title('Rank Sweep 计划: L6 QKV 参数量 vs Rank', fontsize=14, fontweight='bold')
ax.set_xticks(ranks)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Baseline 参考标注（参数量远超 rank sweep 范围，放在图内提示）
ax.annotate(f'Baseline L6: {baseline_qkv_params[1]:.1f}M\n(Tucker 最大 rank=32 仅 0.6M)',
            xy=(0.98, 0.95), xycoords='axes fraction',
            ha='right', va='top', fontsize=9, color='#E74C3C', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C', alpha=0.85))

plt.tight_layout()
fig.savefig(output_dir + '/fig4_rank_sweep_preview.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✅ fig4_rank_sweep_preview.png 已保存")

print("\n🎉 全部图表生成完毕！")
print(f"   输出目录: {output_dir}")
for f in os.listdir(output_dir):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"   📊 {f} ({size/1024:.0f} KB)")
