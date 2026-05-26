#!/bin/bash
# Stage 3: Tucker Rank Sweep — 固定 L=6, 扫描 rank=4/8/12/16/24/32
# 串行训练，每次 50 epochs
# 用法: bash run_rank_sweep.sh

set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PYTHON="E:/Anaconda/python.exe"
DATA="../example.txt"
OUTPUT="./results"

RANKS=(4 8 12 16 24 32)
TOTAL=${#RANKS[@]}

for i in "${!RANKS[@]}"; do
    R=${RANKS[$i]}
    IDX=$((i + 1))
    echo ""
    echo "===== [$IDX/$TOTAL] Tucker L6 rank=$R ====="
    $PYTHON -u ../train_bert.py \
        --variant tucker \
        --from_scratch \
        --num_layers 6 \
        --tucker_rank $R \
        --epochs 50 \
        --batch_size 4 \
        --lr 2e-4 \
        --data_file "$DATA" \
        --output_dir "$OUTPUT" \
        > "$OUTPUT/tucker_scratch_L6_rank${R}.log" 2>&1
    echo "Tucker L6 rank=$R 完成"
done

echo ""
echo "===== 全部 Rank Sweep 完成 ====="
echo "Rank 序列: ${RANKS[*]}"
echo "结果目录: $OUTPUT"
