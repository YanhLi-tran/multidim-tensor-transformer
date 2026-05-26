#!/bin/bash
# 串行执行剩余实验
# baseline_qkv L12 + tucker L3/L6/L9/L12
# 全部 from_scratch, 50 epochs
# 每个实验自动检测checkpoint续训

set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
DATA="./example.txt"
OUTDIR="./results"

echo "========================================"
echo "实验序列: L12 → Tucker L3 → L6 → L9 → L12"
echo "========================================"

# -------- baseline_qkv L12 --------
echo ""
echo ">>> [1/5] baseline_qkv L12"
LOG="$OUTDIR/baseline_qkv_scratch_L12.log"
python -u train_bert.py \
    --variant baseline_qkv \
    --from_scratch \
    --num_layers 12 \
    --epochs 50 \
    --data_file "$DATA" \
    > "$LOG" 2>&1
echo ">>> baseline_qkv L12 完成"

# -------- tucker L3 --------
echo ""
echo ">>> [2/5] tucker L3"
LOG="$OUTDIR/tucker_scratch_L3.log"
python -u train_bert.py \
    --variant tucker \
    --from_scratch \
    --num_layers 3 \
    --epochs 50 \
    --data_file "$DATA" \
    > "$LOG" 2>&1
echo ">>> tucker L3 完成"

# -------- tucker L6 --------
echo ""
echo ">>> [3/5] tucker L6"
LOG="$OUTDIR/tucker_scratch_L6.log"
python -u train_bert.py \
    --variant tucker \
    --from_scratch \
    --num_layers 6 \
    --epochs 50 \
    --data_file "$DATA" \
    > "$LOG" 2>&1
echo ">>> tucker L6 完成"

# -------- tucker L9 --------
echo ""
echo ">>> [4/5] tucker L9"
LOG="$OUTDIR/tucker_scratch_L9.log"
python -u train_bert.py \
    --variant tucker \
    --from_scratch \
    --num_layers 9 \
    --epochs 50 \
    --data_file "$DATA" \
    > "$LOG" 2>&1
echo ">>> tucker L9 完成"

# -------- tucker L12 --------
echo ""
echo ">>> [5/5] tucker L12"
LOG="$OUTDIR/tucker_scratch_L12.log"
python -u train_bert.py \
    --variant tucker \
    --from_scratch \
    --num_layers 12 \
    --epochs 50 \
    --data_file "$DATA" \
    > "$LOG" 2>&1
echo ">>> tucker L12 完成"

echo ""
echo "========================================"
echo "全部5组实验完成!"
echo "========================================"
