#!/bin/bash
# Tucker 系列串行训练: L3 -> L6 -> L9 -> L12
# 用法: bash run_tucker_chain.sh

set -e
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PYTHON="E:/Anaconda/python.exe"
DATA="./example.txt"

echo "===== [1/4] Tucker L3 ====="
$PYTHON -u train_bert.py --variant tucker --from_scratch --num_layers 3 --epochs 50 --data_file "$DATA" > results/tucker_scratch_L3.log 2>&1
echo "Tucker L3 完成"

echo "===== [2/4] Tucker L6 ====="
$PYTHON -u train_bert.py --variant tucker --from_scratch --num_layers 6 --epochs 50 --data_file "$DATA" > results/tucker_scratch_L6.log 2>&1
echo "Tucker L6 完成"

echo "===== [3/4] Tucker L9 ====="
$PYTHON -u train_bert.py --variant tucker --from_scratch --num_layers 9 --epochs 50 --data_file "$DATA" > results/tucker_scratch_L9.log 2>&1
echo "Tucker L9 完成"

echo "===== [4/4] Tucker L12 ====="
$PYTHON -u train_bert.py --variant tucker --from_scratch --num_layers 12 --epochs 50 --data_file "$DATA" > results/tucker_scratch_L12.log 2>&1
echo "Tucker L12 完成"

echo "===== 全部完成 ====="
