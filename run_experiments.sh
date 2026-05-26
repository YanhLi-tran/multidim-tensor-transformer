#!/bin/bash
# 批量实验：BaselineQKV vs Tucker，从零初始化，遍历 3/6/9/12 层
# 每 epoch 保存 checkpoint（覆盖），保存最佳模型

set -e

cd "$(dirname "$0")"
PY="python"
RESULTS="./results"
DATA="./example.txt"
# 如果相对路径不存在，尝试其他地方
if [ ! -f "$DATA" ]; then
  DATA="D:/PythonProject/python_example/AI课程练习/example.txt"
fi
EPOCHS=50
BATCH=4
LR=2e-4

mkdir -p "$RESULTS"

echo "========================================================"
echo "  批量实验：共 8 组"
echo "  Epochs=$EPOCHS Batch=$BATCH LR=$LR Device=cuda"
echo "========================================================"
START_ALL=$(date +%s)

for variant in baseline_qkv tucker; do
  for layers in 3 6 9 12; do
    TAG="${variant}_scratch_L${layers}"
    LOG="${RESULTS}/${TAG}.log"
    CKPT="${RESULTS}/bert_${TAG}_checkpoint.pt"

    echo ""
    echo "===== [$variant L=${layers}] ====="
    echo "日志: $LOG"

    START_ONE=$(date +%s)

    RESUME=""
    if [ -f "$CKPT" ]; then
      RESUME="--resume $CKPT"
      echo "  [续训] $CKPT"
    fi

    # 运行训练（-u = unbuffered）
    python -u train_bert.py \
      --variant "$variant" \
      --from_scratch \
      --num_layers "$layers" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH" \
      --lr "$LR" \
      --save_every 1 \
      --data_file "$DATA" \
      --output_dir "$RESULTS" \
      --device cuda \
      $RESUME \
      > "$LOG" 2>&1

    RC=$?
    ELAPSED=$(( $(date +%s) - START_ONE ))
    if [ $RC -eq 0 ]; then
      echo "  ✅ 完成! 耗时=${ELAPSED}s"
    else
      echo "  ❌ 失败! RC=$RC 耗时=${ELAPSED}s"
      echo "  最后 20 行日志:"
      tail -20 "$LOG"
    fi
  done
done

TOTAL=$(( $(date +%s) - START_ALL ))
echo ""
echo "========================================================"
echo "  全部完成! 总耗时=${TOTAL}s = $((TOTAL/60))min"
echo "========================================================"
