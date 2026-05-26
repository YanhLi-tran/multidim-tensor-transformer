"""
训练脚本：在文本语料上对 BERT Baseline / Variant 进行 MLM 微调。

MLM 任务：
- 随机 mask 掉 15% 的 token
- 其中 80% 替换为 [MASK]、10% 替换为随机 token、10% 保持不变
- 仅对 masked 位置计算 loss
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

import argparse
import time
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
from tqdm import tqdm

from models import (BERTBaseline, BERTMultidimVariant, BERTTuckerVariant,
                       BERTQKVOnly, BERTBaselineFrozen, BERTBaselineQKV)


# ============================================================
# MLM 数据处理
# ============================================================

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    对 input_ids 执行 BERT 风格的 masking。

    返回:
        input_ids: 带 mask 的输入
        labels: 仅 masked 位置保留原始 token id，其余为 -100
    """
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # 不对 special tokens 做 mask
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # 非 masked 的位置 label = -100
    labels[~masked_indices] = -100

    # 80% → [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% → 随机 token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        len(tokenizer), labels.shape, dtype=torch.long
    )
    inputs[indices_random] = random_words[indices_random]

    # 剩余 10% → 保持不变

    return inputs, labels


def get_chinese_char_ids(tokenizer):
    """
    从 BERT vocab.txt 中提取所有单中文字符的 token id。

    BERT-base-chinese 的 vocab 结构：
      行 0: [PAD]
      行 1-99: [unusedX]
      行 100-104: [UNK], [CLS], [SEP], [MASK], special tokens
      行 105-106: <S>, <T>
      行 107+: 标点符号、英文字符、数字等
      行 106-8118: 真实中文字符 + 部分英文/符号
      行 8119+: ##开头的 subword tokens

    策略：取 token 文本只含单个汉字的 id（Unicode范围: 0x4E00-0x9FFF）
    """
    import re
    chinese_char_pattern = re.compile(r'^[\u4e00-\u9fff]$')

    char_ids = []
    for i in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(i)
        if chinese_char_pattern.match(token):
            char_ids.append(i)

    print(f"vocab 中共 {len(tokenizer)} 个 token，提取到 {len(char_ids)} 个单中文字符")
    return char_ids


def prepare_dataset(tokenizer, max_length=128, max_samples=None, use_synthetic=False,
                    data_file=None):
    """
    准备文本数据集。

    优先级：本地文件 > HuggingFace 数据集 > 合成数据
    """
    # 1) 本地文件（最高优先级）
    if data_file and os.path.exists(data_file):
        print(f"加载本地文件: {data_file} ({os.path.getsize(data_file)//1024}KB)")
        with open(data_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 按段落切分，过滤太短的
        paragraphs = [p.strip() for p in raw_text.split("\n") if len(p.strip()) >= 20]
        print(f"  切分得到 {len(paragraphs)} 段文本")

        if max_samples and len(paragraphs) > max_samples:
            paragraphs = paragraphs[:max_samples]

        tokenized_inputs = {"input_ids": [], "attention_mask": []}
        for text in tqdm(paragraphs, desc="Tokenizing"):
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            tokenized_inputs["input_ids"].append(encoded["input_ids"][0])
            tokenized_inputs["attention_mask"].append(encoded["attention_mask"][0])

        return Dataset.from_dict(tokenized_inputs)

    # 2) HuggingFace 数据集
    dataset = None

    # 尝试真实语料
    if not use_synthetic:
        for name, kwargs in [
            ("shibing624/nlp_zh", {"split": "train", "trust_remote_code": True}),
            ("wikitext", {"name": "wikitext-2-raw-v1", "split": "train", "trust_remote_code": True}),
        ]:
            try:
                print(f"尝试加载: {name}...")
                dataset = load_dataset(name, **kwargs)
                text_field = "text" if "text" in dataset.column_names else dataset.column_names[0]
                print(f"成功! 使用 {name}")
                break
            except Exception as e:
                print(f"  失败: {type(e).__name__}")
                continue

    # 合成数据回退（vocab-based 伪中文）
    if dataset is None or use_synthetic:
        print("使用合成数据（vocab 中文字符）")
        char_ids = get_chinese_char_ids(tokenizer)
        n = max_samples or 1000

        # 加权：高频中文字符（位置靠前）更常被选中
        # BERT vocab 大致按频率排序，所以用指数衰减权重
        weights = torch.exp(-torch.linspace(0, 2, len(char_ids)))

        input_ids_list = []
        attention_mask_list = []
        for _ in range(n):
            seq_len = max_length
            # 按权重采样中文字符
            sampled_indices = torch.multinomial(weights, seq_len, replacement=True)
            ids = torch.tensor([char_ids[idx] for idx in sampled_indices])
            input_ids_list.append(ids)
            attention_mask_list.append(torch.ones(seq_len))

        return Dataset.from_dict({
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
        })

    # 真实语料处理
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    tokenized_inputs = {"input_ids": [], "attention_mask": []}

    for i in tqdm(range(len(dataset)), desc="Tokenizing"):
        text = dataset[i].get(text_field, "")
        if not text or len(str(text).strip()) < 10:
            continue
        encoded = tokenizer(
            str(text),
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        if encoded["input_ids"].sum() == 0:
            continue
        tokenized_inputs["input_ids"].append(encoded["input_ids"][0])
        tokenized_inputs["attention_mask"].append(encoded["attention_mask"][0])

    return Dataset.from_dict(tokenized_inputs)


def collate_mlm_batch(batch, tokenizer, mlm_probability=0.15):
    """整理 batch 并执行 MLM masking。"""
    # 确保每个 item 是 tensor（兼容 list）
    input_ids = torch.stack([
        item["input_ids"] if isinstance(item["input_ids"], torch.Tensor)
        else torch.tensor(item["input_ids"])
        for item in batch
    ])
    attention_mask = torch.stack([
        item["attention_mask"] if isinstance(item["attention_mask"], torch.Tensor)
        else torch.tensor(item["attention_mask"])
        for item in batch
    ])

    # MLM masking
    input_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ============================================================
# 训练 & 评估
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, max_steps=None):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not sys.stdout.isatty())
    for step, batch in enumerate(pbar):
        if max_steps and step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        loss = outputs["loss"]
        if loss is None:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "ppl": f"{torch.exp(torch.tensor(loss.item())).item():.2f}"
        })

    avg_loss = total_loss / max(num_batches, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


@torch.no_grad()
def evaluate(model, dataloader, device, max_steps=None):
    """评估 perplexity。"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for step, batch in enumerate(tqdm(dataloader, desc="Eval", disable=not sys.stdout.isatty())):
        if max_steps and step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        loss = outputs["loss"]
        if loss is not None:
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BERT 多维 Embedding 实验训练")
    parser.add_argument("--variant", type=str, default="baseline_frozen",
                        choices=[
                            "baseline_frozen",   # 全冻结，下界
                            "baseline_qkv",      # 全连接 QKV 可训练，上界
                            "qkv_only",          # 低秩 QKV，对照组
                            "tucker",            # Tucker 分解 QKV，你的方案
                        ])
    parser.add_argument("--model_path", type=str,
                        default="E:/BaiduNetdiskDownload/bert-base-chinese")
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--dropout_axis", type=int, default=2)
    parser.add_argument("--tucker_rank", type=int, default=16,
                        help="Tucker 分解的 rank")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练 epoch 数（数据少时需要更多 epoch）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="最多使用多少样本（快速验证）")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_eval_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成数据（离线验证）")
    parser.add_argument("--data_file", type=str, default=None,
                        help="本地文本文件路径（优先级最高）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练（checkpoint 路径）")
    parser.add_argument("--save_every", type=int, default=5,
                        help="每隔几个 epoch 保存一次 checkpoint（默认5）")
    parser.add_argument("--from_scratch", action="store_true",
                        help="从零随机初始化，不加载预训练权重")
    parser.add_argument("--num_layers", type=int, default=12,
                        choices=[3, 6, 9, 12],
                        help="Transformer 层数（3/6/9/12）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 tokenizer
    print(f"加载 tokenizer: {args.model_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # 准备数据集
    print("准备数据集...")
    dataset = prepare_dataset(
        tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
        use_synthetic=args.synthetic,
        data_file=args.data_file,
    )

    # 划分 train/val
    split_idx = int(len(dataset) * 0.9)
    train_dataset = dataset.select(range(split_idx))
    eval_dataset = dataset.select(range(split_idx, len(dataset)))

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_mlm_batch(batch, tokenizer),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_mlm_batch(batch, tokenizer),
    )
    # ============================================================
    # 创建模型
    # ============================================================
    print(f"\n创建模型: {args.variant} (from_scratch={args.from_scratch}, layers={args.num_layers})...")
    if args.variant == "baseline_frozen":
        model = BERTBaselineFrozen(model_path=args.model_path, from_scratch=args.from_scratch)
    elif args.variant == "baseline_qkv":
        model = BERTBaselineQKV(
            model_path=args.model_path,
            from_scratch=args.from_scratch,
            num_layers=args.num_layers,
        )
    elif args.variant == "qkv_only":
        model = BERTQKVOnly(model_path=args.model_path, rank=51, num_layers=args.num_layers)
    elif args.variant == "tucker":
        model = BERTTuckerVariant(
            model_path=args.model_path,
            rank=args.tucker_rank,
            dropout_p=args.dropout_p,
            sinkhorn_tau=1.0,
            from_scratch=args.from_scratch,
            num_layers=args.num_layers,
        )
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    model = model.to(device)

    # ============================================================
    # checkpoint / results 路径（含 from_scratch 和 num_layers 标识）
    # ============================================================
    tag = args.variant
    if args.from_scratch:
        tag += "_scratch"
    tag += f"_L{args.num_layers}"

    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 1
    results = {
        "variant": args.variant,
        "from_scratch": args.from_scratch,
        "num_layers": args.num_layers,
        "epochs": [],
    }
    best_ppl = float("inf")

    ckpt_path = os.path.join(args.output_dir, f"bert_{tag}_checkpoint.pt")
    results_path = os.path.join(args.output_dir, f"bert_{tag}_results.json")

    # 打印参数量
    trainable = sum(p.numel() for p in model.get_trainable_params())
    total = sum(p.numel() for p in model.parameters())
    print(f"  总参数: {total:,}")
    print(f"  可训练: {trainable:,}")

    variant_labels = {
        "baseline_frozen": "Baseline-Frozen (下界)",
        "baseline_qkv":    f"Baseline-QKV  (上界, {trainable//1000000}M QKV)",
        "qkv_only":        f"QKV-Only      (低秩对照, {trainable//1000000}M)",
        "tucker":          f"Tucker QKV    (你的方案, {trainable//1000000}M)",
    }
    print(f"  角色: {variant_labels.get(args.variant, args.variant)}")

    # ============================================================
    # 恢复 checkpoint
    # ============================================================
    if args.resume and os.path.exists(args.resume):
        print(f"\n从 checkpoint 恢复: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ppl = ckpt.get("best_ppl", float("inf"))
        results = ckpt.get("results", results)
        print(f"  恢复到 Epoch {start_epoch}, Best PPL={best_ppl:.1f}")
    elif os.path.exists(ckpt_path):
        print(f"\n发现自动保存的 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_ppl = ckpt.get("best_ppl", float("inf"))
        results = ckpt.get("results", results)
        print(f"  恢复到 Epoch {start_epoch}, Best PPL={best_ppl:.1f}")

    # ============================================================
    # Optimizer / Scheduler
    # ============================================================
    # baseline_frozen 不需要 optimizer（没参数）
    if trainable == 0:
        print("  [INFO] 无可训练参数，仅评估")
        optimizer = None
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=args.lr)
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        elif os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        total_steps = args.epochs * len(train_loader)
        if args.max_train_steps:
            total_steps = min(total_steps, args.max_train_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

    # ============================================================
    # 训练循环
    # ============================================================
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()

        if trainable == 0:
            # baseline_frozen：只评估，不训练
            eval_loss, eval_ppl = evaluate(
                model, eval_loader, device, max_steps=args.max_eval_steps,
            )
            train_ppl = eval_ppl  # 无训练，直接用 eval
        else:
            train_loss, train_ppl = train_epoch(
                model, train_loader, optimizer, scheduler, device, epoch,
                max_steps=args.max_train_steps,
            )
            eval_loss, eval_ppl = evaluate(
                model, eval_loader, device, max_steps=args.max_eval_steps,
            )

        elapsed = time.time() - start_time

        epoch_result = {
            "epoch": epoch,
            "train_ppl": train_ppl,
            "eval_ppl": eval_ppl,
            "time": elapsed,
        }
        results["epochs"].append(epoch_result)

        if sys.stdout.isatty():
            print(f"\nEpoch {epoch}: Train PPL={train_ppl:.2f}, Eval PPL={eval_ppl:.2f}, Time={elapsed:.1f}s")

        if eval_ppl < best_ppl:
            best_ppl = eval_ppl
            # 保存最佳模型（单独文件，不覆盖）
            best_path = os.path.join(args.output_dir, f"bert_{tag}_best.pt")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_ppl": best_ppl}, best_path)
            if sys.stdout.isatty():
                print(f"  🎯 新的最佳模型! PPL={best_ppl:.2f} → {best_path}")

        # 每 epoch 保存 checkpoint（覆盖模式，用于续训）
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_ppl": best_ppl,
            "results": results,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, ckpt_path)
        if sys.stdout.isatty():
            print(f"  💾 Checkpoint 已保存 (epoch {epoch}): {ckpt_path}")

    # ============================================================
    # 训练完成，保存最终结果
    # ============================================================
    results["best_eval_ppl"] = best_ppl
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== 训练完成 ===")
    print(f"最佳 Eval PPL: {best_ppl:.2f}")
    print(f"结果: {results_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"下次运行: python train_bert.py --variant {args.variant} --from_scratch --num_layers {args.num_layers} 即可自动续训")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
