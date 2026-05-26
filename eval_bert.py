"""
评估脚本：评估 BERT Baseline / Variant 在 MLM 任务上的表现。

同时可视化维度置换矩阵 P。
"""

import argparse
import json
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from models import BERTBaseline, BERTMultidimVariant
from train_bert import prepare_dataset, collate_mlm_batch


@torch.no_grad()
def evaluate_ppl(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
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
    return ppl


@torch.no_grad()
def visualize_permutation(model, save_path):
    """可视化置换矩阵 P。"""
    device = next(model.parameters()).device
    model.eval()

    if not hasattr(model, "multidim_embedding"):
        print("Baseline 模型没有置换矩阵，跳过可视化")
        return

    dummy_hidden = torch.randn(1, 32, 768).to(device)
    _, P = model.multidim_embedding.sinkhorn(dummy_hidden)
    P = P.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im = axes[0].imshow(P, cmap="viridis", aspect="auto")
    axes[0].set_title("Permutation Matrix P (full 768×768)", fontsize=14)
    axes[0].set_xlabel("Output Dimension")
    axes[0].set_ylabel("Input Dimension")
    plt.colorbar(im, ax=axes[0])

    zoom = min(64, P.shape[0])
    im2 = axes[1].imshow(P[:zoom, :zoom], cmap="viridis", aspect="auto")
    axes[1].set_title(f"Permutation Matrix P (zoom {zoom}×{zoom})", fontsize=14)
    axes[1].set_xlabel("Output Dimension")
    axes[1].set_ylabel("Input Dimension")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"置换矩阵可视化已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="BERT 多维 Embedding 评估")
    parser.add_argument("--model_path", type=str,
                        default="E:/BaiduNetdiskDownload/bert-base-chinese")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的 checkpoint 路径")
    parser.add_argument("--variant", type=str, default="variant",
                        choices=["baseline", "variant"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    print("准备评估数据...")
    dataset = prepare_dataset(tokenizer, max_length=args.max_length, max_samples=args.max_samples)
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_mlm_batch(batch, tokenizer),
    )

    if args.variant == "baseline":
        model = BERTBaseline(model_path=args.model_path)
    else:
        model = BERTMultidimVariant(model_path=args.model_path)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    ppl = evaluate_ppl(model, eval_loader, device)
    print(f"\nPerplexity: {ppl:.2f}")

    result = {"checkpoint": args.checkpoint, "ppl": ppl}
    result_path = os.path.join(args.output_dir, f"bert_{args.variant}_eval.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if args.visualize and args.variant == "variant":
        viz_path = os.path.join(args.output_dir, f"bert_{args.variant}_permutation.png")
        visualize_permutation(model, viz_path)


if __name__ == "__main__":
    main()
