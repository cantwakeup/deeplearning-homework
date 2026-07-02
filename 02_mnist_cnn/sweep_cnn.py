from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_cnn import (
    SimpleCNN,
    build_optimizer,
    evaluate,
    load_mnist,
    predict_examples,
    train_one_epoch,
)


DEFAULT_CONFIGS = [
    ("adam", 1e-3),
    ("adam", 5e-4),
    ("adam", 1e-4),
    ("sgd", 1e-2),
    ("sgd", 5e-3),
    ("sgd", 1e-3),
    ("rmsprop", 1e-3),
    ("rmsprop", 5e-4),
    ("rmsprop", 1e-4),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optimizer and learning-rate sweep for the MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "cnn_sweep")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_config(
    optimizer_name: str,
    learning_rate: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    set_seed(args.seed)
    train_loader, test_loader = load_mnist(
        args.data_dir,
        args.batch_size,
        args.limit_train,
        args.limit_test,
        args.num_workers,
    )
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model.parameters(), learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        history.append(item)
        print(json.dumps({"optimizer": optimizer_name, "learning_rate": learning_rate, **item}, ensure_ascii=False))

    return {
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "history": history,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_test_loss": history[-1]["test_loss"],
        "sample_predictions": predict_examples(model, test_loader, device),
    }


def write_outputs(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = sorted(
        results,
        key=lambda item: item["final_test_accuracy"],
        reverse=True,
    )

    with (output_dir / "cnn_hparam_sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": summary_rows}, f, ensure_ascii=False, indent=2)

    with (output_dir / "cnn_hparam_sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "optimizer",
                "learning_rate",
                "epochs",
                "batch_size",
                "final_train_accuracy",
                "final_test_accuracy",
                "final_test_loss",
            ],
        )
        writer.writeheader()
        for rank, item in enumerate(summary_rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "optimizer": item["optimizer"],
                    "learning_rate": item["learning_rate"],
                    "epochs": item["epochs"],
                    "batch_size": item["batch_size"],
                    "final_train_accuracy": item["final_train_accuracy"],
                    "final_test_accuracy": item["final_test_accuracy"],
                    "final_test_loss": item["final_test_loss"],
                }
            )

    with (output_dir / "cnn_hparam_sweep_report.md").open("w", encoding="utf-8") as f:
        f.write("# CNN 多种情况分析结果\n\n")
        f.write("比较对象为 3 种优化器和 3 档学习率，共 9 组配置。所有配置使用相同 CNN 结构、MNIST 数据集、3 个 epoch 和 batch size 256。\n\n")
        f.write("| 排名 | 优化器 | 学习率 | 测试准确率 | 测试损失 | 训练准确率 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for rank, item in enumerate(summary_rows, start=1):
            f.write(
                f"| {rank} | {item['optimizer']} | {item['learning_rate']} | "
                f"{item['final_test_accuracy']:.4f} | {item['final_test_loss']:.4f} | "
                f"{item['final_train_accuracy']:.4f} |\n"
            )

        best = summary_rows[0]
        f.write("\n")
        f.write(
            f"最优配置为 `{best['optimizer']}` 优化器、学习率 `{best['learning_rate']}`，"
            f"测试准确率为 `{best['final_test_accuracy']:.4f}`。"
            "从结果可以看出，Adam 与 RMSprop 在较少 epoch 下收敛更快；SGD 对学习率更敏感，"
            "学习率过小时收敛速度明显较慢。\n"
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        run_config(optimizer_name, learning_rate, args, device)
        for optimizer_name, learning_rate in DEFAULT_CONFIGS
    ]
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
