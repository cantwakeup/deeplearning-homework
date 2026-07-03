from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn

from train_cnn import (
    SimpleCNN,
    build_optimizer,
    evaluate,
    load_mnist,
    predict_examples,
    set_seed,
    train_one_epoch,
)


ARCHITECTURE_CONFIGS = [
    # 1. 以 baseline 为参照，后续配置尽量只改变一个结构因素，便于做消融分析。
    {
        "name": "baseline_k3_c32-64_bn_do25_pool",
        "factor": "baseline",
        "channels": (32, 64),
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.25,
        "pooling": True,
    },
    {
        "name": "kernel5_c32-64_bn_do25_pool",
        "factor": "kernel_size",
        "channels": (32, 64),
        "kernel_size": 5,
        "batch_norm": True,
        "dropout": 0.25,
        "pooling": True,
    },
    {
        "name": "narrow_k3_c16-32_bn_do25_pool",
        "factor": "channels",
        "channels": (16, 32),
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.25,
        "pooling": True,
    },
    {
        "name": "wide_k3_c64-128_bn_do25_pool",
        "factor": "channels",
        "channels": (64, 128),
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.25,
        "pooling": True,
    },
    {
        "name": "no_batch_norm_k3_c32-64_do25_pool",
        "factor": "batch_norm",
        "channels": (32, 64),
        "kernel_size": 3,
        "batch_norm": False,
        "dropout": 0.25,
        "pooling": True,
    },
    {
        "name": "dropout0_k3_c32-64_bn_pool",
        "factor": "dropout",
        "channels": (32, 64),
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.0,
        "pooling": True,
    },
    {
        "name": "dropout50_k3_c32-64_bn_pool",
        "factor": "dropout",
        "channels": (32, 64),
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.5,
        "pooling": True,
    },
    {
        "name": "no_pooling_k3_c32-64_bn_do25",
        "factor": "pooling",
        "channels": (32, 64),
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.25,
        "pooling": False,
    },
    {
        "name": "compact_k3_c16-32_bn_do0_pool",
        "factor": "compact",
        "channels": (16, 32),
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.0,
        "pooling": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run architecture ablation for the MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "cnn_arch_sweep")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def run_config(
    config: dict,
    args: argparse.Namespace,
    device: torch.device,
    train_loader,
    test_loader,
) -> dict:
    # 2. 用同一训练流程评估一个结构配置，记录参数量和最终测试表现。
    set_seed(args.seed)
    model = SimpleCNN(
        channels=config["channels"],
        kernel_size=config["kernel_size"],
        batch_norm=config["batch_norm"],
        dropout=config["dropout"],
        pooling=config["pooling"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.learning_rate)

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
        print(json.dumps({"config": config["name"], **item}, ensure_ascii=False))

    return {
        **config,
        "channels": list(config["channels"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "final_test_loss": history[-1]["test_loss"],
        "sample_predictions": predict_examples(model, test_loader, device),
    }


def write_outputs(results: list[dict], output_dir: Path) -> None:
    # 3. 消融结果按测试准确率排序，同时保留配置名、变化因素和参数量。
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = sorted(results, key=lambda item: item["final_test_accuracy"], reverse=True)

    with (output_dir / "cnn_arch_sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": summary_rows}, f, ensure_ascii=False, indent=2)

    fields = [
        "rank",
        "name",
        "factor",
        "channels",
        "kernel_size",
        "batch_norm",
        "dropout",
        "pooling",
        "parameter_count",
        "optimizer",
        "learning_rate",
        "final_train_accuracy",
        "final_test_accuracy",
        "final_test_loss",
    ]
    with (output_dir / "cnn_arch_sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rank, item in enumerate(summary_rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "name": item["name"],
                    "factor": item["factor"],
                    "channels": "-".join(str(value) for value in item["channels"]),
                    "kernel_size": item["kernel_size"],
                    "batch_norm": item["batch_norm"],
                    "dropout": item["dropout"],
                    "pooling": item["pooling"],
                    "parameter_count": item["parameter_count"],
                    "optimizer": item["optimizer"],
                    "learning_rate": item["learning_rate"],
                    "final_train_accuracy": item["final_train_accuracy"],
                    "final_test_accuracy": item["final_test_accuracy"],
                    "final_test_loss": item["final_test_loss"],
                }
            )

    with (output_dir / "cnn_arch_sweep_report.md").open("w", encoding="utf-8") as f:
        f.write("# CNN 架构消融实验结果\n\n")
        f.write(
            "在固定优化器和学习率下，比较卷积核大小、通道宽度、BatchNorm、Dropout 与池化。"
            "该部分用于补充优化器/学习率 sweep，回答哪些结构因素对 MNIST CNN 更敏感。\n\n"
        )
        f.write("| 排名 | 配置 | 消融因素 | 参数量 | 测试准确率 | 测试损失 |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: |\n")
        for rank, item in enumerate(summary_rows, start=1):
            f.write(
                f"| {rank} | `{item['name']}` | {item['factor']} | {item['parameter_count']} | "
                f"{item['final_test_accuracy']:.4f} | {item['final_test_loss']:.4f} |\n"
            )
        best = summary_rows[0]
        f.write("\n")
        f.write(
            f"最优架构为 `{best['name']}`，测试准确率 `{best['final_test_accuracy']:.4f}`。"
            "报告中可结合参数量说明，MNIST 任务上过宽模型未必带来同比例收益，"
            "BatchNorm 与合适 Dropout 更直接影响收敛稳定性。\n"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 4. 数据只加载一次，各结构配置共用同一数据划分，减少非结构因素干扰。
    train_loader, test_loader = load_mnist(
        args.data_dir,
        args.batch_size,
        args.limit_train,
        args.limit_test,
        args.num_workers,
    )
    # 5. 逐个运行卷积核、通道数、BatchNorm、Dropout 和池化的对比配置。
    results = [
        run_config(config, args, device, train_loader, test_loader)
        for config in ARCHITECTURE_CONFIGS
    ]
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
