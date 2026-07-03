from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn

from train_resnet import (
    build_optimizer,
    build_resnet_mnist,
    evaluate,
    load_mnist,
    predict_examples,
    set_seed,
    train_one_epoch,
)


RESNET_CONFIGS = [
    # 1. 把 ResNet-50 baseline 放进同一轮 sweep，再改变深度、残差、stem、优化器和学习率。
    {
        "name": "resnet18_mnist_adam_1e-3",
        "factor": "depth",
        "variant": "resnet18",
        "residual": True,
        "stem": "mnist",
        "optimizer": "adam",
        "learning_rate": 1e-3,
    },
    {
        "name": "resnet34_mnist_adam_1e-3",
        "factor": "depth",
        "variant": "resnet34",
        "residual": True,
        "stem": "mnist",
        "optimizer": "adam",
        "learning_rate": 1e-3,
    },
    {
        "name": "resnet50_mnist_adam_1e-3",
        "factor": "depth",
        "variant": "resnet50",
        "residual": True,
        "stem": "mnist",
        "optimizer": "adam",
        "learning_rate": 1e-3,
    },
    {
        "name": "plain18_mnist_adam_1e-3",
        "factor": "residual",
        "variant": "resnet18",
        "residual": False,
        "stem": "mnist",
        "optimizer": "adam",
        "learning_rate": 1e-3,
    },
    {
        "name": "resnet18_imagenet_stem_adam_1e-3",
        "factor": "stem",
        "variant": "resnet18",
        "residual": True,
        "stem": "imagenet",
        "optimizer": "adam",
        "learning_rate": 1e-3,
    },
    {
        "name": "resnet18_mnist_adam_5e-4",
        "factor": "learning_rate",
        "variant": "resnet18",
        "residual": True,
        "stem": "mnist",
        "optimizer": "adam",
        "learning_rate": 5e-4,
    },
    {
        "name": "resnet18_mnist_sgd_1e-2",
        "factor": "optimizer",
        "variant": "resnet18",
        "residual": True,
        "stem": "mnist",
        "optimizer": "sgd",
        "learning_rate": 1e-2,
    },
    {
        "name": "resnet18_mnist_rmsprop_1e-4",
        "factor": "optimizer",
        "variant": "resnet18",
        "residual": True,
        "stem": "mnist",
        "optimizer": "rmsprop",
        "learning_rate": 1e-4,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ResNet depth and architecture sweep on MNIST.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "02_mnist_cnn" / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "resnet_sweep")
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
    # 2. 每个配置重新初始化模型，保证比较的是配置差异而不是上一轮训练残留。
    set_seed(args.seed)
    model = build_resnet_mnist(
        variant=config["variant"],
        residual=config["residual"],
        stem_type=config["stem"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["learning_rate"])

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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
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
    # 3. 按测试准确率排序，并把参数量一起保存，方便讨论“更深是否一定更好”。
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = sorted(results, key=lambda item: item["final_test_accuracy"], reverse=True)

    with (output_dir / "resnet_sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump({"results": summary_rows}, f, ensure_ascii=False, indent=2)

    fields = [
        "rank",
        "name",
        "factor",
        "variant",
        "residual",
        "stem",
        "optimizer",
        "learning_rate",
        "parameter_count",
        "final_train_accuracy",
        "final_test_accuracy",
        "final_test_loss",
    ]
    with (output_dir / "resnet_sweep_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for rank, item in enumerate(summary_rows, start=1):
            writer.writerow({field: item[field] if field != "rank" else rank for field in fields})

    with (output_dir / "resnet_sweep_report.md").open("w", encoding="utf-8") as f:
        f.write("# ResNet-MNIST 深度与结构消融结果\n\n")
        f.write(
            "该实验比较 ResNet-18/34/50、是否保留残差连接、MNIST 友好 stem 与 ImageNet stem，"
            "并对 ResNet-18 补充优化器和学习率对比。\n\n"
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
            f"最优配置为 `{best['name']}`，测试准确率 `{best['final_test_accuracy']:.4f}`。"
            "该结果适合在报告中讨论：MNIST 图像尺寸较小，专门设计的 3x3 stem 通常比 ImageNet 风格 stem 更适配，"
            "残差连接则能作为深层模型稳定训练的关键对照。\n"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 4. 数据加载一次后复用，保证 ResNet 各消融配置面对同一套 MNIST 数据。
    train_loader, test_loader = load_mnist(
        args.data_dir,
        args.batch_size,
        args.limit_train,
        args.limit_test,
        args.num_workers,
    )
    # 5. 顺序运行所有 ResNet 对照实验，最后统一写出 JSON、CSV 和 Markdown 报告。
    results = [run_config(config, args, device, train_loader, test_loader) for config in RESNET_CONFIGS]
    write_outputs(results, args.output_dir)


if __name__ == "__main__":
    main()
