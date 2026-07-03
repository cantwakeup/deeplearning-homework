from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from mlp_numpy import NumpyMLP, make_spiral_data, standardize_train_test, train_test_split


ACTIVATIONS = ["relu", "tanh", "sigmoid"]
HIDDEN_DIMS = [16, 32, 64, 128]
LEARNING_RATES = [0.1, 0.05, 0.01, 0.005]
OPTIMIZERS = ["gd", "momentum", "adam"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLP hyper-parameter sweep on the spiral dataset.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5, help="Repeat the top K configs with multiple seeds.")
    parser.add_argument("--repeat-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "mlp_sweep")
    return parser.parse_args()


def run_single_config(
    *,
    activation: str,
    hidden_dim: int,
    learning_rate: float,
    optimizer: str,
    seed: int,
    epochs: int,
    log_every: int,
) -> dict:
    # 单组配置的流程与基础实验一致，只是把超参数显式传入，便于批量比较。
    x, y = make_spiral_data(seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=seed)
    x_train, x_test = standardize_train_test(x_train, x_test)

    model = NumpyMLP(
        input_dim=x_train.shape[1],
        hidden_dim=hidden_dim,
        output_dim=3,
        learning_rate=learning_rate,
        activation=activation,
        optimizer=optimizer,
        seed=seed,
    )
    history = model.fit(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=epochs,
        log_every=log_every,
    )
    final = history[-1]
    return {
        "activation": activation,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "seed": seed,
        "epochs": epochs,
        "final_loss": final.loss,
        "final_train_accuracy": final.train_accuracy,
        "final_test_accuracy": final.test_accuracy,
        "history": [
            {
                "epoch": item.epoch,
                "loss": item.loss,
                "train_accuracy": item.train_accuracy,
                "test_accuracy": item.test_accuracy,
            }
            for item in history
        ],
    }


def summarize_repeats(rows: list[dict]) -> list[dict]:
    # 多随机种子复验用于判断 Top 配置是否稳定，而不是只看一次初始化的结果。
    grouped: dict[tuple[str, int, float, str], list[dict]] = {}
    for row in rows:
        key = (row["activation"], row["hidden_dim"], row["learning_rate"], row["optimizer"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (activation, hidden_dim, learning_rate, optimizer), items in grouped.items():
        accuracies = np.array([item["final_test_accuracy"] for item in items], dtype=np.float64)
        losses = np.array([item["final_loss"] for item in items], dtype=np.float64)
        summary.append(
            {
                "activation": activation,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "runs": len(items),
                "mean_test_accuracy": float(accuracies.mean()),
                "std_test_accuracy": float(accuracies.std(ddof=0)),
                "mean_final_loss": float(losses.mean()),
            }
        )
    return sorted(summary, key=lambda item: item["mean_test_accuracy"], reverse=True)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, summary_rows: list[dict], repeat_summary: list[dict]) -> None:
    with (output_dir / "mlp_sweep_report.md").open("w", encoding="utf-8") as f:
        f.write("# NumPy MLP 多因素实验结果\n\n")
        f.write(
            "本实验在三分类螺旋数据集上系统比较激活函数、隐藏层宽度、学习率和优化器。"
            "主实验采用单一随机种子进行全因子搜索，随后对排名靠前的配置进行多随机种子复验。\n\n"
        )
        f.write("## 单种子全因子搜索 Top 10\n\n")
        f.write("| 排名 | 激活函数 | 隐藏层 | 学习率 | 优化器 | 测试准确率 | 训练准确率 | 损失 |\n")
        f.write("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |\n")
        for rank, item in enumerate(summary_rows[:10], start=1):
            f.write(
                f"| {rank} | {item['activation']} | {item['hidden_dim']} | {item['learning_rate']} | "
                f"{item['optimizer']} | {item['final_test_accuracy']:.4f} | "
                f"{item['final_train_accuracy']:.4f} | {item['final_loss']:.4f} |\n"
            )

        if repeat_summary:
            f.write("\n## Top 配置多种子复验\n\n")
            f.write("| 排名 | 激活函数 | 隐藏层 | 学习率 | 优化器 | 平均测试准确率 | 标准差 | 平均损失 |\n")
            f.write("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |\n")
            for rank, item in enumerate(repeat_summary, start=1):
                f.write(
                    f"| {rank} | {item['activation']} | {item['hidden_dim']} | {item['learning_rate']} | "
                    f"{item['optimizer']} | {item['mean_test_accuracy']:.4f} | "
                    f"{item['std_test_accuracy']:.4f} | {item['mean_final_loss']:.4f} |\n"
                )

        best = repeat_summary[0] if repeat_summary else summary_rows[0]
        best_acc_key = "mean_test_accuracy" if repeat_summary else "final_test_accuracy"
        f.write("\n")
        f.write(
            f"综合来看，较优配置为 `{best['activation']}` 激活、隐藏层 `{best['hidden_dim']}`、"
            f"学习率 `{best['learning_rate']}`、`{best['optimizer']}` 优化器，"
            f"测试准确率达到 `{best[best_acc_key]:.4f}`。"
            "该结果可用于报告中说明手写反向传播网络也能通过合理超参数取得稳定分类边界。\n"
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 全因子搜索：激活函数、隐藏层宽度、学习率、优化器全部组合一遍。
    configs = list(itertools.product(ACTIVATIONS, HIDDEN_DIMS, LEARNING_RATES, OPTIMIZERS))
    results = []
    for index, (activation, hidden_dim, learning_rate, optimizer) in enumerate(configs, start=1):
        row = run_single_config(
            activation=activation,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            optimizer=optimizer,
            seed=args.seed,
            epochs=args.epochs,
            log_every=args.log_every,
        )
        results.append(row)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(configs)}",
                    "activation": activation,
                    "hidden_dim": hidden_dim,
                    "learning_rate": learning_rate,
                    "optimizer": optimizer,
                    "test_accuracy": row["final_test_accuracy"],
                },
                ensure_ascii=False,
            )
        )

    # 2. 先按单种子测试准确率排序，再挑出前几名做复验。
    summary_rows = sorted(results, key=lambda item: item["final_test_accuracy"], reverse=True)
    top_configs = summary_rows[: args.top_k]
    repeat_rows = []
    for config in top_configs:
        for seed in args.repeat_seeds:
            repeat_rows.append(
                run_single_config(
                    activation=config["activation"],
                    hidden_dim=config["hidden_dim"],
                    learning_rate=config["learning_rate"],
                    optimizer=config["optimizer"],
                    seed=seed,
                    epochs=args.epochs,
                    log_every=args.log_every,
                )
            )
    repeat_summary = summarize_repeats(repeat_rows)

    # 3. 保存原始结果、CSV 摘要和 Markdown 报告，支撑实验报告中的表格和结论。
    with (args.output_dir / "mlp_sweep_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "full_factorial_results": summary_rows,
                "repeat_results": repeat_rows,
                "repeat_summary": repeat_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary_csv_rows = [
        {
            "rank": rank,
            "activation": item["activation"],
            "hidden_dim": item["hidden_dim"],
            "learning_rate": item["learning_rate"],
            "optimizer": item["optimizer"],
            "epochs": item["epochs"],
            "seed": item["seed"],
            "final_loss": item["final_loss"],
            "final_train_accuracy": item["final_train_accuracy"],
            "final_test_accuracy": item["final_test_accuracy"],
        }
        for rank, item in enumerate(summary_rows, start=1)
    ]
    write_csv(
        args.output_dir / "mlp_sweep_summary.csv",
        summary_csv_rows,
        [
            "rank",
            "activation",
            "hidden_dim",
            "learning_rate",
            "optimizer",
            "epochs",
            "seed",
            "final_loss",
            "final_train_accuracy",
            "final_test_accuracy",
        ],
    )
    write_csv(
        args.output_dir / "mlp_top_seed_summary.csv",
        repeat_summary,
        [
            "activation",
            "hidden_dim",
            "learning_rate",
            "optimizer",
            "runs",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_final_loss",
        ],
    )
    write_report(args.output_dir, summary_rows, repeat_summary)


if __name__ == "__main__":
    main()
