from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mlp_numpy import NumpyMLP, make_spiral_data, standardize_train_test, train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy MLP on a spiral dataset.")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--optimizer", choices=["gd", "momentum", "adam"], default="gd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 准备数据：生成螺旋数据、划分训练测试集，并做标准化。
    x, y = make_spiral_data(seed=args.seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=args.seed)
    x_train, x_test = standardize_train_test(x_train, x_test)

    # 2. 建立手写 MLP，并按命令行传入的隐藏层、激活函数和优化器训练。
    model = NumpyMLP(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=3,
        learning_rate=args.learning_rate,
        activation=args.activation,
        optimizer=args.optimizer,
        seed=args.seed,
    )
    history = model.fit(x_train, y_train, x_test, y_test, epochs=args.epochs)

    # 3. 汇总最终指标和样例预测，便于报告引用。
    final_metrics = history[-1]
    result = {
        "dataset": "synthetic three-class spiral",
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "activation": args.activation,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "final_loss": final_metrics.loss,
        "final_train_accuracy": final_metrics.train_accuracy,
        "final_test_accuracy": final_metrics.test_accuracy,
        "sample_predictions": [
            {"true": int(true), "pred": int(pred)}
            for true, pred in zip(y_test[:10], model.predict(x_test[:10]))
        ],
    }

    # 4. 保存结构化结果和训练曲线，后续画图、写报告都从这些文件读取。
    with (args.output_dir / "mlp_numpy_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "loss_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_accuracy", "test_accuracy"])
        writer.writeheader()
        for item in history:
            writer.writerow(
                {
                    "epoch": item.epoch,
                    "loss": item.loss,
                    "train_accuracy": item.train_accuracy,
                    "test_accuracy": item.test_accuracy,
                }
            )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
