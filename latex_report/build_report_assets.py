from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-deepl-homework")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value: str | float | int, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_acc(value: str | float | int) -> str:
    number = to_float(value)
    if math.isnan(number):
        return "--"
    return f"{number * 100:.2f}%"


def fmt_acc_text(value: str | float | int) -> str:
    return fmt_acc(value).replace("%", "\\%")


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def save_table(path: Path, headers: list[str], rows: list[list[object]], aligns: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"\\begin{{tabular}}{{{aligns}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(latex_escape(cell) for cell in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        }
    )


def cnn_arch_label(row: dict[str, str], rank: int | None = None) -> str:
    prefix = f"C{rank}: " if rank is not None else ""
    channels = row.get("channels", "").replace("-", "/")
    kernel = row.get("kernel_size", "")
    bn = "BN" if row.get("batch_norm") == "True" else "NoBN"
    dropout = row.get("dropout", "")
    pooling = "Pool" if row.get("pooling") == "True" else "NoPool"
    return f"{prefix}k={kernel}, {channels}, {bn}, D={dropout}, {pooling}"


def resnet_label(row: dict[str, str], rank: int | None = None) -> str:
    prefix = f"R{rank}: " if rank is not None else ""
    variant = row.get("variant", "").replace("resnet", "ResNet-")
    if row.get("residual") == "False":
        variant = variant.replace("ResNet-", "Plain-")
    stem = "MNIST" if row.get("stem") == "mnist" else "ImageNet"
    optimizer = row.get("optimizer", "")
    lr = row.get("learning_rate", "")
    return f"{prefix}{variant}, {stem}, {optimizer} {lr}"


def plot_mlp_optimizer_lr(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    lrs = sorted({to_float(row["learning_rate"]) for row in rows}, reverse=True)
    optimizers = sorted({row["optimizer"] for row in rows})
    plt.figure(figsize=(6.2, 3.6))
    for optimizer in optimizers:
        y_values = []
        for lr in lrs:
            candidates = [
                to_float(row["final_test_accuracy"])
                for row in rows
                if row["optimizer"] == optimizer and abs(to_float(row["learning_rate"]) - lr) < 1e-12
            ]
            y_values.append(max(candidates) if candidates else math.nan)
        plt.plot(lrs, y_values, marker="o", label=optimizer)
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Learning rate")
    plt.ylabel("Best test accuracy")
    plt.title("MLP optimizer and learning-rate sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mlp_optimizer_lr.png")
    plt.close()


def plot_mlp_activation_hidden(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    activations = ["relu", "tanh", "sigmoid"]
    hidden_dims = sorted({int(row["hidden_dim"]) for row in rows})
    matrix = np.zeros((len(activations), len(hidden_dims)), dtype=np.float64)
    for i, activation in enumerate(activations):
        for j, hidden_dim in enumerate(hidden_dims):
            candidates = [
                to_float(row["final_test_accuracy"])
                for row in rows
                if row["activation"] == activation and int(row["hidden_dim"]) == hidden_dim
            ]
            matrix[i, j] = max(candidates) if candidates else np.nan
    plt.figure(figsize=(6.2, 3.1))
    image = plt.imshow(matrix, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), cmap="viridis")
    plt.colorbar(image, label="Best test accuracy")
    plt.xticks(range(len(hidden_dims)), hidden_dims)
    plt.yticks(range(len(activations)), activations)
    plt.xlabel("Hidden dimension")
    plt.ylabel("Activation")
    plt.title("MLP activation and width ablation")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mlp_activation_hidden.png")
    plt.close()


def plot_mlp_decision_boundary(best_row: dict[str, str]) -> None:
    if not best_row:
        return
    sys.path.insert(0, str(ROOT / "01_numpy_mlp"))
    from mlp_numpy import NumpyMLP, make_spiral_data, standardize_train_test, train_test_split

    seed = 42
    x, y = make_spiral_data(seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=seed)
    x_train_std, x_test_std = standardize_train_test(x_train, x_test)
    model = NumpyMLP(
        input_dim=2,
        hidden_dim=int(best_row["hidden_dim"]),
        output_dim=3,
        learning_rate=to_float(best_row["learning_rate"]),
        activation=best_row["activation"],
        optimizer=best_row["optimizer"],
        seed=seed,
    )
    model.fit(x_train_std, y_train, x_test_std, y_test, epochs=1000, log_every=250)

    all_x = np.vstack([x_train_std, x_test_std])
    x_min, x_max = all_x[:, 0].min() - 0.6, all_x[:, 0].max() + 0.6
    y_min, y_max = all_x[:, 1].min() - 0.6, all_x[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(4.8, 4.2))
    plt.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5, 2.5], cmap="viridis")
    plt.scatter(x_train_std[:, 0], x_train_std[:, 1], c=y_train, s=14, cmap="viridis", edgecolor="none", alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("MLP decision boundary")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mlp_decision_boundary.png")
    plt.close()


def plot_cnn_hparam(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    optimizers = ["adam", "sgd", "rmsprop"]
    lrs = sorted({to_float(row["learning_rate"]) for row in rows}, reverse=True)
    matrix = np.full((len(optimizers), len(lrs)), np.nan, dtype=np.float64)
    for row in rows:
        i = optimizers.index(row["optimizer"])
        j = lrs.index(to_float(row["learning_rate"]))
        matrix[i, j] = to_float(row["final_test_accuracy"])
    plt.figure(figsize=(6.0, 3.3))
    image = plt.imshow(matrix, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), cmap="magma")
    plt.colorbar(image, label="Test accuracy")
    plt.xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
    plt.yticks(range(len(optimizers)), optimizers)
    plt.xlabel("Learning rate")
    plt.ylabel("Optimizer")
    plt.title("CNN optimizer and learning-rate sweep")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "cnn_optimizer_lr_heatmap.png")
    plt.close()


def plot_ranked_accuracy_bars(
    rows: list[dict[str, str]],
    value_field: str,
    title: str,
    filename: str,
    label_kind: str,
) -> None:
    if not rows:
        return
    ranked_rows = list(enumerate(rows, start=1))
    rows_sorted = sorted(ranked_rows, key=lambda item: to_float(item[1][value_field]))
    if label_kind == "cnn":
        labels = [f"C{rank}  {row['factor']}" for rank, row in rows_sorted]
        color = "#2F6F9F"
    else:
        labels = [f"R{rank}  {row['factor']}" for rank, row in rows_sorted]
        color = "#5A7D3A"
    values = [to_float(row[value_field]) for _, row in rows_sorted]
    fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.38 * len(rows_sorted))))
    ax.barh(range(len(rows_sorted)), values, color=color, height=0.62)
    ax.set_yticks(range(len(rows_sorted)), labels)
    ax.set_xlabel("Test accuracy")
    ax.set_title(title)
    ax.set_xlim(min(values) - 0.004, min(1.0, max(values) + 0.0035))
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    for i, value in enumerate(values):
        ax.text(value + 0.00035, i, f"{value:.4f}", va="center", fontsize=8)
    fig.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.16)
    fig.savefig(FIGURE_DIR / filename)
    plt.close(fig)


def plot_horizontal_accuracy(rows: list[dict[str, str]], label_field: str, value_field: str, title: str, filename: str) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda row: to_float(row[value_field]))
    labels = [row[label_field].replace("_", " ") for row in rows_sorted]
    values = [to_float(row[value_field]) for row in rows_sorted]
    plt.figure(figsize=(8.0, max(3.4, 0.45 * len(rows_sorted))))
    plt.barh(range(len(rows_sorted)), values, color="#3572A5")
    plt.yticks(range(len(rows_sorted)), labels, fontsize=7)
    plt.xlabel("Test accuracy")
    plt.title(title)
    for i, value in enumerate(values):
        plt.text(value + 0.0005, i, f"{value:.4f}", va="center", fontsize=8)
    plt.xlim(min(values) - 0.01, min(1.0, max(values) + 0.01))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename)
    plt.close()


def plot_resnet_params(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for rank, row in enumerate(rows, start=1):
        params_m = to_float(row["parameter_count"]) / 1_000_000.0
        acc = to_float(row["final_test_accuracy"])
        ax.scatter(params_m, acc, s=58)
        x_offset = 6 if rank in {7, 8} else 5
        y_offset = 6 if rank % 2 else -11
        ax.annotate(
            f"R{rank}",
            (params_m, acc),
            textcoords="offset points",
            xytext=(x_offset, y_offset),
            fontsize=8,
            weight="bold",
        )
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("ResNet parameter count vs accuracy")
    ax.set_ylim(min(to_float(row["final_test_accuracy"]) for row in rows) - 0.002, 0.995)
    ax.grid(alpha=0.25)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(FIGURE_DIR / "resnet_params_accuracy.png")
    plt.close(fig)


def build_tables(
    mlp_repeat: list[dict[str, str]],
    mlp_rows: list[dict[str, str]],
    cnn_hparam: list[dict[str, str]],
    cnn_arch: list[dict[str, str]],
    resnet_rows: list[dict[str, str]],
) -> None:
    mlp_source = mlp_repeat if mlp_repeat else mlp_rows
    mlp_rows_tex = []
    for row in mlp_source[:6]:
        if "mean_test_accuracy" in row:
            mlp_rows_tex.append(
                [
                    row["activation"],
                    row["hidden_dim"],
                    row["learning_rate"],
                    row["optimizer"],
                    fmt_acc(row["mean_test_accuracy"]),
                    f"{to_float(row['std_test_accuracy']) * 100:.2f}%",
                ]
            )
        else:
            mlp_rows_tex.append(
                [
                    row["activation"],
                    row["hidden_dim"],
                    row["learning_rate"],
                    row["optimizer"],
                    fmt_acc(row["final_test_accuracy"]),
                    "--",
                ]
            )
    save_table(
        TABLE_DIR / "mlp_top.tex",
        ["激活", "隐藏层", "学习率", "优化器", "测试准确率", "标准差"],
        mlp_rows_tex,
        "llllll",
    )

    save_table(
        TABLE_DIR / "cnn_hparam_top.tex",
        ["优化器", "学习率", "训练准确率", "测试准确率", "测试损失"],
        [
            [
                row["optimizer"],
                row["learning_rate"],
                fmt_acc(row["final_train_accuracy"]),
                fmt_acc(row["final_test_accuracy"]),
                f"{to_float(row['final_test_loss']):.4f}",
            ]
            for row in cnn_hparam[:6]
        ],
        "lllll",
    )

    save_table(
        TABLE_DIR / "cnn_arch_top.tex",
        ["编号与配置", "消融因素", "参数量", "测试准确率", "测试损失"],
        [
            [
                cnn_arch_label(row, rank),
                row["factor"],
                row["parameter_count"],
                fmt_acc(row["final_test_accuracy"]),
                f"{to_float(row['final_test_loss']):.4f}",
            ]
            for rank, row in enumerate(cnn_arch[:7], start=1)
        ],
        "lllll",
    )

    save_table(
        TABLE_DIR / "resnet_top.tex",
        ["编号与配置", "因素", "参数量", "优化器", "学习率", "测试准确率"],
        [
            [
                resnet_label(row, rank),
                row["factor"],
                row["parameter_count"],
                row["optimizer"],
                row["learning_rate"],
                fmt_acc(row["final_test_accuracy"]),
            ]
            for rank, row in enumerate(resnet_rows[:8], start=1)
        ],
        "llllll",
    )


def best_or_empty(rows: list[dict[str, str]]) -> dict[str, str]:
    return rows[0] if rows else {}


def build_main_tex(
    mlp_repeat: list[dict[str, str]],
    mlp_rows: list[dict[str, str]],
    cnn_hparam: list[dict[str, str]],
    cnn_arch: list[dict[str, str]],
    resnet_rows: list[dict[str, str]],
) -> None:
    mlp_best = best_or_empty(mlp_repeat or mlp_rows)
    cnn_hparam_best = best_or_empty(cnn_hparam)
    cnn_arch_best = best_or_empty(cnn_arch)
    resnet_best = best_or_empty(resnet_rows)

    mlp_metric = mlp_best.get("mean_test_accuracy", mlp_best.get("final_test_accuracy", ""))
    tex = rf"""\documentclass[UTF8]{{ctexart}}
\usepackage[a4paper,margin=2.4cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{amsmath}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}

\title{{深度学习课程实验报告}}
\author{{}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
本报告围绕课程任务书中的三个实验展开：纯 NumPy 实现多层感知机、基于 PyTorch 的 MNIST 卷积神经网络、以及面向 MNIST 的 ResNet。除完成基础训练和预测流程外，本文进一步补充了较系统的超参数搜索与结构消融实验，包括学习率、优化器、激活函数、隐藏层宽度、卷积核大小、通道数、BatchNorm、Dropout、池化、ResNet 深度、残差连接和 stem 设计等因素。实验结果以 JSON、CSV、Markdown、图表和 LaTeX 表格形式保留，便于复现与汇报。
\end{{abstract}}

\section{{实验目标与组织}}
任务书要求按三个文件夹分别提交程序：\texttt{{01\_numpy\_mlp}}、\texttt{{02\_mnist\_cnn}}、\texttt{{03\_resnet\_mnist}}。本仓库保持这一结构，并将综合报告材料集中保存在 \texttt{{latex\_report}}，其中 \texttt{{figures}} 保存图像，\texttt{{tables}} 保存 LaTeX 表格。

\section{{实验环境}}
实验使用项目内隔离环境 \texttt{{.conda-env}}，Python 版本为 3.10，深度学习框架为 PyTorch 2.4.0 与 torchvision 0.19.0。CNN 与 ResNet 实验在 GPU 上训练，NumPy MLP 使用 CPU 完成全批量梯度更新。所有实验默认固定随机种子，并在输出文件中记录训练配置。

\section{{实验一：NumPy MLP}}
\subsection{{方法}}
第一题仅使用 NumPy 实现输入层、隐藏层、输出层、激活函数、Softmax 交叉熵损失和反向传播。基础版本使用 ReLU 和普通梯度下降；扩展实验比较 ReLU、Tanh、Sigmoid 三类激活函数，隐藏层宽度 16/32/64/128，学习率 0.1/0.05/0.01/0.005，以及 GD、Momentum、Adam 三类优化方式。排名靠前的配置进一步使用多个随机种子复验，降低偶然性。

\subsection{{结果}}
MLP 最优配置为 \texttt{{{latex_escape(mlp_best.get("activation", "--"))}}} 激活、隐藏层 \texttt{{{latex_escape(mlp_best.get("hidden_dim", "--"))}}}、学习率 \texttt{{{latex_escape(mlp_best.get("learning_rate", "--"))}}}、\texttt{{{latex_escape(mlp_best.get("optimizer", "--"))}}} 优化器，测试准确率为 {fmt_acc_text(mlp_metric)}。

\begin{{table}}[H]
\centering
\caption{{MLP 超参数搜索靠前配置}}
\input{{tables/mlp_top.tex}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{figures/mlp_optimizer_lr.png}}
\caption{{MLP 不同优化器与学习率下的最佳测试准确率}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{figures/mlp_activation_hidden.png}}
\caption{{MLP 激活函数和隐藏层宽度消融}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.58\linewidth]{{figures/mlp_decision_boundary.png}}
\caption{{MLP 在螺旋数据集上的分类边界示例}}
\end{{figure}}

\section{{实验二：MNIST CNN}}
\subsection{{方法}}
第二题使用 PyTorch 搭建两层卷积网络完成 MNIST 分类。为满足“分析多种情况并保留分析结果”的要求，实验分为两组：第一组比较 Adam、SGD、RMSprop 与多档学习率；第二组在固定优化策略下比较卷积核大小、通道宽度、BatchNorm、Dropout 和池化。

\subsection{{结果}}
优化器/学习率搜索中，最佳组合为 \texttt{{{latex_escape(cnn_hparam_best.get("optimizer", "--"))}}}，学习率 \texttt{{{latex_escape(cnn_hparam_best.get("learning_rate", "--"))}}}，测试准确率 {fmt_acc_text(cnn_hparam_best.get("final_test_accuracy", ""))}。架构消融中，最佳配置为 \texttt{{{latex_escape(cnn_arch_best.get("name", "--"))}}}，测试准确率 {fmt_acc_text(cnn_arch_best.get("final_test_accuracy", ""))}。

\begin{{table}}[H]
\centering
\caption{{CNN 优化器与学习率搜索 Top 配置}}
\input{{tables/cnn_hparam_top.tex}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{CNN 架构消融 Top 配置}}
\input{{tables/cnn_arch_top.tex}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{figures/cnn_optimizer_lr_heatmap.png}}
\caption{{CNN 优化器与学习率热力图}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{figures/cnn_arch_ablation.png}}
\caption{{CNN 架构消融测试准确率，C 编号对应上方表格}}
\end{{figure}}

\section{{实验三：ResNet-MNIST}}
\subsection{{方法}}
第三题实现面向 MNIST 的 ResNet。基础提交保持 ResNet-50；扩展实验比较 ResNet-18、ResNet-34、ResNet-50，进一步加入无残差 Plain-18、ImageNet 风格 stem、不同优化器和学习率。这样既能覆盖任务书建议的 50 层残差网络，也能说明深度、残差连接和输入分辨率适配对结果的影响。

\subsection{{结果}}
ResNet sweep 中最佳配置为 \texttt{{{latex_escape(resnet_best.get("name", "--"))}}}，测试准确率 {fmt_acc_text(resnet_best.get("final_test_accuracy", ""))}，参数量为 \texttt{{{latex_escape(resnet_best.get("parameter_count", "--"))}}}。

\begin{{table}}[H]
\centering
\caption{{ResNet 深度与结构消融结果}}
\input{{tables/resnet_top.tex}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\linewidth]{{figures/resnet_ablation.png}}
\caption{{ResNet 不同配置测试准确率，R 编号对应上方表格}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.80\linewidth]{{figures/resnet_params_accuracy.png}}
\caption{{ResNet 参数量与测试准确率关系，R 编号对应上方表格}}
\end{{figure}}

\section{{综合分析与汇报思路}}
汇报时建议先说明三部分递进关系：NumPy MLP 体现前向传播、损失函数和反向传播的底层实现；CNN 体现局部感受野和参数共享在图像识别上的效果；ResNet 体现残差连接对深层网络训练稳定性的作用。随后按“基础实现是否满足任务书要求、扩展对比实验如何设计、结果说明什么”展开。

从结果看，MNIST 上浅层 CNN 已经能够达到较高准确率，ResNet 的主要价值不只是单点准确率，而是通过深度、残差连接和 stem 对比展示深层网络设计原则。MLP 实验则显示，在非线性二维数据上，激活函数和优化器会显著影响收敛速度与分类边界质量。

\section{{复现方式}}
\begin{{verbatim}}
.conda-env/bin/python 01_numpy_mlp/run_experiment.py --epochs 1500
.conda-env/bin/python 01_numpy_mlp/sweep_mlp.py --epochs 1000
.conda-env/bin/python 02_mnist_cnn/train_cnn.py --epochs 3 --batch-size 256
.conda-env/bin/python 02_mnist_cnn/sweep_cnn.py --epochs 3 --batch-size 256 --num-workers 4
.conda-env/bin/python 02_mnist_cnn/sweep_arch_cnn.py --epochs 3 --batch-size 256 --num-workers 4
.conda-env/bin/python 03_resnet_mnist/train_resnet.py --epochs 3 --batch-size 256
.conda-env/bin/python 03_resnet_mnist/sweep_resnet.py --epochs 3 --batch-size 256 --num-workers 4
.conda-env/bin/python latex_report/build_report_assets.py
\end{{verbatim}}

\section{{结论}}
三个实验均完成了任务书目标，并进一步保留了多种实验条件下的结构化结果。报告材料能够支持课堂汇报和后续本地 LaTeX 编译；代码输出中的 JSON/CSV 文件可作为实验可复现证据。

\end{{document}}
"""
    (REPORT_DIR / "main.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot()

    mlp_rows = read_csv(ROOT / "01_numpy_mlp" / "outputs" / "mlp_sweep" / "mlp_sweep_summary.csv")
    mlp_repeat = read_csv(ROOT / "01_numpy_mlp" / "outputs" / "mlp_sweep" / "mlp_top_seed_summary.csv")
    cnn_hparam = read_csv(ROOT / "02_mnist_cnn" / "outputs" / "cnn_sweep" / "cnn_hparam_sweep_summary.csv")
    cnn_arch = read_csv(ROOT / "02_mnist_cnn" / "outputs" / "cnn_arch_sweep" / "cnn_arch_sweep_summary.csv")
    resnet_rows = read_csv(ROOT / "03_resnet_mnist" / "outputs" / "resnet_sweep" / "resnet_sweep_summary.csv")

    mlp_rows = sorted(mlp_rows, key=lambda row: int(row.get("rank", "999999")))
    cnn_hparam = sorted(cnn_hparam, key=lambda row: int(row.get("rank", "999999")))
    cnn_arch = sorted(cnn_arch, key=lambda row: int(row.get("rank", "999999")))
    resnet_rows = sorted(resnet_rows, key=lambda row: int(row.get("rank", "999999")))
    mlp_repeat = sorted(mlp_repeat, key=lambda row: to_float(row.get("mean_test_accuracy", 0)), reverse=True)

    build_tables(mlp_repeat, mlp_rows, cnn_hparam, cnn_arch, resnet_rows)
    plot_mlp_optimizer_lr(mlp_rows)
    plot_mlp_activation_hidden(mlp_rows)
    plot_mlp_decision_boundary(best_or_empty(mlp_repeat or mlp_rows))
    plot_cnn_hparam(cnn_hparam)
    plot_ranked_accuracy_bars(cnn_arch, "final_test_accuracy", "CNN architecture ablation", "cnn_arch_ablation.png", "cnn")
    plot_ranked_accuracy_bars(resnet_rows, "final_test_accuracy", "ResNet ablation", "resnet_ablation.png", "resnet")
    plot_resnet_params(resnet_rows)
    build_main_tex(mlp_repeat, mlp_rows, cnn_hparam, cnn_arch, resnet_rows)
    print(json.dumps({"report_dir": str(REPORT_DIR), "figures": len(list(FIGURE_DIR.glob("*.png")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
