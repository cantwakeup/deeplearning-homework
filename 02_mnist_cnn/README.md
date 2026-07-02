# 02 MNIST CNN

本实验使用 PyTorch 编写卷积神经网络，并通过 `torchvision.datasets.MNIST` 加载 MNIST 手写数字数据集。

运行：

```bash
python3 02_mnist_cnn/train_cnn.py --epochs 3
```

常用调参示例：

```bash
python3 02_mnist_cnn/train_cnn.py --epochs 5 --optimizer adam --learning-rate 0.0005 --batch-size 64
```

多种情况分析：

```bash
python3 02_mnist_cnn/sweep_cnn.py --epochs 3 --batch-size 256 --num-workers 4
```

如果已经激活项目环境，也可以使用 `python`；否则使用仓库根目录下的 `.conda-env/bin/python`。脚本默认 `--num-workers 0`，适合受限环境。

输出：

- `02_mnist_cnn/outputs/cnn_mnist_results.json`
- `02_mnist_cnn/outputs/cnn_sweep/cnn_hparam_sweep_summary.csv`
- `02_mnist_cnn/outputs/cnn_sweep/cnn_hparam_sweep_report.md`
- `02_mnist_cnn/outputs/cnn_mnist.pt`
