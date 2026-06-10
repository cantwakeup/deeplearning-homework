# 03 ResNet MNIST

本实验使用 PyTorch 实现 ResNet-50，并将输入层适配到 MNIST 的 1 通道 28x28 图像。

运行：

```bash
python3 03_resnet_mnist/train_resnet.py --epochs 3
```

为了快速检查代码，也可以限制样本量：

```bash
python3 03_resnet_mnist/train_resnet.py --epochs 1 --limit-train 1024 --limit-test 256
```

如果已经下载过 CNN 实验的 MNIST 数据，可以复用同一份数据：

```bash
python3 03_resnet_mnist/train_resnet.py --epochs 1 --data-dir 02_mnist_cnn/data
```

脚本默认 `--num-workers 0`，适合受限环境。

输出：

- `03_resnet_mnist/outputs/resnet50_mnist_results.json`
- `03_resnet_mnist/outputs/resnet50_mnist.pt`
