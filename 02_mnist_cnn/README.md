# 02 MNIST CNN

本实验使用 PyTorch 实现卷积神经网络，对 MNIST 手写数字做 10 分类，并补充优化器、学习率和网络结构消融。

## 代码逻辑

核心文件是 `train_cnn.py`：

1. `SimpleCNN` 搭建两组卷积块和一个全连接分类器。
2. 卷积块结构为 `Conv2d -> BatchNorm(可选) -> ReLU -> MaxPool(可选)`。
3. `load_mnist` 使用 torchvision 加载 MNIST，并按标准均值方差归一化。
4. `train_one_epoch` 完成前向、交叉熵、反向传播和优化器更新。
5. `evaluate` 在测试集关闭梯度，统计测试损失和准确率。
6. `sweep_cnn.py` 固定网络结构，比较 Adam、SGD、RMSprop 和学习率。
7. `sweep_arch_cnn.py` 固定训练流程，比较卷积核、通道数、BatchNorm、Dropout 和池化。

## 运行

基础训练：

```bash
.conda-env/bin/python 02_mnist_cnn/train_cnn.py --epochs 3 --batch-size 256
```

优化器和学习率对比：

```bash
.conda-env/bin/python 02_mnist_cnn/sweep_cnn.py --epochs 3 --batch-size 256 --num-workers 4
```

结构消融：

```bash
.conda-env/bin/python 02_mnist_cnn/sweep_arch_cnn.py --epochs 3 --batch-size 256 --num-workers 4
```

## 结果

- 基础 CNN 测试准确率：`0.9900`
- 优化器/学习率 sweep 最优：`RMSprop + lr=0.0001`，测试准确率 `0.9896`
- 架构消融最优：`5x5 卷积核 + 32/64 通道 + BatchNorm + Dropout=0.25 + Pooling`，测试准确率 `0.9915`

## 输出

- `02_mnist_cnn/outputs/cnn_mnist_results.json`
- `02_mnist_cnn/outputs/cnn_sweep/cnn_hparam_sweep_summary.csv`
- `02_mnist_cnn/outputs/cnn_sweep/cnn_hparam_sweep_report.md`
- `02_mnist_cnn/outputs/cnn_arch_sweep/cnn_arch_sweep_summary.csv`
- `02_mnist_cnn/outputs/cnn_arch_sweep/cnn_arch_sweep_report.md`

模型权重 `*.pt` 已在 `.gitignore` 中忽略，提交时主要保留结构化结果。
