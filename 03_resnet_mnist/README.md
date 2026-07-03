# 03 ResNet MNIST

本实验使用 PyTorch 实现面向 MNIST 的 ResNet。基础任务训练 ResNet-50，扩展实验比较 ResNet-18、ResNet-34、Plain-18、MNIST stem、ImageNet stem、优化器和学习率。

## 代码逻辑

核心文件是 `train_resnet.py`：

1. `BasicBlock` 用于 ResNet-18/34，由两层 3x3 卷积构成。
2. `Bottleneck` 用于 ResNet-50，由 `1x1 -> 3x3 -> 1x1` 卷积构成。
3. 残差连接把 identity 加回卷积输出，用于缓解深层网络训练退化。
4. MNIST stem 使用 `3x3, stride=1`，适配 1 通道 28x28 小图。
5. ImageNet stem 使用 `7x7, stride=2 + MaxPool`，作为过早下采样的对照。
6. `build_resnet_mnist` 统一构造 ResNet-18/34/50 或关闭残差的 Plain 网络。
7. `sweep_resnet.py` 将 ResNet-50 baseline 放入同一轮 sweep，再比较深度、残差、stem、优化器和学习率。

## 运行

基础 ResNet-50：

```bash
.conda-env/bin/python 03_resnet_mnist/train_resnet.py --epochs 3 --batch-size 256 --num-workers 4
```

快速检查：

```bash
.conda-env/bin/python 03_resnet_mnist/train_resnet.py --epochs 1 --limit-train 1024 --limit-test 256
```

消融实验：

```bash
.conda-env/bin/python 03_resnet_mnist/sweep_resnet.py --epochs 3 --batch-size 256 --num-workers 4
```

## 结果

- ResNet-50 基础单跑测试准确率：`0.9872`
- sweep 最优：`ResNet-34 + MNIST stem + Adam + lr=0.001`
- sweep 最优测试准确率：`0.9931`
- sweep 内 `Baseline: ResNet-50` 测试准确率：`0.9858`
- Plain-18 无残差测试准确率：`0.9781`
- ImageNet stem 测试准确率：`0.9885`

说明：`resnet50_mnist_results.json` 是基础脚本单独训练结果；`resnet_sweep_summary.csv` 里的 `Baseline: ResNet-50` 是和其他消融配置一起重跑的结果。报告图使用 sweep 内 baseline，便于和 ResNet-18/34、Plain-18 等配置直接对比。

## 输出

- `03_resnet_mnist/outputs/resnet50_mnist_results.json`
- `03_resnet_mnist/outputs/resnet_sweep/resnet_sweep_summary.csv`
- `03_resnet_mnist/outputs/resnet_sweep/resnet_sweep_report.md`

模型权重 `*.pt` 已在 `.gitignore` 中忽略，提交时主要保留结构化结果。
