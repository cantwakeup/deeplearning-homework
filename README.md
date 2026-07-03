# Deep Learning Homework

本仓库用于完成《深度学习》课程实验，按任务书拆成三个实验目录：

1. `01_numpy_mlp/`：只使用 NumPy 手写 MLP，理解前向传播、损失函数、反向传播和参数更新。
2. `02_mnist_cnn/`：使用 PyTorch 实现 CNN，在 MNIST 上完成手写数字识别，并做优化器、学习率和结构消融。
3. `03_resnet_mnist/`：使用 PyTorch 实现 ResNet-50，同时扩展 ResNet-18/34、Plain-18、stem 和优化器对照。

整体汇报主线是：从手写神经网络底层训练机制，到卷积网络图像分类，再到深层残差网络设计。

## 目录结构

```text
.
├── 01_numpy_mlp/
│   ├── mlp_numpy.py              # NumPy MLP 核心实现：数据、前向、损失、反传、优化器
│   ├── run_experiment.py         # 实验一基础训练入口
│   ├── sweep_mlp.py              # 激活函数、隐藏层、学习率、优化器搜索
│   └── outputs/                  # MLP JSON/CSV/Markdown 结果
├── 02_mnist_cnn/
│   ├── train_cnn.py              # CNN 模型、MNIST 加载、训练评估
│   ├── sweep_cnn.py              # 优化器与学习率对比
│   ├── sweep_arch_cnn.py         # CNN 架构消融
│   └── outputs/                  # CNN 实验结果
├── 03_resnet_mnist/
│   ├── train_resnet.py           # BasicBlock、Bottleneck、ResNet-MNIST 训练
│   ├── sweep_resnet.py           # 深度、残差、stem、优化器、学习率消融
│   └── outputs/                  # ResNet 实验结果
├── latex_report/
│   ├── main.tex                  # 正式 LaTeX 实验报告
│   ├── presentation_script.tex   # 可直接照着念的汇报讲稿
│   ├── build_report_assets.py    # 从 CSV/JSON 重新生成图表和 main.tex
│   ├── figures/                  # 报告图
│   └── tables/                   # LaTeX 表格
├── report/
│   ├── 实验报告提纲.md
│   └── 汇报思路.md
├── environment.yml
└── requirements.txt
```

## 环境

项目使用仓库内隔离 conda 环境，不需要修改系统或 base 环境：

```bash
conda activate /home/huiwei/sy/Deepl/.conda-env
```

也可以不激活，直接使用：

```bash
/home/huiwei/sy/Deepl/.conda-env/bin/python
```

本机实验环境：

- Python 3.10.20
- NumPy
- PyTorch 2.4.0+cu118
- torchvision 0.19.0+cu118
- matplotlib 3.10.9

如需重建环境：

```bash
conda env create -f environment.yml
```

或使用 pip：

```bash
pip install -r requirements.txt
```

## 实验一：NumPy MLP

### 目标

实验一只使用 NumPy 实现一个多层感知机分类器，不能依赖 PyTorch 的 `Linear`、激活函数或自动求导。该实验重点验证对神经网络底层训练流程的理解。

### 数据与模型

- 数据集：三分类螺旋数据，每个样本是二维坐标。
- 选择原因：螺旋数据非线性可分，适合展示隐藏层和激活函数的作用。
- 模型结构：`2 -> hidden_dim -> 3` 的单隐藏层 MLP。
- 手写内容：ReLU/Tanh/Sigmoid、Softmax、交叉熵、L2 正则、反向传播、GD/Momentum/Adam。

代码逻辑在 `01_numpy_mlp/mlp_numpy.py` 中按顺序注释：

1. 构造非线性三分类螺旋数据。
2. 随机打乱并划分训练集、测试集。
3. 只用训练集统计量做标准化。
4. 初始化 `w1/b1`、`w2/b2` 和优化器状态。
5. 前向传播：输入层、隐藏层激活、输出层 Softmax。
6. 交叉熵加 L2 正则。
7. 反向传播计算两层参数梯度。
8. 使用 GD、Momentum 或 Adam 更新参数。
9. 训练循环记录 loss、训练准确率和测试准确率。

### 运行

```bash
.conda-env/bin/python 01_numpy_mlp/run_experiment.py --epochs 1500
.conda-env/bin/python 01_numpy_mlp/sweep_mlp.py --epochs 1000
```

### 结果

- 基础训练测试准确率：`0.9667`
- 最优扩展配置：`ReLU + hidden_dim=16 + Adam + lr=0.01`
- 多随机种子平均测试准确率：`0.9917`
- 标准差：约 `0.0068`

主要图表：

- `latex_report/figures/mlp_optimizer_lr.png`：优化器与学习率对准确率的影响。
- `latex_report/figures/mlp_activation_hidden.png`：激活函数与隐藏层宽度消融。
- `latex_report/figures/mlp_decision_boundary.png`：MLP 学到的螺旋分类边界。

## 实验二：MNIST CNN

### 目标

实验二使用 PyTorch 编写 CNN，对 MNIST 手写数字进行 10 分类。重点是说明卷积、池化、BatchNorm 和 Dropout 如何影响图像分类。

### 模型逻辑

`02_mnist_cnn/train_cnn.py` 中的 `SimpleCNN` 包含：

- 两组卷积块：`Conv2d -> BatchNorm2d(可选) -> ReLU -> MaxPool2d(可选)`
- 全连接分类器：`Flatten -> Dropout -> Linear -> ReLU -> Dropout -> Linear`
- 输入：`1x28x28` 灰度图
- 输出：10 类数字 logits

训练流程：

1. 使用 `torchvision.datasets.MNIST` 加载数据。
2. 使用 MNIST 均值和标准差归一化。
3. 每个 batch 前向计算 logits。
4. 使用交叉熵损失反向传播。
5. 每个 epoch 后在测试集上评估。
6. 保存 JSON 结果和模型权重。

### 运行

基础训练：

```bash
.conda-env/bin/python 02_mnist_cnn/train_cnn.py --epochs 3 --batch-size 256
```

优化器与学习率对比：

```bash
.conda-env/bin/python 02_mnist_cnn/sweep_cnn.py --epochs 3 --batch-size 256 --num-workers 4
```

架构消融：

```bash
.conda-env/bin/python 02_mnist_cnn/sweep_arch_cnn.py --epochs 3 --batch-size 256 --num-workers 4
```

### 结果

- 基础 CNN 测试准确率：`0.9900`
- 优化器/学习率 sweep 最优：`RMSprop + lr=0.0001`，测试准确率 `0.9896`
- 架构消融最优：`5x5 卷积核 + 32/64 通道 + BatchNorm + Dropout=0.25 + Pooling`，测试准确率 `0.9915`

主要图表：

- `latex_report/figures/cnn_optimizer_lr_heatmap.png`：真实运行过的优化器/学习率配置排序图。原热力图会出现空白格，因为 9 组实验不是完整网格，现在改成条形图。
- `latex_report/figures/cnn_arch_ablation.png`：CNN 架构消融图。图左侧直接写出配置变化，包含 baseline、5x5 卷积核、宽/窄通道、去掉 BatchNorm、改变 Dropout、去掉池化等。

消融结论：

- `5x5` 卷积核略优，可能因为 MNIST 笔画较粗，稍大的感受野覆盖更多局部形状。
- 去掉池化后准确率最低，说明池化有助于降低特征维度、稳定短周期训练。
- 加宽通道有提升，但提升不大，说明 MNIST 不需要过宽模型。

## 实验三：ResNet-MNIST

### 目标

实验三实现 ResNet，并在 MNIST 上训练。基础任务保留 ResNet-50；扩展实验加入 ResNet-18、ResNet-34、Plain-18、MNIST stem、ImageNet stem、不同优化器和学习率对比。

### 模型逻辑

`03_resnet_mnist/train_resnet.py` 包含：

- `BasicBlock`：ResNet-18/34 使用，两层 3x3 卷积。
- `Bottleneck`：ResNet-50 使用，`1x1 -> 3x3 -> 1x1`。
- 残差连接：将 identity 加回卷积输出，缓解深层网络训练退化。
- MNIST stem：`3x3, stride=1`，适配 1 通道 28x28 小图。
- ImageNet stem：`7x7, stride=2 + MaxPool`，作为对照。

ResNet 的整体前向过程：

1. stem 提取初始特征。
2. 四个残差阶段逐步增加通道数并下采样。
3. 全局平均池化。
4. 全连接层输出 10 类。

### 运行

基础 ResNet-50：

```bash
.conda-env/bin/python 03_resnet_mnist/train_resnet.py --epochs 3 --batch-size 256
```

ResNet 消融：

```bash
.conda-env/bin/python 03_resnet_mnist/sweep_resnet.py --epochs 3 --batch-size 256 --num-workers 4
```

### 结果

- ResNet-50 基础单跑测试准确率：`0.9872`
- sweep 最优：`ResNet-34 + MNIST stem + Adam + lr=0.001`
- 最优测试准确率：`0.9931`
- Plain-18 无残差测试准确率：`0.9781`
- ImageNet stem 测试准确率：`0.9885`

说明：`03_resnet_mnist/outputs/resnet50_mnist_results.json` 是基础脚本单独训练得到的 ResNet-50 结果，准确率为 `0.9872`；消融图和 `resnet_sweep_summary.csv` 里的 `Baseline: ResNet-50` 是同一轮 sweep 中重跑的 ResNet-50，准确率为 `0.9858`。报告图使用 sweep 内 baseline，是为了和其他消融配置保持同一个对比来源。

主要图表：

- `latex_report/figures/resnet_ablation.png`：ResNet 消融准确率图。图中明确标出 `Baseline: ResNet-50`，并直接写出每个配置变化。
- `latex_report/figures/resnet_params_accuracy.png`：参数量与准确率关系图。点旁边直接标明 `Best ResNet-34`、`Baseline ResNet-50`、`Plain-18 no residual` 等含义。

消融结论：

- ResNet-34 最好，说明 MNIST 上不是越深越好。
- Baseline ResNet-50 参数更多，但准确率低于 ResNet-34，说明小数据/简单任务中过深模型不一定占优。
- Plain-18 明显低于 ResNet-18，说明残差连接确实提升训练稳定性。
- ImageNet stem 不如 MNIST stem，说明 28x28 小图不适合过早强下采样。

## 汇总结果

| 实验 | 数据集 | 最优设置 | 测试准确率 |
| --- | --- | --- | --- |
| NumPy MLP | 三分类螺旋数据 | ReLU, hidden=16, Adam, lr=0.01 | 0.9917 |
| CNN | MNIST | 5x5, 32/64 通道, BN, Dropout=0.25, Pooling | 0.9915 |
| ResNet | MNIST | ResNet-34, MNIST stem, Adam, lr=0.001 | 0.9931 |

基础任务结果：

| 实验 | 基础版本 | 测试准确率 |
| --- | --- | --- |
| NumPy MLP | 单隐藏层 MLP | 0.9667 |
| CNN | 基础 SimpleCNN | 0.9900 |
| ResNet | Baseline ResNet-50 单独训练 | 0.9872 |

## 报告和讲稿

正式报告：

```bash
latex_report/main.tex
```

可直接照着念的讲稿：

```bash
latex_report/presentation_script.tex
```

重新生成图、表和 `main.tex`：

```bash
.conda-env/bin/python latex_report/build_report_assets.py
```

本环境中没有 `xelatex`。如果本机安装了 LaTeX，可在 `latex_report/` 下编译：

```bash
cd latex_report
xelatex main.tex
xelatex presentation_script.tex
```

## 输出文件

结构化结果主要保存在：

- `01_numpy_mlp/outputs/mlp_numpy_results.json`
- `01_numpy_mlp/outputs/mlp_sweep/`
- `02_mnist_cnn/outputs/cnn_mnist_results.json`
- `02_mnist_cnn/outputs/cnn_sweep/`
- `02_mnist_cnn/outputs/cnn_arch_sweep/`
- `03_resnet_mnist/outputs/resnet50_mnist_results.json`
- `03_resnet_mnist/outputs/resnet_sweep/`

模型权重文件通常较大，已通过 `.gitignore` 忽略，不作为主要提交内容。数据集目录也已忽略，运行脚本时会自动下载或复用本地数据。

## 最后检查命令

语法检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  01_numpy_mlp/mlp_numpy.py \
  01_numpy_mlp/run_experiment.py \
  01_numpy_mlp/sweep_mlp.py \
  02_mnist_cnn/train_cnn.py \
  02_mnist_cnn/sweep_cnn.py \
  02_mnist_cnn/sweep_arch_cnn.py \
  03_resnet_mnist/train_resnet.py \
  03_resnet_mnist/sweep_resnet.py \
  latex_report/build_report_assets.py
```

查看当前改动：

```bash
git status --short
```

提交和推送：

```bash
git add .
git commit -m "Finalize deep learning homework report"
git push
```
