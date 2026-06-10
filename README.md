# Deep Learning Homework

本仓库用于完成《深度学习》课程实验。任务书要求提交 3 个题目的程序，分 3 个文件夹组织；每位同学另外提交一份 Word 版实验报告。

## 实验目录

- `01_numpy_mlp/`：只使用 NumPy 实现 MLP，包含输入层、隐藏层、输出层、激活函数、损失函数、训练和预测流程。
- `02_mnist_cnn/`：使用 PyTorch 编写卷积神经网络，对 MNIST 手写数字进行识别。
- `03_resnet_mnist/`：使用 PyTorch 实现 ResNet-50，并在 MNIST 上完成训练和预测。
- `report/`：实验报告提纲，最终需要整理成 Word 版提交。

## 环境

任务书要求 Python 3.7 以上、安装 PyTorch，可使用 CPU 版本。

本项目使用目录内的隔离 conda 环境，不修改共享的 base 环境：

```bash
conda activate /home/huiwei/sy/Deepl/.conda-env
```

也可以不激活，直接使用：

```bash
/home/huiwei/sy/Deepl/.conda-env/bin/python
```

当前隔离环境：

- Python 3.10.20
- torch 2.4.0+cu118
- torchvision 0.19.0+cu118

如果需要重建环境，可参考 `environment.yml` 或 `requirements.txt`。本机实际训练使用的是项目内 `.conda-env`。

## 运行方式

第一题：

```bash
.conda-env/bin/python 01_numpy_mlp/run_experiment.py --epochs 1500
```

第二题：

```bash
.conda-env/bin/python 02_mnist_cnn/train_cnn.py --epochs 3
```

第三题：

```bash
.conda-env/bin/python 03_resnet_mnist/train_resnet.py --epochs 3
```

训练脚本默认 `--num-workers 0`，可以在普通终端环境中自行调大；在受限沙箱中保持 0 更稳定。

每个实验都会把训练结果保存到对应目录的 `outputs/` 中。模型权重文件可能较大，默认不纳入 git。

## 当前结果

| 实验 | 数据集 | 训练配置 | 最终测试准确率 |
| --- | --- | --- | --- |
| NumPy MLP | 三分类螺旋数据 | 1500 epochs, hidden_dim=64 | 0.9667 |
| CNN | MNIST | GPU, 3 epochs, batch_size=256 | 0.9900 |
| ResNet-50 | MNIST | GPU, 3 epochs, batch_size=256 | 0.9872 |

## Git 提交

```bash
git init
git branch -M main
git remote add origin https://github.com/cantwakeup/deeplearning-homework.git
git add .
git commit -m "Add deep learning homework experiments"
git push -u origin main
```
