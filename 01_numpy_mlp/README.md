# 01 NumPy MLP

本实验只使用 NumPy 实现一个多层感知机分类器，用三分类螺旋数据验证前向传播、交叉熵损失、反向传播和参数更新流程。

## 代码逻辑

核心文件是 `mlp_numpy.py`：

1. `make_spiral_data` 生成非线性三分类螺旋数据。
2. `train_test_split` 划分训练集和测试集。
3. `standardize_train_test` 只使用训练集统计量做标准化。
4. `NumpyMLP.forward` 完成 `输入 -> 隐藏层 -> 激活函数 -> 输出层 -> Softmax`。
5. `cross_entropy_loss` 计算交叉熵并加入 L2 正则。
6. `backward` 手写两层网络的梯度。
7. `apply_gradients` 支持 GD、Momentum、Adam 三种优化器。
8. `fit` 负责训练循环并记录 loss、训练准确率和测试准确率。

## 运行

基础实验：

```bash
.conda-env/bin/python 01_numpy_mlp/run_experiment.py --epochs 1500
```

多因素搜索：

```bash
.conda-env/bin/python 01_numpy_mlp/sweep_mlp.py --epochs 1000
```

## 结果

- 基础训练测试准确率：`0.9667`
- 最优复验配置：`ReLU + hidden_dim=16 + Adam + lr=0.01`
- 多随机种子平均测试准确率：`0.9917`

## 输出

- `01_numpy_mlp/outputs/mlp_numpy_results.json`
- `01_numpy_mlp/outputs/loss_history.csv`
- `01_numpy_mlp/outputs/mlp_sweep/mlp_sweep_summary.csv`
- `01_numpy_mlp/outputs/mlp_sweep/mlp_top_seed_summary.csv`
- `01_numpy_mlp/outputs/mlp_sweep/mlp_sweep_report.md`
