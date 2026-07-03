# ResNet-MNIST 深度与结构消融结果

该实验比较 ResNet-18/34/50、是否保留残差连接、MNIST 友好 stem 与 ImageNet stem，并对 ResNet-18 补充优化器和学习率对比。

| 排名 | 配置 | 消融因素 | 参数量 | 测试准确率 | 测试损失 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `resnet34_mnist_adam_1e-3` | depth | 21280970 | 0.9931 | 0.0229 |
| 2 | `resnet18_mnist_adam_1e-3` | depth | 11172810 | 0.9917 | 0.0262 |
| 3 | `resnet18_mnist_sgd_1e-2` | optimizer | 11172810 | 0.9916 | 0.0277 |
| 4 | `resnet18_mnist_rmsprop_1e-4` | optimizer | 11172810 | 0.9894 | 0.0325 |
| 5 | `resnet18_imagenet_stem_adam_1e-3` | stem | 11175370 | 0.9885 | 0.0346 |
| 6 | `resnet18_mnist_adam_5e-4` | learning_rate | 11172810 | 0.9876 | 0.0356 |
| 7 | `resnet50_mnist_adam_1e-3` | depth | 23519690 | 0.9858 | 0.0511 |
| 8 | `plain18_mnist_adam_1e-3` | residual | 11172810 | 0.9781 | 0.0674 |

最优配置为 `resnet34_mnist_adam_1e-3`，测试准确率 `0.9931`。该结果适合在报告中讨论：MNIST 图像尺寸较小，专门设计的 3x3 stem 通常比 ImageNet 风格 stem 更适配，残差连接则能作为深层模型稳定训练的关键对照。
