# CNN 架构消融实验结果

在固定优化器和学习率下，比较卷积核大小、通道宽度、BatchNorm、Dropout 与池化。该部分用于补充优化器/学习率 sweep，回答哪些结构因素对 MNIST CNN 更敏感。

| 排名 | 配置 | 消融因素 | 参数量 | 测试准确率 | 测试损失 |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `kernel5_c32-64_bn_do25_pool` | kernel_size | 455114 | 0.9915 | 0.0253 |
| 2 | `wide_k3_c64-128_bn_do25_pool` | channels | 879114 | 0.9907 | 0.0289 |
| 3 | `baseline_k3_c32-64_bn_do25_pool` | baseline | 421834 | 0.9896 | 0.0298 |
| 4 | `no_batch_norm_k3_c32-64_do25_pool` | batch_norm | 421642 | 0.9895 | 0.0320 |
| 5 | `dropout50_k3_c32-64_bn_pool` | dropout | 421834 | 0.9895 | 0.0323 |
| 6 | `dropout0_k3_c32-64_bn_pool` | dropout | 421834 | 0.9886 | 0.0377 |
| 7 | `narrow_k3_c16-32_bn_do25_pool` | channels | 207018 | 0.9872 | 0.0370 |
| 8 | `compact_k3_c16-32_bn_do0_pool` | compact | 207018 | 0.9857 | 0.0436 |
| 9 | `no_pooling_k3_c32-64_bn_do25` | pooling | 6442954 | 0.9845 | 0.0452 |

最优架构为 `kernel5_c32-64_bn_do25_pool`，测试准确率 `0.9915`。报告中可结合参数量说明，MNIST 任务上过宽模型未必带来同比例收益，BatchNorm 与合适 Dropout 更直接影响收敛稳定性。
