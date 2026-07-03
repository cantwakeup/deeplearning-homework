from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. BasicBlock 是 ResNet-18/34 的基本单元：两层 3x3 卷积后再加回输入。
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2. Bottleneck 是 ResNet-50 的基本单元：1x1 降维、3x3 提特征、1x1 升维。
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(x)
            # 残差连接保留原始特征，缓解深层网络训练中的梯度衰减问题。
            out += identity
        return self.relu(out)


class ResNetMNIST(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        layers: list[int],
        num_classes: int = 10,
        residual: bool = True,
        stem_type: str = "mnist",
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.block = block
        self.residual = residual
        self.stem_type = stem_type
        if stem_type == "mnist":
            # 3. MNIST 是 1 通道 28x28 小图，3x3/stride=1 的 stem 能保留更多笔画细节。
            self.stem = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        elif stem_type == "imagenet":
            self.stem = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            raise ValueError(f"Unsupported stem_type: {stem_type}")

        # 4. 四个残差阶段逐步增加通道数，并通过 stride=2 完成空间下采样。
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        # 5. 当尺寸或通道数变化时，用 1x1 卷积对 identity 分支做匹配。
        downsample = None
        expanded_channels = out_channels * self.block.expansion
        if stride != 1 or self.in_channels != expanded_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, expanded_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(expanded_channels),
            )

        layers = [
            self.block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                residual=self.residual,
            )
        ]
        self.in_channels = expanded_channels
        for _ in range(1, blocks):
            layers.append(self.block(self.in_channels, out_channels, residual=self.residual))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 6. 整体前向：stem -> 四个残差阶段 -> 全局平均池化 -> 全连接分类。
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_resnet_mnist(variant: str = "resnet50", residual: bool = True, stem_type: str = "mnist") -> ResNetMNIST:
    # 7. 通过 variant 选择 ResNet-18/34/50，通过 residual 开关构造 Plain 网络对照。
    if variant == "resnet18":
        return ResNetMNIST(BasicBlock, [2, 2, 2, 2], residual=residual, stem_type=stem_type)
    if variant == "resnet34":
        return ResNetMNIST(BasicBlock, [3, 4, 6, 3], residual=residual, stem_type=stem_type)
    if variant == "resnet50":
        return ResNetMNIST(Bottleneck, [3, 4, 6, 3], residual=residual, stem_type=stem_type)
    raise ValueError(f"Unsupported ResNet variant: {variant}")


def resnet50_mnist() -> ResNetMNIST:
    return build_resnet_mnist("resnet50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-50 on MNIST.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--variant", choices=["resnet18", "resnet34", "resnet50"], default="resnet50")
    parser.add_argument("--plain", action="store_true", help="Disable residual additions for a plain-network ablation.")
    parser.add_argument("--stem", choices=["mnist", "imagenet"], default="mnist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "02_mnist_cnn" / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--limit-train", type=int, default=0, help="Use only N train samples for quick tests.")
    parser.add_argument("--limit-test", type=int, default=0, help="Use only N test samples for quick tests.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(name: str, parameters, learning_rate: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def load_mnist(
    data_dir: Path,
    batch_size: int,
    limit_train: int,
    limit_test: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    # 8. ResNet 实验复用 MNIST 数据加载流程，输入保持 1 通道灰度图。
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on local binary packages
        raise RuntimeError(
            "torchvision 导入失败。请安装与当前 torch 版本匹配的 torchvision 后再运行。"
        ) from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    if limit_train > 0:
        train_dataset = Subset(train_dataset, range(min(limit_train, len(train_dataset))))
    if limit_test > 0:
        test_dataset = Subset(test_dataset, range(min(limit_test, len(test_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    # 9. 单轮训练流程与 CNN 一致：前向、交叉熵、反向传播、优化器更新。
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    # 10. 测试时不更新梯度，只比较预测类别和真实标签。
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict_examples(model: nn.Module, loader: DataLoader, device: torch.device, limit: int = 10) -> list[dict[str, int]]:
    model.eval()
    images, labels = next(iter(loader))
    logits = model(images.to(device))
    predictions = logits.argmax(dim=1).cpu()
    return [
        {"true": int(true), "pred": int(pred)}
        for true, pred in zip(labels[:limit], predictions[:limit])
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 11. 按参数选择 ResNet 深度、stem 类型和是否保留残差连接。
    train_loader, test_loader = load_mnist(
        args.data_dir,
        args.batch_size,
        args.limit_train,
        args.limit_test,
        args.num_workers,
    )
    model = build_resnet_mnist(
        variant=args.variant,
        residual=not args.plain,
        stem_type=args.stem,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.learning_rate)

    # 12. 每个 epoch 训练后评估一次，记录 ResNet 的收敛过程。
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }
        history.append(item)
        print(json.dumps(item, ensure_ascii=False))

    # 13. 保存实验配置、最终准确率、样例预测和模型权重。
    result = {
        "model": args.variant,
        "dataset": "MNIST",
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "architecture": {
            "variant": args.variant,
            "residual": not args.plain,
            "stem": args.stem,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "num_workers": args.num_workers,
        "history": history,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "sample_predictions": predict_examples(model, test_loader, device),
    }
    output_stem = args.variant if not args.plain else f"plain_{args.variant}"
    with (args.output_dir / f"{output_stem}_mnist_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), args.output_dir / f"{output_stem}_mnist.pt")


if __name__ == "__main__":
    main()
