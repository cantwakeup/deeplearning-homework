from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def make_spiral_data(
    samples_per_class: int = 160,
    num_classes: int = 3,
    noise: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = samples_per_class * num_classes
    x = np.zeros((total, 2), dtype=np.float64)
    y = np.zeros(total, dtype=np.int64)

    for class_id in range(num_classes):
        ix = range(class_id * samples_per_class, (class_id + 1) * samples_per_class)
        radius = np.linspace(0.0, 1.0, samples_per_class)
        theta = np.linspace(class_id * 4.0, (class_id + 1) * 4.0, samples_per_class)
        theta += rng.normal(0.0, noise, samples_per_class)
        x[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[ix] = class_id

    indices = rng.permutation(total)
    return x[indices], y[indices]


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    test_size = int(x.shape[0] * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


@dataclass
class TrainingMetrics:
    epoch: int
    loss: float
    train_accuracy: float
    test_accuracy: float


class NumpyMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate
        self.l2 = l2
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float64)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim), dtype=np.float64)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        a1 = self.relu(z1)
        logits = a1 @ self.w2 + self.b2
        probabilities = self.softmax(logits)
        cache = {"x": x, "z1": z1, "a1": a1, "probabilities": probabilities}
        return probabilities, cache

    def cross_entropy_loss(self, probabilities: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-12
        clipped = np.clip(probabilities, eps, 1.0 - eps)
        data_loss = -np.sum(y_true * np.log(clipped)) / y_true.shape[0]
        reg_loss = 0.5 * self.l2 * (np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2))
        return float(data_loss + reg_loss)

    def backward(self, cache: dict[str, np.ndarray], y_true: np.ndarray) -> None:
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]
        probabilities = cache["probabilities"]
        batch_size = x.shape[0]

        d_logits = (probabilities - y_true) / batch_size
        d_w2 = a1.T @ d_logits + self.l2 * self.w2
        d_b2 = d_logits.sum(axis=0, keepdims=True)
        d_a1 = d_logits @ self.w2.T
        d_z1 = d_a1 * self.relu_grad(z1)
        d_w1 = x.T @ d_z1 + self.l2 * self.w1
        d_b1 = d_z1.sum(axis=0, keepdims=True)

        self.w1 -= self.learning_rate * d_w1
        self.b1 -= self.learning_rate * d_b1
        self.w2 -= self.learning_rate * d_w2
        self.b2 -= self.learning_rate * d_b2

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 1500,
        log_every: int = 100,
    ) -> list[TrainingMetrics]:
        num_classes = int(max(y_train.max(), y_test.max()) + 1)
        y_train_one_hot = one_hot(y_train, num_classes)
        history: list[TrainingMetrics] = []

        for epoch in range(1, epochs + 1):
            probabilities, cache = self.forward(x_train)
            loss = self.cross_entropy_loss(probabilities, y_train_one_hot)
            self.backward(cache, y_train_one_hot)

            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                history.append(
                    TrainingMetrics(
                        epoch=epoch,
                        loss=loss,
                        train_accuracy=self.accuracy(x_train, y_train),
                        test_accuracy=self.accuracy(x_test, y_test),
                    )
                )

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        probabilities, _ = self.forward(x)
        return probabilities.argmax(axis=1)

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(x) == y))

