"""
EEGNet 分类器（PyTorch，sklearn 兼容接口）
参考 Lawhern et al., 2018
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class _EEGNetModule(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int = 4):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)
        self.conv4 = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self._forward_features(dummy)
            self.n_features = out.shape[1]

        self.fc = nn.Linear(self.n_features, n_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        return self.fc(self._forward_features(x))


class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """EEGNet 包装器，输入 (n_trials, n_channels, n_times)。"""

    def __init__(
        self,
        n_epochs_train: int = 80,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.n_epochs_train = n_epochs_train
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = self.scaler_.transform(X.reshape(len(X), -1)).reshape(X.shape)
        X = (X - X.mean(axis=-1, keepdims=True)) / (X.std(axis=-1, keepdims=True) + 1e-8)
        t = torch.tensor(X[:, None, :, :], dtype=torch.float32)
        return t

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n_channels, n_times = X.shape[1], X.shape[2]
        n_classes = len(np.unique(y))

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X.reshape(len(X), -1))

        self.model_ = _EEGNetModule(n_channels, n_times, n_classes)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(self._to_tensor(X), torch.tensor(y))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model_.train()
        for _ in range(self.n_epochs_train):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(self._to_tensor(np.asarray(X, dtype=np.float32)))
            return logits.argmax(dim=1).cpu().numpy()

    def score(self, X, y):
        return float((self.predict(X) == y).mean())

    def finetune(self, X, y, n_epochs_finetune: int = 30, learning_rate_finetune: float = 1e-4):
        """在已训练模型上继续微调（用于迁移学习）。"""
        if not hasattr(self, "model_"):
            raise RuntimeError("请先调用 fit() 预训练")
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=learning_rate_finetune)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(self._to_tensor(X), torch.tensor(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for _ in range(n_epochs_finetune):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
        return self
