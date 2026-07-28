"""CNN for wafer map classification.

Input is a 64x64 map one-hot encoded over {outside, pass, fail} so the network
sees wafer geometry and failing die as separate channels rather than ordinal
values.
"""

import numpy as np
import torch
from torch import nn


def maps_to_tensor(X: np.ndarray) -> torch.Tensor:
    """(N, H, W) uint8 in {0,1,2} -> (N, 3, H, W) float one-hot."""
    t = torch.from_numpy(X).long()
    return torch.nn.functional.one_hot(t, num_classes=3).permute(0, 3, 1, 2).float()


class WaferCNN(nn.Module):
    def __init__(self, n_classes: int = 9):
        super().__init__()
        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 32),    # 64 -> 32
            block(32, 64),   # 32 -> 16
            block(64, 128),  # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
