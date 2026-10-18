# SPDX-License-Identifier: Apache-2.0
# EEGNeX-style independent PyTorch implementation. See THIRD_PARTY_LICENSES.md.

import torch
import torch.nn as nn


class EEGNeXStyleBackbone(nn.Module):
    """Standalone EEGNeX-style backbone for 22-channel motor-imagery EEG."""
    def __init__(self, n_classes=2, n_channels=22, dropout=0.5):
        super().__init__()
        self.n_channels = int(n_channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 32, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, (self.n_channels, 1), groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, (1, 16), padding=(0, 15), dilation=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 32, (1, 16), padding=(0, 30), dilation=(1, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_channels:
            raise ValueError(f'Expected (B,22,T) or (B,1,22,T), got {tuple(x.shape)}')
        return self.classifier(self.pool(self.features(x)))


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = EEGNeXStyleBackbone()
    for t in (750,):
        print(f'T={t}: output={tuple(m(torch.randn(2,22,t)).shape)}')
    print('trainable_parameters=', count_trainable_parameters(m))
