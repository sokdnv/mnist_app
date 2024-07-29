from torch import nn
import torch


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 5, stride=2),
            nn.GELU(),
            nn.Conv2d(8, 4, 5),
            nn.GELU()
        )

        self.clf_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(144, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, pic: torch.Tensor):
        out = self.backbone(pic)
        clf_out = self.clf_head(out)
        box_out = self.box_head(out)
        return clf_out, torch.sigmoid(box_out)