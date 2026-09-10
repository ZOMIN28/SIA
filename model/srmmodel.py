import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

# =========================================================
#  SRM Layer
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMConv2d(nn.Module):
    def __init__(self):
        super().__init__()

        # 30 SRM filters (经典论文常用版本)
        self.kernel = nn.Parameter(self._build_srm_kernels(), requires_grad=False)

    def _build_srm_kernels(self):

        filters = []

        # 1. basic high-pass filters
        filters.append([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])

        filters.append([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])

        filters.append([[-1, 2, -1],
                        [2, -4, 2],
                        [-1, 2, -1]])

        filters.append([[1, -2, 1],
                        [-2, 4, -2],
                        [1, -2, 1]])

        # convert to tensor
        kernels = torch.tensor(filters, dtype=torch.float32)

        # shape: (out_channels, 1, 3, 3)
        kernels = kernels.unsqueeze(1)

        return kernels

    def forward(self, x):
        # x: (B, 3, H, W)

        out = []

        for c in range(3):
            out.append(
                F.conv2d(
                    x[:, c:c+1],
                    self.kernel,
                    padding=1
                )
            )

        return torch.cat(out, dim=1)


# =========================================================
# 3. SRM-Xception
# =========================================================

class SRMXception(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # self.srm = SRMConv2d()

        self.backbone = timm.create_model(
            'xception',
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )

        self.fc = nn.Linear(
            self.backbone.num_features,
            num_classes
        )

    def forward(self, x):

        # ---------------------------------------------
        # SRM residual extraction
        # ---------------------------------------------

        # x = self.srm(x)

        # x = torch.clamp(x, -3, 3)

        # ---------------------------------------------
        # backbone classification
        # ---------------------------------------------

        feat = self.backbone(x)
        out = self.fc(feat)

        return out
