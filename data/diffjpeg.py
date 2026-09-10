
import sys
import torch
import torch.nn as nn
from DiffJPEG.DiffJPEG import DiffJPEG

class RandomJPEG(nn.Module):
    def __init__(self, height=224, width=224, p=0.3, quality_range=(60, 95)):
        super().__init__()
        self.p = p
        self.height = height
        self.width = width
        self.quality_range = quality_range

        self.jpeg = DiffJPEG(
            height=height,
            width=width,
            differentiable=False,  
            quality=80
        )

    def forward(self, x):

        if torch.rand(1).item() > self.p:
            return x

        single = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True

        q = torch.randint(self.quality_range[0], self.quality_range[1], (1,)).item()
        self.jpeg.quality = q


        with torch.no_grad():
            x = self.jpeg(x)

        x = x.detach()   # 

        if single:
            x = x.squeeze(0)

        return x