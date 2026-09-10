import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

def fft_transform(x):
    # FFT
    fft = torch.fft.fft2(x, dim=(-2, -1))
    fft = torch.fft.fftshift(fft, dim=(-2, -1))

    real = fft.real
    imag = fft.imag

    mag = torch.sqrt(real**2 + imag**2 + 1e-6)
    phase = torch.atan2(imag, real)

    # log magnitude
    mag = torch.log1p(mag)

    B, C, H, W = mag.shape
    mask = torch.ones((H, W), device=x.device)
    center = H // 2
    radius = H // 8 

    mask[center-radius:center+radius, center-radius:center+radius] = 0
    mask = mask.unsqueeze(0).unsqueeze(0)

    mag = mag * mask
    phase = phase * mask

    return torch.cat([mag, phase], dim=1)


class FFTBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.net(x)
        return self.fc(x.flatten(1))

    

class NextModelFFT(nn.Module):
    def __init__(self, num_classes=10, fft_dim=256):
        super().__init__()

        # ConvNeXt backbone
        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=0  
        )

        rgb_dim = self.backbone.num_features  # =1024

        # FFT
        self.fft_branch = FFTBranch(fft_dim)


        self.classifier = nn.Sequential(
            nn.Linear(rgb_dim + fft_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )


    def forward(self, x):
        rgb_feat = self.backbone(x)          # (B,1024)

        fft = fft_transform(x)
        fft_feat = self.fft_branch(fft)      # (B,256)

        feat = torch.cat([rgb_feat, fft_feat], dim=1)


        return self.classifier(feat)
    
    def forward_features(self, x):
        rgb_feat = self.backbone(x)

        fft = fft_transform(x)
        fft_feat = self.fft_branch(fft)

        feat = torch.cat([rgb_feat, fft_feat], dim=1)
        return feat
    