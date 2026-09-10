import torch
import torch.nn as nn

class ModelDINO(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14'
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.backbone.blocks[-1].parameters():
            p.requires_grad = True
        
        for p in self.backbone.blocks[-2].parameters():
            p.requires_grad = True
        
        for p in self.backbone.blocks[-3].parameters():
            p.requires_grad = True

        self.fc = nn.Linear(768, num_classes)

    def forward(self,x):
        feat = self.backbone(x)
        return self.fc(feat)
    
    def forward_features(self, x):
        feat = self.backbone(x)
        return feat