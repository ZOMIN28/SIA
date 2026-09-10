import torch
import torch.nn as nn
from transformers import CLIPModel
import torch.nn.functional as F


class ModelCLIP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

        # 只用 vision encoder
        self.vision_model = self.model.vision_model

        for p in self.vision_model.parameters():
            p.requires_grad = False
        
        for p in self.vision_model.encoder.layers[-1].parameters():
            p.requires_grad = True
        
        for p in self.vision_model.encoder.layers[-2].parameters():
            p.requires_grad = True
        
        for p in self.vision_model.encoder.layers[-3].parameters():
            p.requires_grad = True
  

        self.fc = nn.Linear(768, num_classes) 

    def forward(self, x):
        outputs = self.vision_model(pixel_values=x)

        feat = outputs.pooler_output  

        feat = F.normalize(feat, dim=1)

        return self.fc(feat)
    

    def forward_features(self, x):
        outputs = self.vision_model(pixel_values=x)
        feat = outputs.pooler_output
        feat = F.normalize(feat, dim=1)
        return feat