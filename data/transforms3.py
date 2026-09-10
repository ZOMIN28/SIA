import torchvision.transforms as T
from torchvision.transforms import v2
from data.diffjpeg import RandomJPEG
from data.resize_restore import RandomResizeRestore
import torch


# =========================================================
#  train_transform
# =========================================================

train_transform_fft = T.Compose([
    RandomResizeRestore(
        p=0.2,
        scale_range=(0.6, 0.9)
    ),

    T.Resize((288, 288)),

    T.RandomApply([
        T.CenterCrop(256)
    ], p=0.2),
    T.RandomResizedCrop(256, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    
    T.RandomHorizontalFlip(p=0.2),

    T.RandomApply([T.GaussianBlur(3)], p=0.2),
    T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.1)], p=0.4),
    T.RandomGrayscale(p=0.15),
    T.RandomRotation(3),
    

    T.ToTensor(),

    RandomJPEG(
        height=256,
        width=256,
        p=0.4,
        quality_range=(50, 95)
    ),  # JPEG

    # ImageNet normalization
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])




train_transform_clip = T.Compose([

    T.Resize((256,256)),

    T.RandomApply([
        T.CenterCrop(224)
    ], p=0.2),
    T.RandomResizedCrop(
        224,
        scale=(0.85, 1.0),
        ratio=(0.95, 1.05)
    ),

    T.RandomHorizontalFlip(), 

    T.RandomApply([T.GaussianBlur(3)], p=0.1),
    T.RandomApply([T.ColorJitter(0.1,0.1,0.1,0.05)], p=0.2),
    T.RandomGrayscale(p=0.15),
    T.RandomRotation(2),

    T.ToTensor(),

    RandomJPEG(
        height=224,
        width=224,
        p=0.2,
        quality_range=(65, 95)
    ), 


    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711])
])


train_transform_dino = T.Compose([
    T.Resize((256, 256)),
    
    T.RandomApply([
        T.CenterCrop(224)
    ], p=0.2),
    T.RandomResizedCrop(
        224,
        scale=(0.85, 1.0),
        ratio=(0.95, 1.05)
    ),

    T.RandomHorizontalFlip(),

    # 中等增强
    T.RandomApply([T.GaussianBlur(3)], p=0.2),
    T.RandomApply([T.ColorJitter(0.1,0.1,0.1,0.05)], p=0.25),
    T.RandomRotation(3),
    T.RandomGrayscale(p=0.2),

    T.ToTensor(),

    RandomJPEG(
        height=224,
        width=224,
        p=0.25,
        quality_range=(65, 95)
    ), 

    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])



train_transform_srm = T.Compose([
    T.Resize((320, 320)),

    T.RandomApply([
        T.CenterCrop(288)
    ], p=0.2),

    T.RandomResizedCrop(
        288,
        scale=(0.85, 1.0),
        ratio=(0.95, 1.05)
    ),

    T.RandomHorizontalFlip(p=0.2),

    T.RandomApply([T.GaussianBlur(3)], p=0.2),
    T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.1)], p=0.2),
    T.RandomGrayscale(p=0.1),
    T.RandomRotation(2),

    T.ToTensor(),

    RandomJPEG(
        height=288,
        width=288,
        p=0.3,
        quality_range=(60, 95)
    ),

    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# =========================================================
#  test_transform
# =========================================================


test_transform_fft = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])


test_transform_clip = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711])
])

test_transform_dino = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])

test_transform_srm = T.Compose([
    T.Resize((288, 288)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

