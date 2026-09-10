import torchvision.transforms as T
from torchvision.transforms import v2
from data.diffjpeg import RandomJPEG
from data.resize_restore import RandomResizeRestore
import torch

add = 256

train_transform_fft_bin = T.Compose([
    RandomResizeRestore(
        p=0.2,
        scale_range=(0.6, 0.9)
    ),

    T.Resize((288+add, 288+add)),

    T.RandomApply([
        T.CenterCrop(256+add)
    ], p=0.2),
    T.RandomResizedCrop(256+add, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    
    T.RandomHorizontalFlip(p=0.2),

    T.RandomApply([T.GaussianBlur(3)], p=0.2),
    T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.1)], p=0.4),
    T.RandomGrayscale(p=0.15),
    T.RandomRotation(3),
    

    T.ToTensor(),

    RandomJPEG(
        height=256+add,
        width=256+add,
        p=0.4,
        quality_range=(50, 95)
    ),  # JPEG

    # ImageNet normalization
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])


train_transform_srm_bin = T.Compose([
    T.Resize((344, 344)),

    T.RandomApply([
        T.CenterCrop(304)
    ], p=0.2),

    T.RandomResizedCrop(
        304,
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
        height=304,
        width=304,
        p=0.3,
        quality_range=(60, 95)
    ),

    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])




test_transform_fft_bin = T.Compose([
    T.Resize((256+add, 256+add)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],
                [0.229,0.224,0.225])
])


test_transform_srm_bin = T.Compose([
    T.Resize((304, 304)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])