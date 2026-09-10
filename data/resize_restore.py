import random
import torchvision.transforms.functional as TF
from PIL import Image


class RandomResizeRestore:
    def __init__(
        self,
        p=0.2,
        scale_range=(0.5, 0.9),
        interpolation=Image.BICUBIC
    ):
        self.p = p
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img):

        if random.random() > self.p:
            return img

        w, h = img.size

        scale = random.uniform(*self.scale_range)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # downsample
        img = TF.resize(
            img,
            (new_h, new_w),
            interpolation=self.interpolation
        )

        # restore
        img = TF.resize(
            img,
            (h, w),
            interpolation=self.interpolation
        )

        return img