from torch.utils.data import Dataset, DataLoader
from PIL import Image

class SIA_Dataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_path = "dataset/Data/"
        img_path = base_path + row["path"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, row["ID"]
        else:
            return img, row["y"]
    