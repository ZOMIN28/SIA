from torch.utils.data import Dataset, DataLoader
from data.SIAdataset import SIA_Dataset
from CFG import CFG

def build_loader(df, transform, shuffle, batch_size=CFG.batch_size):
    ds = SIA_Dataset(df, transform)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=True
    )