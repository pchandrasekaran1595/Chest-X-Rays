import torch
from torch.utils.data import Dataset

class DS(Dataset):
    def __init__(this, X=None, y=None, transform=None, mode="train"):
        this.transform = transform
        this.mode = mode
        this.X = X
        if mode == "train" or mode == "valid":
            this.y = y

    def __len__(this):
        return this.X.shape[0]

    def __getitem__(this, idx):
        img = this.transform(this.X[idx])
        if this.mode == "train" or this.mode == "valid":
            return img, torch.LongTensor(this.y[idx])
        else:
            return img