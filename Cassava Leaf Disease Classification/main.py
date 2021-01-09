import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader as DL

import gc
import os

from sklearn.model_selection import train_test_split

from Dataset import DS
from Models import CNN
from fit_predict import fit_1, predict_2

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

seed = 42
root_dir = "G:/ML Projects/CLDC"


def breaker():
    print("\n" + 50*"-" + "\n")


class CFG():
    tr_batch_size = 16
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OL = 11

    def __init__(this, filter_sizes=[64, 128, 256, 512], HL=[2048], AP_size=3, epochs=50, n_folds=5):
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.AP_size = AP_size
        this.epochs = epochs
        this.n_folds = n_folds


if __name__ == "__main__":

    images = np.load(os.path.join(root_dir, "images_1x384x384.npy"))
    labels = np.load(os.path.join(root_dir, "labels.npy"))

    tr_images, va_images, tr_labels, va_labels = train_test_split(images,
                                                                  labels,
                                                                  test_size=0.2,
                                                                  shuffle=True,
                                                                  random_state=seed,
                                                                  stratify=labels)

    del images, labels

    breaker()
    print("Garbage Collected : {}".format(gc.collect()))
    breaker()

    print(tr_images.shape)

    cfg = CFG(filter_sizes=[4, 4, 4, 4], HL=[4], AP_size=3, epochs=2)

    tr_transform = transforms.Compose([transforms.ToTensor(), ])
    va_transform = transforms.Compose([transforms.ToTensor(), ])

    tr_data_setup = DS(X=tr_images, y=tr_labels.reshape(-1, 1), transform=tr_transform, mode="train")
    va_data_setup = DS(X=va_images, y=va_labels.reshape(-1, 1), transform=va_transform, mode="train")

    tr_data = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True, generator=torch.manual_seed(seed))
    va_data = DL(va_data_setup, batch_size=cfg.tr_batch_size, shuffle=False)

    del tr_data_setup, va_data_setup

    torch.manual_seed(seed)
    model = CNN(filter_sizes=cfg.filter_sizes, HL=cfg.HL, OL=cfg.OL, AP_size=cfg.AP_size, DP=0.5).to(cfg.device)
    optimizer = model.getOptimizer(lr=1e-3, wd=1e-5)
    scheduler = model.getPlateauLR(optimizer=optimizer, patience=4, eps=1e-8)

    Losses, Accuracies, bestLossEpoch, bestAccsEpoch = fit_1(model=model, optimizer=optimizer,
                                                             scheduler=scheduler, epochs=cfg.epochs,
                                                             trainloader=tr_data, validloader=va_data,
                                                             criterion=nn.NLLLoss(), device=cfg.device,
                                                             verbose=True, save_to_file=True,
                                                             path="G:/ML Projects/CLDC/States")

    LT = []
    LV = []
    AT = []
    AV = []

    for i in range(len(Losses)):
        LT.append(Losses[i]["train"])
        LV.append(Losses[i]["valid"])
        AT.append(Accuracies[i]["train"])
        AV.append(Accuracies[i]["valid"])

    plt.figure(figsize=(8, 6))
    plt.plot([i + 1 for i in range(len(LT))], LT, "r", label="Training Loss")
    plt.plot([i + 1 for i in range(len(LV))], LV, "b--", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot([i + 1 for i in range(len(LT))], AT, "r", label="Training Accuracy")
    plt.plot([i + 1 for i in range(len(LV))], AV, "b--", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    y_pred = predict_2(model=model, dataloader=va_data, device=cfg.device,
                       path="G:/ML Projects/CLDC/States/Epoch_{}.pt".format(bestLossEpoch))
