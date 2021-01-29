import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader as DL
from torchvision import transforms
import os
import cv2
import gc

from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Models import CNN_8BN, CNN_11BN, CNN_13BN
import fit_predict as fp
from Dataset import DS

seed = 42


######################################################################
def breaker():
    print("\n" + 50 * "-" + "\n")


######################################################################
def head(x=None, no_of_ele=5):
    print(x[:no_of_ele])


######################################################################
def getFileNames(path):
    f_names = []
    for _, _, filenames in os.walk(path):
        for filename in filenames:
            f_names.append(filename)
    return f_names


######################################################################
def getImages(path=None, filenames=None, size=None, color=False):
    images = []
    for name in filenames:
        try:
            if not color:
                image = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR)
        except Exception as e:
            print(e)

        if size:
            image = cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_LANCZOS4)

        if not color:
            images.append(image.reshape(size, size, 1))
        else:
            images.append(image)

    return np.array(images)


class CFG():
    tr_batch_size = 64
    ts_batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OL = 3

    def __init__(this, in_channels=None, filter_sizes=None, HL=None, AP_size=None, epochs=None, n_folds=None):
        this.in_channels = in_channels
        this.filter_sizes = filter_sizes
        this.HL = HL
        this.AP_size = AP_size
        this.epochs = epochs
        this.n_folds = n_folds


root_dir = "./COVID-19 Radiography Database"
normal_dir = os.path.join(root_dir, "NORMAL")
covid_dir  = os.path.join(root_dir, "COVID")
pnemon_dir = os.path.join(root_dir, "Viral Pneumonia")


if __name__ == "__main__":
    start_time = time()

    cov_filenames = getFileNames(covid_dir)
    nor_filenames = getFileNames(normal_dir)
    pne_filenames = getFileNames(pnemon_dir)

    size = 256

    cov_images = getImages(path=covid_dir, filenames=cov_filenames, size=size, color=False)
    nor_images = getImages(path=normal_dir, filenames=nor_filenames, size=size, color=False)
    pne_images = getImages(path=pnemon_dir, filenames=pne_filenames, size=size, color=False)

    images = np.concatenate((cov_images, nor_images, pne_images), axis=0)
    labels = np.concatenate((np.ones((cov_images.shape[0], 1)) * 0,
                             np.ones((nor_images.shape[0], 1)) * 1,
                             np.ones((pne_images.shape[0], 1)) * 2), axis=0)
    labels = labels.astype("int")

    breaker()
    print("Time Taken to process Data : {:.2f} seconds".format(time() - start_time))

    del cov_filenames, nor_filenames, pne_filenames, cov_images, nor_images, pne_images, start_time

    breaker()
    print("Garbage Collected : {}".format(gc.collect()))
    breaker()

    tr_images, va_images, tr_labels, va_labels = train_test_split(images, labels,
                                                                  test_size=0.2,
                                                                  shuffle=True,
                                                                  random_state=seed,
                                                                  stratify=labels)

    cfg = CFG(in_channels=1,
              filter_sizes=[64, 128, 256, 512],
              HL=[4096, 4096],
              AP_size=3,
              epochs=25,
              n_folds=None)

    tr_transform = transforms.Compose([transforms.ToTensor(), ])
    va_transform = transforms.Compose([transforms.ToTensor(), ])

    tr_data_setup = DS(X=tr_images, y=tr_labels, transform=tr_transform, mode="train")
    va_data_setup = DS(X=va_images, y=va_labels, transform=va_transform, mode="valid")

    tr_data = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True, generator=torch.manual_seed(seed))
    va_data = DL(va_data_setup, batch_size=cfg.tr_batch_size, shuffle=False)

    torch.manual_seed(seed)
    model = CNN_8BN(in_channels=cfg.in_channels,
                    filter_sizes=cfg.filter_sizes,
                    HL=cfg.HL, OL=cfg.OL,
                    AP_size=cfg.AP_size, DP=0.5)

    """model = CNN_11BN(in_channels=cfg.in_channels,
                    filter_sizes=cfg.filter_sizes,
                    HL=cfg.HL, OL=cfg.OL,
                    AP_size=cfg.AP_size, DP=0.5)

    model = CNN_13BN(in_channels=cfg.in_channels,
                    filter_sizes=cfg.filter_sizes,
                    HL=cfg.HL, OL=cfg.OL,
                    AP_size=cfg.AP_size, DP=0.5)"""

    optimizer = model.getOptimizer(A_S=True, lr=1e-3, wd=0)
    scheduler = model.getPlateauLR(optimizer=optimizer, patience=5, eps=1e-8)

    Losses, Accuracies, bestLossEpoch, bestAccsEpoch = fp.fit_(model=model, optimizer=optimizer, scheduler=None,
                                                            epochs=cfg.epochs,
                                                            trainloader=tr_data, validloader=va_data,
                                                            criterion=nn.NLLLoss(), device=cfg.device,
                                                            verbose=True)

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
    plt.plot([i + 1 for i in range(len(AT))], AT, "r", label="Training Accuracy")
    plt.plot([i + 1 for i in range(len(AV))], AV, "b--", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()

    """y_pred = fp.predict_(model=model, dataloader=ts_data, 
                         device=cfg.device, path="./Epoch_{}.pt".format(bestLossEpoch))
    y_pred = fp.predict_(model=model, dataloader=ts_data,
                         device=cfg.device, path="./Epoch_{}.pt".format(bestAccsEpoch))"""

    print("Garbage Collected : {}".format(gc.collect()))
    breaker()
    print("EXECUTION COMPLETE")
    breaker()
