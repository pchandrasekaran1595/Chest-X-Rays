import torch
from torch.utils.data import DataLoader as DL
import numpy as np
from time import time
from sklearn.model_selection import KFold

from Dataset import Dataset
from MLP import MLP

def breaker():
    print("\n" + 50*"-" + "\n")

def singleModelTrain(model=None, optimizer=None, dataloader=None, epochs=None, criterion=None, device=None):
    breaker()
    print("Training ...")

    LP = []

    model.train()
    model.to(device)

    start_time = time()
    for e in range(epochs):
        lossPerPass = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            lossPerPass += loss.item()
            optimizer.step()
        LP.append(lossPerPass)
        print("Epoch {} | Loss : {}".format(e+1, lossPerPass))

    breaker()
    print("Time Taken [{} Epochs] : {:.2f} minutes".format(epochs, (time()-start_time)/60))
    breaker()
    print("Training Complete")
    breaker()

    return LP

def kFoldModelTrain(X=None, y=None, epochs=None, n_folds=None,
                    IL=None, HL=None, OL=None, use_DP=None, DP1=0.2, DP2=0.5,
                    lr=1e-3, wd=0, patience=5, lr_eps=1e-6, use_all=None,
                    tr_batch_size=None, va_batch_size=None,
                    criterion=None, device=None, path=None):

    breaker()
    print("Training ...")
    breaker()

    LP = []
    names = []

    fold = 1
    bestLoss = {"train" : np.inf, "valid" : np.inf}

    start_time = time()
    for tr_idx, va_idx in KFold(n_splits=n_folds, shuffle=True, random_state=0).split(X, y):
        print("Processing Fold {} ...".format(fold))

        X_train, X_valid, y_train, y_valid = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

        tr_data_setup = Dataset(X_train, y_train.reshape(-1,1))
        va_data_setup = Dataset(X_valid, y_valid.reshape(-1,1))

        DLS = {"train" : DL(tr_data_setup, batch_size=tr_batch_size, shuffle=True, generator=torch.manual_seed(0)),
               "valid" : DL(va_data_setup, batch_size=va_batch_size, shuffle=False)
              }

        torch.manual_seed(0)
        model = MLP(IL, HL, OL, use_DP, DP1, DP2)
        model.to(device)

        optimizer = model.getOptimizer(lr=lr, wd=wd)
        scheduler = model.getPlateauScheduler(optimizer, patience, lr_eps)

        for e in range(epochs):
            epochLoss = {"train" : 0.0, "valid" : 0.0}
            for phase in ["train", "valid"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()
                lossPerPass = 0

                for features, labels in DLS[phase]:
                    features, labels = features.to(device), labels.to(device)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        output = model(features)
                        loss = criterion(output, labels)
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    lossPerPass += (loss.item()/labels.shape[0])
                epochLoss[phase] = lossPerPass
            LP.append(epochLoss)
            scheduler.step(epochLoss["valid"])
            if not use_all:
                if epochLoss["valid"] < bestLoss["valid"]:
                    bestLoss = epochLoss
                    name = "Model_Fold_{fold}.pt".format(fold=fold)
                    names.append(name)
                    torch.save(model.state_dict(), path + name)
            else:
                name = "Model_Fold_{fold}.pt".format(fold=fold)
                names.append(name)
                torch.save(model.state_dict(), path + name)
                if epochLoss["valid"] < bestLoss["valid"]:
                    bestLoss = epochLoss
            print("Epoch {}, Fold {} | Train Loss : {}, Val Loss : {}".format(e+1, fold, epochLoss["train"], epochLoss["valid"]))
        fold += 1

    breaker()
    print("Time Taken [{} Epochs, {} Folds] : {:.2f} minutes".format(epochs, n_folds, (time()-start_time)/60))

    breaker()
    sum_t = 0
    sum_v = 0

    for i in range(len(LP)):
        sum_t = sum_t + LP[i]["train"]
        sum_v = sum_v + LP[i]["valid"]

    print("Average Loss [Train] : {:.5f}".format(sum_t / len(LP)))
    print("Average Loss [Valid] : {:.5f}".format(sum_v / len(LP)))
    breaker()

    print("Training Complete")
    breaker()

    return names, LP


def seedAvgKFoldModelTrain(X=None, y=None, epochs=None, n_folds=None, n_seeds=None,
                           IL=None, HL=None, OL=None, use_DP=None, DP1=0.2, DP2=0.5,
                           lr=1e-3, wd=0, patience=5, lr_eps=1e-6, use_all=None,
                           tr_batch_size=None, va_batch_size=None,
                           criterion=None, device=None, path=None):

    breaker()
    print("Training ...")
    breaker()

    LP = []
    names = []

    bestLoss = {"train" : np.inf, "valid" : np.inf}
    seeders = [i for i in range(n_seeds)]

    start_time = time()
    for seed in seeders:
        fold = 1
        for tr_idx, va_idx in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y):
            print("Processing Seed {}, Fold {} ...".format(seed, fold))

            X_train, X_valid, y_train, y_valid = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]

            tr_data_setup = Dataset(X_train, y_train.reshape(-1,1))
            va_data_setup = Dataset(X_valid, y_valid.reshape(-1,1))

            DLS = {"train" : DL(tr_data_setup, batch_size=tr_batch_size, shuffle=True, generator=torch.manual_seed(0)),
                   "valid" : DL(va_data_setup, batch_size=va_batch_size, shuffle=False)
                  }

            torch.manual_seed(0)
            model = MLP(IL, HL, OL, use_DP, DP1, DP2)
            model.to(device)

            optimizer = model.getOptimizer(lr=lr, wd=wd)
            scheduler = model.getPlateauScheduler(optimizer, patience, lr_eps)

            for e in range(epochs):
                epochLoss = {"train" : 0.0, "valid" : 0.0}
                for phase in ["train", "valid"]:
                    if phase == "train":
                        model.train()
                    else:
                        model.eval()
                    lossPerPass = 0

                    for features, labels in DLS[phase]:
                        features, labels = features.to(device), labels.to(device)

                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == "train"):
                            output = model(features)
                            loss = criterion(output, labels)
                            if phase == "train":
                                loss.backward()
                                optimizer.step()
                        lossPerPass += (loss.item()/labels.shape[0])
                    epochLoss[phase] = lossPerPass
                LP.append(epochLoss)
                scheduler.step(epochLoss["valid"])
                if not use_all:
                    if epochLoss["valid"] < bestLoss["valid"]:
                        bestLoss = epochLoss
                        name = "Model_Fold_{fold}.pt".format(fold=fold)
                        names.append(name)
                        torch.save(model.state_dict(), path + name)
                else:
                    name = "Model_Seed_{seed}_Fold_{fold}.pt".format(seed=seed, fold=fold)
                    names.append(name)
                    torch.save(model.state_dict(), path + name)
                    if epochLoss["valid"] < bestLoss["valid"]:
                        bestLoss = epochLoss
                print("Epoch {}, Fold {}, Seed {} | Train Loss : {}, Val Loss : {}".format(e+1, fold, seed, epochLoss["train"], epochLoss["valid"]))
            fold += 1

    breaker()
    print("Time Taken [{} Epochs, {} Folds, {} Seeds] : {:.2f} minutes".format(epochs, n_folds, n_seeds, (time()-start_time)/60))

    breaker()
    sum_t = 0
    sum_v = 0

    for i in range(len(LP)):
        sum_t = sum_t + LP[i]["train"]
        sum_v = sum_v + LP[i]["valid"]

    print("Average Loss [Train] : {:.5f}".format(sum_t / len(LP)))
    print("Average Loss [Valid] : {:.5f}".format(sum_v / len(LP)))
    breaker()

    print("Training Complete")
    breaker()

    return names, LP

#######################################################################################################

def singleModelEval(model=None, dataloader=None, batch_size=None, device=None):
    y_pred = torch.zeros(batch_size, 1).to(device)

    model.eval()
    model.to(device)

    for X in dataloader:
        X = X.to(device)
        with torch.no_grad():
            output = torch.sigmoid(model(X))
        y_pred = torch.cat((y_pred, output), dim=0)
    y_pred = y_pred[batch_size:].cpu().numpy()

    y_pred[np.argwhere(y_pred > 0.5)] = 1
    y_pred[np.argwhere(y_pred <= 0.5)] = 0
    return y_pred.astype(int)


def kFoldModelEval(model=None, names=None, dataloader=None, num_obs=None, batch_size=None, device=None, path=None):
    y_pred = np.zeros((num_obs, 1))

    for name in names:
        model.load_state_dict(torch.load(path + name))
        model.eval()
        Pred = torch.zeros(batch_size, 1).to(device)
        for X in dataloader:
            X = X.to(device)
            with torch.no_grad():
                output = torch.sigmoid(model(X))
            Pred = torch.cat((Pred, output), dim=0)
        Pred = Pred[batch_size:].cpu().numpy()
        y_pred = np.add(y_pred, Pred)

    y_pred = np.divide(y_pred, len(names))

    y_pred[np.argwhere(y_pred > 0.5)] = 1
    y_pred[np.argwhere(y_pred <= 0.5)] = 0
    return y_pred.astype(int)
