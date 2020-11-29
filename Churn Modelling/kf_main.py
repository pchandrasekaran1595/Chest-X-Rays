import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import train_val as tv
from MLP import MLP
from Dataset import Dataset

import torch
from torch import nn
from torch.utils.data import DataLoader as DL

root_path = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Churn Modelling/"
model_path = "C:/Users/Ourself/Desktop/Machine Learning/Projects/Churn Modelling/Models/"

class CFG():
    tr_batch_size = 64
    va_batch_size = 64
    ts_batch_size = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OL = 1

    def __init__(this, IL=None, HL=None, epochs=None, n_folds=None, use_DP=False, DP1=0.2, DP2=0.5):
        this.IL = IL
        this.HL = HL
        this.epochs = epochs
        this.n_folds = n_folds
        this.use_DP = use_DP
        this.DP1 = DP1
        this.DP2 = DP2


def breaker():
    print("\n" + 50 * "-" + "\n")


def head(x=None, no_of_ele=5):
    breaker()
    print(x[:no_of_ele])
    breaker()


def getCol(x=None):
    return [col for col in x.columns]


def getObj(x=None):
    s = (x.dtypes == "object")
    return list(s[s].index)

def preprocess(x=None, *args):
    df = x.copy()
    df[args[0]] = df[args[0]].map({'Female' : 0, 'Male' : 1})
    df[args[1]] = df[args[1]].map({'No' : 0, 'Yes' : 1})
    df[args[2]] = df[args[2]].map({'No' : 0, 'Yes' : 1})
    df[args[3]] = df[args[3]].map({'No' : 0, 'Yes' : 1})
    df[args[4]] = df[args[4]].map({'No' : 0, 'No phone service' : 1, 'Yes' : 2})
    df[args[5]] = df[args[5]].map({'DSL' : 0, 'Fiber optic' : 1, 'No' : 2})
    df[args[6]] = df[args[6]].map({'No' : 0, 'No internet service' : 1, 'Yes' : 2})
    df[args[7]] = df[args[7]].map({'No' : 0, 'No internet service' : 1, 'Yes' : 2})
    df[args[8]] = df[args[8]].map({'No' : 0, 'No internet service' : 1, 'Yes' : 2})
    df[args[9]] = df[args[9]].map({'No' : 0, 'No internet service' : 1, 'Yes' : 2})
    df[args[10]] = df[args[10]].map({'No' : 0, 'No internet service' : 1, 'Yes' : 2})
    df[args[11]] = df[args[11]].map({'No' : 0, 'No internet service' : 1, 'Yes' : 2})
    df[args[12]] = df[args[12]].map({'Month-to-month' : 0, 'One year' : 1, 'Two year' : 2})
    df[args[13]] = df[args[13]].map({'No' : 0, 'Yes' : 1})
    df[args[14]] = df[args[14]].map({'Bank transfer (automatic)' : 0,
                                     'Credit card (automatic)' : 1,
                                     'Electronic check' : 2,
                                     'Mailed check' : 3})
    return df


si_mu = SimpleImputer(missing_values=np.nan, strategy="mean")
si_mf = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
sc_X = StandardScaler()

if __name__ == "__main__":

    tr_features = pd.read_csv(root_path + "train.csv")
    ts_features = pd.read_csv(root_path + "test.csv")

    tr_features = tr_features.drop(labels="id", axis=1)
    ts_features = ts_features.drop(labels="id", axis=1)

    tr_features["TotalCharges"] = pd.to_numeric(tr_features["TotalCharges"], errors="coerce")
    ts_features["TotalCharges"] = pd.to_numeric(ts_features["TotalCharges"], errors="coerce")

    tr_features = preprocess(tr_features, 'gender', 'Partner', 'Dependents', 'PhoneService',
                             'MultipleLines', 'InternetService', 'OnlineSecurity',
                             'OnlineBackup', 'DeviceProtection', 'TechSupport',
                             'StreamingTV', 'StreamingMovies', 'Contract',
                             'PaperlessBilling', 'PaymentMethod')

    ts_features = preprocess(ts_features, 'gender', 'Partner', 'Dependents', 'PhoneService',
                             'MultipleLines', 'InternetService', 'OnlineSecurity',
                             'OnlineBackup', 'DeviceProtection', 'TechSupport',
                             'StreamingTV', 'StreamingMovies', 'Contract',
                             'PaperlessBilling', 'PaymentMethod')

    X = tr_features.iloc[:, :-1].copy().values
    y = tr_features.iloc[:, -1].copy().values

    X_test = ts_features.copy().values

    X = si_mf.fit_transform(X)
    X_test = si_mf.transform(X_test)

    X = sc_X.fit_transform(X)
    X_test = sc_X.transform(X_test)

    cfg = CFG(IL=X.shape[1], HL=[256, 256], epochs=50, n_folds=10, use_DP=True, DP1=0.2, DP2=0.5)

    Names, LP = tv.kFoldModelTrain(X=X, y=y, epochs=cfg.epochs, n_folds=cfg.n_folds,
                                   IL=cfg.IL, HL=cfg.HL, OL=cfg.OL, use_DP=cfg.use_DP, DP1=cfg.DP1, DP2=cfg.DP2,
                                   lr=1e-3, wd=1e-5, patience=4, lr_eps=1e-8, use_all=True,
                                   tr_batch_size=cfg.tr_batch_size,
                                   va_batch_size=cfg.va_batch_size,
                                   criterion=nn.BCEWithLogitsLoss(),
                                   device=cfg.device, path=model_path)

    LPV = []
    LPT = []
    for i in range(len(LP)):
        LPT.append(LP[i]["train"])
        LPV.append(LP[i]["valid"])

    """xAxis = [i+1 for i in range(cfg.epochs)]
    plt.figure(figsize=(20, 20))
    for fold in range(cfg.n_folds):
        plt.subplot(cfg.n_folds, 1, fold + 1)
        plt.plot(xAxis, LPT[fold * cfg.epochs:(fold + 1) * cfg.epochs], "b", label="Training Loss")
        plt.plot(xAxis, LPV[fold * cfg.epochs:(fold + 1) * cfg.epochs], "r--", label="Validation Loss")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Fold {fold}".format(fold=fold + 1))
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()"""

    torch.manual_seed(0)
    Model = MLP(cfg.IL, cfg.HL, cfg.OL, use_DP=cfg.use_DP)

    ts_data_setup = Dataset(X_test, None, "test")
    ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

    y_pred = tv.kFoldModelEval(Model, set(Names), ts_data, ts_data_setup.__len__(), cfg.ts_batch_size,
                               cfg.device, model_path)

    ss = pd.read_csv(root_path + "sample_submission.csv")
    ss["Churn"] = y_pred
    ss.to_csv("./submission.csv", index=False)

    print("Program Execution Complete")
    breaker()
