import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from Dataset import Dataset
from MLP import MLP
import train_val as tv

import torch
from torch import nn
from torch.utils.data import DataLoader as DL


class CFG():
    tr_batch_size = 64
    ts_batch_size = 256

    epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OL = 1
    def __init__(this, IL=None, HL=None):
        this.IL = IL
        this.HL = HL


def breaker():
    print("\n" + 50*"-" + "\n")

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

    tr_features = pd.read_csv("C:/Users/Ourself/Desktop/Machine Learning/Projects/Churn Modelling/train.csv")
    ts_features = pd.read_csv("C:/Users/Ourself/Desktop/Machine Learning/Projects/Churn Modelling/test.csv")

    tr_features = tr_features.drop(labels="id", axis=1)
    ts_features = ts_features.drop(labels="id", axis=1)

    tr_features["TotalCharges"] = pd.to_numeric(tr_features["TotalCharges"], errors="coerce")
    ts_features["TotalCharges"] = pd.to_numeric(ts_features["TotalCharges"], errors="coerce")

    #tr_objcols = getObj(tr_features)
    #ts_objcols = getObj(ts_features)

    #for col_name in tr_objcols:
    #    print(col_name + " -", end=" ")
    #    print(set(tr_features[col_name]))

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

    #HL = int(input("Enter the number of neurons in HL : "))
    HL = [128, 64]

    cfg = CFG(IL=X.shape[1], HL=HL)

    tr_data_setup = Dataset(X, y.reshape(-1,1))
    tr_data = DL(tr_data_setup, batch_size=cfg.tr_batch_size, shuffle=True, generator=torch.manual_seed(0))

    model = MLP(cfg.IL, cfg.HL, cfg.OL)
    optimizer = model.getOptimizer(lr=1e-3, wd=0)

    LP = tv.singleModelTrain(model, optimizer, tr_data, 5, nn.BCEWithLogitsLoss(), cfg.device)

    plt.ion()
    plt.figure(figsize=(6,6))
    plt.plot(np.arange(len(LP)), LP, "r")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show(block=False)
    plt.pause(2.5)
    plt.close()

    ts_data_setup = Dataset(X_test, None, "test")
    ts_data = DL(ts_data_setup, batch_size=cfg.ts_batch_size, shuffle=False)

    y_pred = tv.singleModelEval(model, ts_data, cfg.ts_batch_size, cfg.device)

    ss = pd.read_csv("C:/Users/Ourself/Desktop/Machine Learning/Projects/Churn Modelling/sample_submission.csv")
    ss["Churn"] = y_pred
    ss.to_csv("./submission.csv", index=False)

    print("Program Execution Complete")
    breaker()
