from torch import nn, optim
import torch.nn.functional as F


class CNN_8BN(nn.Module):
    def __init__(this, in_channels=None, filter_sizes=None, HL=None, OL=None, AP_size=None, DP=0.5):

        super(CNN_8BN, this).__init__()

        this.DP_ = nn.Dropout(p=DP)
        this.AP_ = nn.AdaptiveAvgPool2d(output_size=AP_size)
        this.MP_ = nn.MaxPool2d(kernel_size=2)

        this.CN1 = nn.Conv2d(in_channels=in_channels, out_channels=filter_sizes[0], kernel_size=3, stride=1, padding=1)
        this.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        this.CN2 = nn.Conv2d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=3, stride=1,
                             padding=1)
        this.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        this.CN3 = nn.Conv2d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=3, stride=1,
                             padding=1)
        this.BN3 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        this.CN4 = nn.Conv2d(in_channels=filter_sizes[2], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                             padding=1)
        this.BN4 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN5 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                             padding=1)
        this.BN5 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.FC1 = nn.Linear(in_features=filter_sizes[3] * (AP_size ** 2), out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=HL[1])
        this.FC3 = nn.Linear(in_features=HL[1], out_features=OL)

    def getOptimizer(this, A_S=True, lr=1e-3, wd=0):
        if A_S:
            return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.SGD(this.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(this, x):
        x = F.relu(this.MP_(this.BN1(this.CN1(x))))
        x = F.relu(this.MP_(this.BN2(this.CN2(x))))
        x = F.relu(this.MP_(this.BN3(this.CN3(x))))
        x = F.relu(this.MP_(this.BN4(this.CN4(x))))
        x = F.relu(this.MP_(this.BN5(this.CN5(x))))

        x = this.AP_(x)
        x = x.view(x.shape[0], -1)

        x = F.relu(this.DP_(this.FC1(x)))
        x = F.relu(this.DP_(this.FC2(x)))
        x = F.log_softmax(this.FC3(x), dim=1)

        return x


#########################################################################################################

class CNN_11BN(nn.Module):
    def __init__(this, in_channels=None, filter_sizes=None, HL=None, OL=None, AP_size=None, DP=0.5):

        super(CNN_11BN, this).__init__()

        this.DP_ = nn.Dropout(p=DP)
        this.AP_ = nn.AdaptiveAvgPool2d(output_size=AP_size)
        this.MP_ = nn.MaxPool2d(kernel_size=2)

        this.CN1 = nn.Conv2d(in_channels=in_channels, out_channels=filter_sizes[0], kernel_size=3, stride=1, padding=1)
        this.BN1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        this.CN2 = nn.Conv2d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=3, stride=1,
                             padding=1)
        this.BN2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        this.CN3_1 = nn.Conv2d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=3, stride=1,
                               padding=1)
        this.BN3_1 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)
        this.CN3_2 = nn.Conv2d(in_channels=filter_sizes[2], out_channels=filter_sizes[2], kernel_size=3, stride=1,
                               padding=1)
        this.BN3_2 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        this.CN4_1 = nn.Conv2d(in_channels=filter_sizes[2], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN4_1 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        this.CN4_2 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN4_2 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN5_1 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN5_1 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        this.CN5_2 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN5_2 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.FC1 = nn.Linear(in_features=filter_sizes[3] * (AP_size ** 2), out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=HL[1])
        this.FC3 = nn.Linear(in_features=HL[1], out_features=OL)

    def getOptimizer(this, A_S=True, lr=1e-3, wd=0):
        if A_S:
            return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.SGD(this.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(this, x):
        x = F.relu(this.MP_(this.BN1(this.CN1(x))))
        x = F.relu(this.MP_(this.BN2(this.CN2(x))))
        x = F.relu(this.MP_(this.BN3_2(this.CN3_2(F.relu(this.BN3_1(this.CN3_1(x)))))))
        x = F.relu(this.MP_(this.BN4_2(this.CN4_2(F.relu(this.BN4_1(this.CN4_1(x)))))))
        x = F.relu(this.MP_(this.BN5_2(this.CN5_2(F.relu(this.BN5_1(this.CN5_1(x)))))))

        x = this.AP_(x)
        x = x.view(x.shape[0], -1)

        x = F.relu(this.DP_(this.FC1(x)))
        x = F.relu(this.DP_(this.FC2(x)))
        x = F.log_softmax(this.FC3(x), dim=1)

        return x


#########################################################################################################

class CNN_13BN(nn.Module):
    def __init__(this, in_channels=None, filter_sizes=None, HL=None, OL=None, AP_size=None, DP=0.5):

        super(CNN_13BN, this).__init__()

        this.DP_ = nn.Dropout(p=DP)
        this.AP_ = nn.AdaptiveAvgPool2d(output_size=AP_size)
        this.MP_ = nn.MaxPool2d(kernel_size=2)

        this.CN1_1 = nn.Conv2d(in_channels=in_channels, out_channels=filter_sizes[0], kernel_size=3, stride=1,
                               padding=1)
        this.BN1_1 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)
        this.CN1_2 = nn.Conv2d(in_channels=filter_sizes[0], out_channels=filter_sizes[0], kernel_size=3, stride=1,
                               padding=1)
        this.BN1_2 = nn.BatchNorm2d(num_features=filter_sizes[0], eps=1e-5)

        this.CN2_1 = nn.Conv2d(in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=3, stride=1,
                               padding=1)
        this.BN2_1 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)
        this.CN2_2 = nn.Conv2d(in_channels=filter_sizes[1], out_channels=filter_sizes[1], kernel_size=3, stride=1,
                               padding=1)
        this.BN2_2 = nn.BatchNorm2d(num_features=filter_sizes[1], eps=1e-5)

        this.CN3_1 = nn.Conv2d(in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=3, stride=1,
                               padding=1)
        this.BN3_1 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)
        this.CN3_2 = nn.Conv2d(in_channels=filter_sizes[2], out_channels=filter_sizes[2], kernel_size=3, stride=1,
                               padding=1)
        this.BN3_2 = nn.BatchNorm2d(num_features=filter_sizes[2], eps=1e-5)

        this.CN4_1 = nn.Conv2d(in_channels=filter_sizes[2], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN4_1 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        this.CN4_2 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN4_2 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.CN5_1 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN5_1 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)
        this.CN5_2 = nn.Conv2d(in_channels=filter_sizes[3], out_channels=filter_sizes[3], kernel_size=3, stride=1,
                               padding=1)
        this.BN5_2 = nn.BatchNorm2d(num_features=filter_sizes[3], eps=1e-5)

        this.FC1 = nn.Linear(in_features=filter_sizes[3] * (AP_size ** 2), out_features=HL[0])
        this.FC2 = nn.Linear(in_features=HL[0], out_features=HL[1])
        this.FC3 = nn.Linear(in_features=HL[1], out_features=OL)

    def getOptimizer(this, A_S=True, lr=1e-3, wd=0):
        if A_S:
            return optim.Adam(this.parameters(), lr=lr, weight_decay=wd)
        else:
            return optim.SGD(this.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    def getMultiStepLR(this, optimizer=None, milestones=None, gamma=0.1):
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

    def getPlateauLR(this, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps, verbose=True)

    def forward(this, x):
        x = F.relu(this.MP_(this.BN1_2(this.CN1_2(F.relu(this.BN1_1(this.CN1_1(x)))))))
        x = F.relu(this.MP_(this.BN2_2(this.CN2_2(F.relu(this.BN2_1(this.CN2_1(x)))))))
        x = F.relu(this.MP_(this.BN3_2(this.CN3_2(F.relu(this.BN3_1(this.CN3_1(x)))))))
        x = F.relu(this.MP_(this.BN4_2(this.CN4_2(F.relu(this.BN4_1(this.CN4_1(x)))))))
        x = F.relu(this.MP_(this.BN5_2(this.CN5_2(F.relu(this.BN5_1(this.CN5_1(x)))))))

        x = this.AP_(x)
        x = x.view(x.shape[0], -1)

        x = F.relu(this.DP_(this.FC1(x)))
        x = F.relu(this.DP_(this.FC2(x)))
        x = F.log_softmax(this.FC3(x), dim=1)

        return x