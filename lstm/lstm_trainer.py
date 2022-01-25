import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from data_serializer import save_pickle
from lstm.lstm_model import LSTM
from torch.nn import L1Loss
import matplotlib.pyplot as plt
import copy

from utils import create_folder


class TimeseriesValues(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return np.array(self.features[index]), np.array(self.labels[index])


def sliding_windows(features, labels, batch_size, window_step=1):
    features_array = []
    features = features.astype(float)
    labels_array = []
    labels = labels.astype(float)
    for i in range(0, features.shape[0] - batch_size + 1, window_step):
        features_array.append(features[i: i + batch_size].values)
        labels_array.append(labels[i: i + batch_size].values)

    features_array = np.array(features_array)
    labels_array = np.array(labels_array)

    return features_array, labels_array


def train(df, company):
    batch_size = 5

    x_train, x_test, y_train, y_test = train_test_split(df[["Sentiment", "Close"]], df["Next_Close"], train_size=0.75, random_state=11)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    label_scaler = MinMaxScaler(feature_range=(0, 1))

    x_train['Close'] = feature_scaler.fit_transform(x_train['Close'].values.reshape(-1,1))
    x_test['Close'] = feature_scaler.transform(x_test['Close'].values.reshape(-1,1))

    y_train = pd.DataFrame(label_scaler.fit_transform(y_train.values.reshape(-1,1)),columns=["Next_Close"],index=y_train.index)
    y_test = pd.DataFrame(label_scaler.transform(y_test.values.reshape(-1,1)),columns=["Next_Close"],index=y_test.index)

    x_train, y_train = sliding_windows(x_train, y_train, batch_size)
    x_test, y_test = sliding_windows(x_test, y_test, batch_size)

    model = LSTM(2, 64, 2, 1)
    loss_function = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_error = np.empty(0)
    test_error = np.empty(0)

    train_loader = DataLoader(TimeseriesValues(x_train, y_train), batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(TimeseriesValues(x_test, y_test), batch_size, shuffle=False, drop_last=True)

    epoch = 0
    best_epoch = None
    while(True):
        print(f"Epoch = {epoch}")
        model.train()
        err = []
        # train model
        for j, k in train_loader:
            y_train_pred = model(j.float())
            loss = loss_function(y_train_pred.squeeze(), k.squeeze().float())
            err.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_error = np.append(train_error, (sum(err) / len(err)))

        model.eval()
        err = []
        # validate model
        for j, k in test_loader:
            y_val_pred = model(j.float())
            loss = loss_function(y_val_pred.squeeze(), k.squeeze().float())
            err.append(loss.detach().item())
        test_error = np.append(test_error, (sum(err) / len(err)))
        print(f"Test_error = {(sum(err) / len(err))}")
        if epoch > 50:
            if test_error[-1] <= test_error[50:].min():
                best_model = copy.deepcopy(model)
                best_epoch = epoch
            if best_epoch is not None and epoch-best_epoch>20 or epoch>300:
                break
        epoch+=1


    #save best model
    save_pickle(company, "lstm", best_model, "models")

    #save scalers
    save_pickle(company, "feature_scaler", feature_scaler, "scalers")
    save_pickle(company, "label_scaler", label_scaler, "scalers")

    print(f"Best_epoch = {best_epoch}, Best_error= {test_error.min()}")
    fig, ax = plt.subplots()
    ax.plot(train_error, label="Train")
    ax.plot(test_error, label="Test")
    ax.set(xlabel='Epoch', ylabel='Loss', title=f'LSTM Loss per Epoch {company}')
    plt.axvline(best_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.legend(loc="upper right")
    create_folder("charts/loss_curves")
    plt.savefig(f"charts/loss_curves/lstm_training_{company}.png")
