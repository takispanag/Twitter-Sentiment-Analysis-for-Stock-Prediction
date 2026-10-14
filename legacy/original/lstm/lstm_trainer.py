import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from data_serializer import save_pickle
from lstm.lstm_model import LSTM
from torch.nn import L1Loss
import matplotlib.pyplot as plt
import copy
import plotly.graph_objects as go
import plotly.io as pio
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


def train(x_train, x_val, y_train, y_val, company):
    batch_size = 5

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    label_scaler = MinMaxScaler(feature_range=(0, 1))
    # Transform only stock market Data a.e no Sentiment
    x_train.loc[:, x_train.columns != 'Sentiment'] = feature_scaler.fit_transform(x_train.loc[:, x_train.columns != 'Sentiment'])
    x_val.loc[:, x_val.columns != 'Sentiment'] = feature_scaler.transform(x_val.loc[:, x_val.columns != 'Sentiment'])

    y_train = pd.DataFrame(label_scaler.fit_transform(y_train.values.reshape(-1, 1)), columns=["Label"],
                           index=y_train.index)
    y_val = pd.DataFrame(label_scaler.transform(y_val.values.reshape(-1, 1)), columns=["Label"],
                          index=y_val.index)

    x_train, y_train = sliding_windows(x_train, y_train, batch_size)
    x_val, y_val = sliding_windows(x_val, y_val, batch_size)

    model = LSTM(x_train.shape[2], 64, 2, 1)
    loss_function = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=0.001)

    train_error = np.empty(0)
    val_error = np.empty(0)

    train_loader = DataLoader(TimeseriesValues(x_train, y_train), batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(TimeseriesValues(x_val, y_val), batch_size, shuffle=False, drop_last=True)

    epoch = 0
    best_epoch = None
    while True:
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
        for j, k in val_loader:
            y_val_pred = model(j.float())
            loss = loss_function(y_val_pred.squeeze(), k.squeeze().float())
            err.append(loss.detach().item())
        val_error = np.append(val_error, (sum(err) / len(err)))
        print(f"Test_error = {(sum(err) / len(err))}")
        if epoch > 50:
            if val_error[-1] <= val_error[51:].min():
                best_model = copy.deepcopy(model)
                best_epoch = epoch
            if best_epoch is not None and epoch - best_epoch > 20 or epoch > 300:
                break
        epoch += 1

    # save best model
    save_pickle(company, "lstm", best_model, "models")

    # save scalers
    save_pickle(company, "feature_scaler", feature_scaler, "scalers")
    save_pickle(company, "label_scaler", label_scaler, "scalers")

    print(f"Best_epoch = {best_epoch}, Best_error= {val_error.min()}")
    df = pd.DataFrame(train_error.reshape(-1, 1), columns=['Train Error'])
    df['Validation Error'] = val_error.reshape(-1, 1)
    layout = go.Layout(
        autosize=False,
        width=2560,
        height=1440
    )
    fig = go.Figure(layout=layout)
    fig.update_layout(template=pio.templates['plotly_dark'])
    for column, color in zip(df.columns, ["#32a88b", "#a83232"]):
        fig.add_trace(
            go.Scatter(x=df.index, y=df[column], mode='lines+markers', line=dict(color=color), name=column))
    create_folder(f"charts/loss_curves")
    fig.add_vline(best_epoch, line_dash="dash", line_color="blue")
    fig.write_html(f"charts/loss_curves/lstm_training_{company}.html")
    fig.write_image(f"charts/loss_curves/lstm_training_{company}.png")
