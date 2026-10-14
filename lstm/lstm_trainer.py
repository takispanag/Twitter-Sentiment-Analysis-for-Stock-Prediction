"""Legacy sequence training with train-only scaling and endpoint labels."""
import copy
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from data_serializer import save_pickle
from lstm.lstm_model import LSTM
from sentiment_stock.sequence import sequence_windows

LOOKBACK = 5

def sliding_windows(features, labels, batch_size, window_step=1):
    x, offsets = sequence_windows(np.asarray(features, dtype=float), int(batch_size))
    y = np.asarray(labels, dtype=float).reshape(-1)
    if len(y) != len(features):
        raise ValueError("features and labels must have equal row counts")
    return x[::window_step], y[offsets][::window_step, None]

def train(x_train, x_val, y_train, y_val, company):
    x_train, x_val = np.asarray(x_train, dtype=float), np.asarray(x_val, dtype=float)
    y_train, y_val = np.asarray(y_train, dtype=float).reshape(-1), np.asarray(y_val, dtype=float).reshape(-1)
    if len(x_train) != len(y_train) or len(x_val) != len(y_val):
        raise ValueError("features and labels must have equal row counts")
    if len(x_train) < LOOKBACK or len(x_val) < LOOKBACK:
        raise ValueError(f"legacy LSTM requires at least {LOOKBACK} train and validation rows")
    scaler = StandardScaler().fit(x_train)
    tx, ty = sliding_windows(scaler.transform(x_train), y_train, LOOKBACK)
    vx, vy = sliding_windows(scaler.transform(x_val), y_val, LOOKBACK)
    torch.manual_seed(11)
    model = LSTM(tx.shape[2], 32, 1, 1)
    optimizer, loss_fn = Adam(model.parameters(), lr=.001), nn.L1Loss()
    loader = DataLoader(TensorDataset(torch.tensor(tx, dtype=torch.float32), torch.tensor(ty, dtype=torch.float32)), batch_size=16, shuffle=False)
    vx_t, vy_t = torch.tensor(vx, dtype=torch.float32), torch.tensor(vy, dtype=torch.float32)
    best_loss, best_state, patience = float("inf"), None, 0
    for _ in range(50):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(); loss_fn(model(xb), yb).backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_loss = float(loss_fn(model(vx_t), vy_t))
        if val_loss < best_loss:
            best_loss, patience, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= 6: break
    model.load_state_dict(best_state)
    save_pickle(company, "lstm", model, "models")
    save_pickle(company, "feature_scaler", scaler, "scalers")
    return model
