"""Legacy LSTM evaluation with the same lookback and endpoint alignment as training."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_serializer import load_pickle
from sentiment_stock.sequence import sequence_windows
from utils import create_folder

LOOKBACK = 5


def lstm_results(company, data, true_next_close, mode):
    """Evaluate matching endpoint labels without shifting or dropping valid rows."""
    features = data.copy()
    actual = pd.Series(true_next_close).copy()
    if len(features) != len(actual):
        raise ValueError("features and labels must have equal row counts")
    if len(features) < LOOKBACK:
        raise ValueError(f"legacy LSTM requires at least {LOOKBACK} evaluation rows")

    scaler = load_pickle(company, "feature_scaler", "scalers")
    model = load_pickle(company, "lstm", "models")
    scaled = scaler.transform(features.to_numpy(dtype=float))
    windows, offsets = sequence_windows(scaled, LOOKBACK)
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(windows, dtype=torch.float32)
        predictions = model(tensor).cpu().numpy().reshape(-1)

    results = pd.DataFrame(
        {
            "Actual Close": actual.iloc[offsets].to_numpy(),
            "Prediction": predictions,
        },
        index=actual.index[offsets],
    )
    metrics = pd.DataFrame(
        {
            "mae": [mean_absolute_error(results["Actual Close"], results["Prediction"])],
            "rmse": [mean_squared_error(results["Actual Close"], results["Prediction"]) ** 0.5],
        }
    )
    create_folder(f"data/metrics/{mode}")
    metrics.to_csv(f"data/metrics/{mode}/lstm_metrics_{company}.csv", index=False)
    results.attrs.update(mae=metrics.loc[0, "mae"], rmse=metrics.loc[0, "rmse"])
    return results
