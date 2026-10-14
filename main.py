"""Compatibility entry point for the original scripts; configuration is loaded only when run."""
import pandas as pd


def _config():
    try:
        import config
    except ImportError as exc:
        raise RuntimeError("Legacy training needs a local config.py; copy config.example.py and add company_names. Credentials are not needed for training.") from exc
    return config


def _company_names(config):
    names = getattr(config, "company_names", None) or getattr(config, "COMPANIES", None)
    if not names:
        raise RuntimeError("Legacy config.py must define company_names or COMPANIES.")
    return names

def chronological_splits(data, train_fraction=.8, validation_fraction=.2):
    """Split sorted rows into past-only fit, validation, and final test blocks."""
    data = data.sort_index()
    train_end = int(len(data) * train_fraction)
    train_validation = data.iloc[:train_end]
    fit_end = int(len(train_validation) * (1 - validation_fraction))
    return train_validation.iloc[:fit_end], train_validation.iloc[fit_end:], data.iloc[train_end:]

def train_models():
    config = _config()
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import LinearSVR
    from xgboost import XGBRegressor
    from linear_pipeline import model_results, model_training
    from lstm import lstm_trainer
    from lstm.lstm_pipeline import lstm_results
    from sklearn.preprocessing import StandardScaler
    for company_name in _company_names(config):
        company_sentiment = pd.read_csv(f"data/sentiment/{company_name}.csv", index_col=0)
        company_data = pd.read_csv(f"data/stock_prices/{company_name}_stock_price.csv", index_col=0)
        company_data['Sentiment'] = company_sentiment.iloc[:, 0].to_numpy()
        company_data['Label'] = company_data['Close'].shift(-1)
        company_data = company_data.dropna().sort_index().round(5)
        fit_data, val_data, test_data = chronological_splits(company_data)
        model_x = ['Close', 'Sentiment']
        for name, estimator in {"Linear": LinearRegression(), "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, n_jobs=1), "SVR": LinearSVR(random_state=11, max_iter=5000)}.items():
            model_training(fit_data[model_x], fit_data.Label, company_name, name, estimator)
            for data, mode in ((fit_data, "train"), (val_data, "val"), (test_data, "test")):
                model_results(company_name, data[model_x], data.Label, mode, name)
        lstm_trainer.train(fit_data[model_x].copy(), val_data[model_x].copy(), fit_data.Label.copy(), val_data.Label.copy(), company_name)
        for data, mode in ((fit_data, "train"), (val_data, "val"), (test_data, "test")):
            lstm_results(company_name, data[model_x], data.Label, mode)

if __name__ == "__main__":
    train_models()
