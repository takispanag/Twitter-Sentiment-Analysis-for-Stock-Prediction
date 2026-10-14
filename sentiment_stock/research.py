"""Deterministic, date-aligned walk-forward sentiment research."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
import joblib

from .features import ATTENTION_FEATURES, PRICE_FEATURES, SENTIMENT_FEATURES
from .sequence import sequence_windows

LOOKBACK = 5

TARGETS = {"return": "target_return", "next_close": "next_close", "volatility": "target_volatility"}
FEATURE_SETS = {
    "price_only": PRICE_FEATURES,
    "attention_only": ATTENTION_FEATURES,
    "price_attention": PRICE_FEATURES + ATTENTION_FEATURES,
    "price_sentiment": PRICE_FEATURES + ATTENTION_FEATURES + SENTIMENT_FEATURES,
    "shuffled_sentiment": PRICE_FEATURES + ATTENTION_FEATURES + SENTIMENT_FEATURES,
}


def _target_column(features: pd.DataFrame) -> str:
    kinds = features.get("target_kind", pd.Series("return", index=features.index)).dropna().unique()
    kind = str(kinds[0]) if len(kinds) else "return"
    if kind not in TARGETS:
        raise ValueError(f"unsupported target_kind: {kind}")
    return TARGETS[kind]


def _estimator(name: str, seed: int):
    if name == "linear":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), LinearRegression())
    if name == "linear_svr":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler(), LinearSVR(random_state=seed, max_iter=20000, dual="auto", tol=5e-2))
    if name == "xgboost":
        from xgboost import XGBRegressor
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), XGBRegressor(n_estimators=100, max_depth=3, learning_rate=.04, subsample=.9, colsample_bytree=.9, reg_lambda=1, random_state=seed, n_jobs=1, objective="reg:squarederror"))
    raise ValueError(name)


def _lstm_predict(train_x, train_y, test_x, seed: int, context_x=None, artifact_path=None, feature_columns=None):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    medians = np.nanmedian(train_x, axis=0)
    medians[~np.isfinite(medians)] = 0
    train_x = np.where(np.isfinite(train_x), train_x, medians)
    test_x = np.where(np.isfinite(test_x), test_x, medians)
    context_x = np.asarray(context_x, dtype=float) if context_x is not None else np.empty((0, train_x.shape[1]))
    context_x = np.where(np.isfinite(context_x), context_x, medians)
    mean, std = train_x.mean(0), train_x.std(0)
    std[std == 0] = 1
    train_x = (train_x - mean) / std
    context_test_x = (np.concatenate([context_x, test_x]) - mean) / std
    train_windows, train_offsets = sequence_windows(train_x, LOOKBACK)
    test_windows, _ = sequence_windows(context_test_x, LOOKBACK)
    train_y = np.asarray(train_y, dtype=float)[train_offsets]
    xt = torch.tensor(train_windows, dtype=torch.float32)
    yt = torch.tensor(train_y, dtype=torch.float32)
    xv = torch.tensor(test_windows, dtype=torch.float32)
    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(train_x.shape[1], 12, batch_first=True)
            self.head = nn.Linear(12, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)
    model = Regressor()
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=.01), nn.L1Loss()
    split = max(1, int(len(xt) * .85))
    best, patience, best_state = float("inf"), 0, None
    for _ in range(20):
        model.train()
        for start in range(0, split, 16):
            optimizer.zero_grad()
            loss = loss_fn(model(xt[start:start + 16]), yt[start:start + 16])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_x = xt[split:] if split < len(xt) else xt[-1:]
            validation_y = yt[split:] if split < len(xt) else yt[-1:]
            val_loss = float(loss_fn(model(validation_x), validation_y))
        if val_loss < best - 1e-8:
            best, patience, best_state = val_loss, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(xv).numpy()
    if artifact_path is not None:
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "input_medians": torch.tensor(medians), "input_mean": torch.tensor(mean), "input_std": torch.tensor(std), "lookback": LOOKBACK, "hidden_size": 12, "feature_columns": list(feature_columns or [])}, artifact_path)
    return predictions


def _model_names(include_models: bool) -> list[str]:
    names = ["linear", "linear_svr"]
    if include_models:
        try:
            import xgboost  # noqa: F401
            names.append("xgboost")
        except ImportError:
            pass
        try:
            import torch  # noqa: F401
            names.append("lstm")
        except ImportError:
            pass
    return names


def _safe_features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    absent = set(columns) - set(frame.columns)
    if absent:
        raise ValueError(f"feature columns missing: {', '.join(sorted(absent))}")
    return frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _fit_predict(name, train, train_y, test, columns, seed, context=None, artifact_path=None):
    if name == "lstm":
        return _lstm_predict(_safe_features(train, columns), train_y, _safe_features(test, columns), seed, _safe_features(context, columns) if context is not None and len(context) else None, artifact_path=artifact_path, feature_columns=columns)
    estimator = _estimator(name, seed)
    estimator.fit(_safe_features(train, columns), train_y)
    return estimator.predict(_safe_features(test, columns)), estimator


def _folds(group: pd.DataFrame, n_splits: int):
    eligible = group.loc[group[_target_column(group)].notna()].sort_values("feature_date", kind="stable")
    if len(eligible) < n_splits * (LOOKBACK + 1) + 3:
        return
    first_test = max(3, len(eligible) // (n_splits + 1))
    test_blocks = np.array_split(np.arange(first_test, len(eligible)), n_splits)
    for fold_num, block in enumerate(test_blocks, start=1):
        if not len(block):
            continue
        raw_test = eligible.iloc[block]
        if len(raw_test) <= LOOKBACK - 1:
            continue
        context = raw_test.iloc[:LOOKBACK - 1]
        test = raw_test.iloc[LOOKBACK - 1:]
        cutoff = pd.to_datetime(test.issued_at.iloc[0], utc=True)
        train = eligible.iloc[:block[0]]
        if "target_close_at" in train:
            train = train.loc[pd.to_datetime(train.target_close_at, utc=True) < cutoff]
        if len(train) < 3:
            continue
        yield fold_num, train, test, context


def _bootstrap_delta(values: np.ndarray, seed: int, block_length: int = 5) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    if len(values) < 3:
        point = float(values.mean())
        if abs(point) <= 1e-12:
            return 0.0, 0.0, 0.0
        return point, point, point
    samples = []
    for _ in range(1000):
        chosen = []
        while len(chosen) < len(values):
            start = int(rng.integers(0, len(values)))
            chosen.extend(values[(start + np.arange(block_length)) % len(values)])
        samples.append(np.mean(chosen[:len(values)]))
    point, low, high = float(values.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))
    if max(abs(point), abs(low), abs(high)) <= 1e-12:
        return 0.0, 0.0, 0.0
    return point, low, high


def run_research(features: pd.DataFrame, output: Path, *, seed: int = 11, n_splits: int = 3, include_models: bool = False, dataset_metadata: dict | None = None) -> dict:
    if n_splits < 1:
        raise ValueError("n_splits must be positive")
    frame = features.copy()
    if "feature_date" not in frame:
        frame["feature_date"] = frame["date"]
    frame["feature_date"] = pd.to_datetime(frame.feature_date).dt.normalize()
    frame = frame.sort_values(["ticker", "feature_date"], kind="stable").reset_index(drop=True)
    target_col = _target_column(frame)
    if target_col not in frame:
        raise ValueError(f"features do not contain target column {target_col}")
    for required in ("issued_at", "input_cutoff"):
        if required not in frame:
            raise ValueError(f"features missing required timestamp {required}")
    models = _model_names(include_models)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    forecast_rows, fitted = [], {}
    input_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    run_id = hashlib.sha256(f"{seed}|{input_hash}".encode()).hexdigest()[:16]
    for ticker, group in frame.groupby("ticker", sort=True):
        for fold, train, test, context in _folds(group, n_splits) or []:
            y_train = train[target_col].astype(float).to_numpy()
            train_y = np.nan_to_num(y_train, nan=0.0)
            actual = test[target_col].astype(float).to_numpy()
            naive = float(train_y.mean()) if target_col != "next_close" else float(train.next_close.iloc[-1])
            for local_i, (_, test_row) in enumerate(test.iterrows()):
                base = _forecast_row(test_row, run_id, ticker, fold, "naive", "naive", naive, actual[local_i], dataset_metadata, pd.to_datetime(train.target_close_at, utc=True).max() if "target_close_at" in train else pd.NaT)
                forecast_rows.append(base)
            for model_name in models:
                for feature_set, columns in FEATURE_SETS.items():
                    x_train, x_test = train.copy(), test.copy()
                    used_columns = columns
                    if feature_set == "shuffled_sentiment":
                        x_train.loc[:, SENTIMENT_FEATURES] = x_train[SENTIMENT_FEATURES].sample(frac=1, random_state=seed + fold * 100 + 1).to_numpy()
                        x_test.loc[:, SENTIMENT_FEATURES] = x_test[SENTIMENT_FEATURES].sample(frac=1, random_state=seed + fold * 100 + 2).to_numpy()
                    artifact_path = model_dir / f"{ticker}_lstm_{feature_set}_fold{fold}.pt" if model_name == "lstm" and feature_set == "price_sentiment" and fold == n_splits else None
                    prediction_result = _fit_predict(model_name, x_train, train_y, x_test, used_columns, seed, context=context, artifact_path=artifact_path)
                    prediction = prediction_result[0] if isinstance(prediction_result, tuple) else prediction_result
                    estimator = prediction_result[1] if isinstance(prediction_result, tuple) else None
                    if estimator is not None and fold == list(range(1, n_splits + 1))[-1] and feature_set == "price_sentiment":
                        fitted[(ticker, model_name, feature_set)] = estimator
                    for local_i, (_, test_row) in enumerate(test.iterrows()):
                        forecast_rows.append(_forecast_row(test_row, run_id, ticker, fold, model_name, feature_set, float(prediction[local_i]), actual[local_i], dataset_metadata, pd.to_datetime(train.target_close_at, utc=True).max() if "target_close_at" in train else pd.NaT))
    forecasts = pd.DataFrame(forecast_rows)
    if forecasts.empty:
        raise ValueError("not enough resolved rows for chronological walk-forward folds")
    frame.to_csv(output / "features.csv", index=False)
    forecasts.to_csv(output / "forecasts.csv", index=False)
    posts_sample = dataset_metadata.get("posts_sample") if dataset_metadata else None
    sample_columns = ["post_id", "ticker", "created_at", "available_at", "text", "author", "sentiment_score", "source"]
    sample_frame = pd.DataFrame(posts_sample) if posts_sample is not None else pd.DataFrame(columns=sample_columns)
    sample_frame.to_csv(output / "posts_sample.csv", index=False)
    errors = forecasts.dropna(subset=["actual"]).copy()
    metric_rows = []
    for keys, group in errors.groupby(["ticker", "model", "feature_set", "fold", "event_day"], dropna=False):
        ticker, model, feature_set, fold, event_day = keys
        err = group.prediction.to_numpy() - group.actual.to_numpy()
        metric_rows.append({"ticker": ticker, "model": model, "feature_set": feature_set, "fold": fold, "n": len(group), "event_day": int(event_day), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))), "directional_accuracy": float(np.mean(np.sign(group.prediction) == np.sign(group.actual))) if target_col != "next_close" else np.nan, "no_post_share": float(pd.to_numeric(group.sentiment_missing, errors="coerce").fillna(1).mean())})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    regime = errors.groupby(["ticker", "model", "feature_set", "fold", "volatility_regime"], dropna=False).apply(lambda g: pd.Series({"n": len(g), "mae": mean_absolute_error(g.actual, g.prediction), "rmse": mean_squared_error(g.actual, g.prediction) ** .5}), include_groups=False).reset_index()
    regime.to_csv(output / "regime_metrics.csv", index=False)
    comparison = _comparisons(errors, seed)
    comparison.to_csv(output / "comparison.csv", index=False)
    for (ticker, model_name, feature_set), estimator in fitted.items():
        joblib.dump(estimator, model_dir / f"{ticker}_{model_name}_{feature_set}.joblib")
    revision, dirty = _git_state(Path(__file__).resolve().parents[1])
    source_hashes = {name: _sha_file(Path(__file__).with_name(name)) for name in ("features.py", "research.py", "sequence.py", "cli.py")}
    meta = {"run_id": run_id, "seed": seed, "n_splits": n_splits, "include_models": include_models, "model_configuration": {"linear_svr": {"max_iter": 20000, "tol": 0.05}, "lstm": {"lookback": LOOKBACK, "hidden_size": 12, "max_epochs": 20, "early_stopping_patience": 4}}, "target_column": target_col, "feature_sets": FEATURE_SETS, "feature_definitions": {"price": PRICE_FEATURES, "attention": ATTENTION_FEATURES, "sentiment": SENTIMENT_FEATURES}, "target_definitions": {"return": "next session close / next session open - 1", "next_close": "next session close price", "volatility": "absolute next session open-to-close return (single-session realized-volatility proxy)"}, "dataset_metadata": dataset_metadata or {}, "dataset_type": (dataset_metadata or {}).get("dataset_type", "unspecified"), "input_rows": len(frame), "covered_dates": [frame.feature_date.min().date().isoformat(), frame.feature_date.max().date().isoformat()], "sentiment_coverage": {"scored_session_share": float(1 - pd.to_numeric(frame.sentiment_missing, errors="coerce").fillna(1).mean()), "no_post_share": float(pd.to_numeric(frame.sentiment_missing, errors="coerce").fillna(1).mean()), "scored_date_range": [frame.loc[frame.sentiment_missing.eq(0), "feature_date"].min().date().isoformat() if frame.sentiment_missing.eq(0).any() else None, frame.loc[frame.sentiment_missing.eq(0), "feature_date"].max().date().isoformat() if frame.sentiment_missing.eq(0).any() else None]}, "input_hash": input_hash, "code_revision": revision, "working_tree_dirty": dirty, "source_hashes": source_hashes, "python": platform.python_version(), "packages": {pkg: _version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "torch"]}, "time_assumptions": "Inputs available through each market close; forecast issued 15 minutes after close; date-only posts become available at next UTC midnight."}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"output": str(output), "run_id": run_id, "forecast_rows": len(forecasts), "metric_rows": len(metrics), "models": models}


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _git_state(repo):
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None

def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _forecast_row(row, run_id, ticker, fold, model, feature_set, prediction, actual, dataset_metadata, training_cutoff=pd.NaT):
    d_type = (dataset_metadata or {}).get("dataset_type", "unspecified")
    return {"run_id": run_id, "ticker": ticker, "feature_date": row.feature_date, "issued_at": row.issued_at, "input_cutoff": row.input_cutoff, "training_cutoff": training_cutoff, "target_date": row.get("target_date"), "target_open_at": row.get("target_open_at"), "target_close_at": row.get("target_close_at"), "target_open": row.get("next_open"), "target_close": row.get("next_close"), "target_kind": row.get("target_kind", "return"), "model": model, "feature_set": feature_set, "fold": fold, "prediction": prediction, "actual": actual, "actual_return": row.get("target_return"), "volatility_regime": row.get("volatility_regime", "unknown"), "sentiment_missing": row.get("sentiment_missing", 1), "post_count": row.get("post_count", 0), "event_day": row.get("event_day", 0), "dataset_type": d_type}


def _comparisons(errors, seed):
    rows = []
    naive = errors.loc[errors.model.eq("naive")].set_index(["ticker", "fold", "feature_date"])
    scored = errors.loc[~errors.model.eq("naive")]
    for (ticker, model, fold), group in scored.groupby(["ticker", "model", "fold"]):
        values = group.pivot_table(index="feature_date", columns="feature_set", values=["actual", "prediction"], aggfunc="first")
        needed = [("prediction", "price_sentiment"), ("prediction", "price_attention"), ("actual", "price_sentiment")]
        if not all(col in values for col in needed):
            continue
        actual = values[("actual", "price_sentiment")].to_numpy()
        mae_att = np.abs(values[("prediction", "price_attention")].to_numpy() - actual)
        mae_sent = np.abs(values[("prediction", "price_sentiment")].to_numpy() - actual)
        baseline_rows = naive.loc[(ticker, fold)]
        baseline_rows = baseline_rows.reindex(values.index)
        mae_naive = np.abs(baseline_rows.prediction.to_numpy(dtype=float) - actual)
        delta, low, high = _bootstrap_delta(mae_att - mae_sent, seed + int(fold))
        rows.append({"ticker": ticker, "model": model, "fold": fold, "n": len(actual), "no_post_share": float(pd.to_numeric(group.sentiment_missing, errors="coerce").fillna(1).mean()), "price_attention_mae": float(mae_att.mean()), "price_sentiment_mae": float(mae_sent.mean()), "naive_mae": float(mae_naive.mean()), "sentiment_mae_improvement_vs_price_attention": delta, "bootstrap_95_low": low, "bootstrap_95_high": high, "block_length": 5, "bootstrap_seed": seed + int(fold), "interval_method": "paired time-block bootstrap of price_attention MAE minus price_sentiment MAE", "evidence": "no meaningful difference within 1e-12 MAE absolute tolerance" if delta == 0 and low == 0 and high == 0 else ("uncertain; 95% MAE-improvement interval includes zero" if low <= 0 <= high else "95% MAE-improvement interval excludes zero; retrospective data caveats apply")})
    return pd.DataFrame(rows)


def forecast_latest(prices: pd.DataFrame, posts: pd.DataFrame, output: Path, *, as_of: pd.Timestamp, model: str = "linear", feature_set: str = "price_sentiment", seed: int = 11) -> pd.DataFrame:
    """Train on resolved targets and forecast each ticker's latest closed session."""
    from .features import build_features
    import pandas_market_calendars as mcal
    if model not in {"linear", "linear_svr"}:
        raise ValueError("latest forecast supports linear and linear_svr models")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set: {feature_set}")
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    as_of = as_of.tz_convert("UTC")
    built = build_features(prices, posts, target="return")
    built["issued_at"] = pd.to_datetime(built.issued_at, utc=True)
    built["target_close_at"] = pd.to_datetime(built.target_close_at, utc=True)
    columns = FEATURE_SETS[feature_set]
    records = []
    for ticker, ticker_rows in built.groupby("ticker", sort=True):
        closed = ticker_rows.loc[ticker_rows.issued_at <= as_of]
        if closed.empty:
            raise ValueError(f"no fully closed feature session is available for {ticker} by as_of")
        latest = closed.iloc[[-1]].copy()
        session = pd.Timestamp(latest.feature_date.iloc[0]).date()
        schedule = mcal.get_calendar("NYSE").schedule(start_date=(session + pd.Timedelta(days=1)).isoformat(), end_date=(session + pd.Timedelta(days=10)).isoformat())
        if not len(schedule):
            raise ValueError(f"could not find the next NYSE session for {ticker}")
        target_open_at, target_close_at = schedule.iloc[0].market_open, schedule.iloc[0].market_close
        issued_at = latest.issued_at.iloc[0]
        if issued_at > as_of or as_of >= target_open_at:
            raise ValueError(f"as_of must satisfy issue time <= as_of < next session open for {ticker}")
        resolved = ticker_rows.loc[ticker_rows.target_return.notna() & (ticker_rows.target_close_at <= as_of)]
        if len(resolved) < 10:
            raise ValueError(f"at least ten resolved historical targets are required for {ticker}")
        estimator = _estimator(model, seed)
        estimator.fit(_safe_features(resolved, columns), resolved.target_return.astype(float))
        prediction = float(estimator.predict(_safe_features(latest, columns))[0])
        target_date = pd.Timestamp(schedule.index[0]).normalize()
        records.append({"run_id": hashlib.sha256(f"forward|{seed}|{ticker}|{issued_at.isoformat()}|{model}|{feature_set}".encode()).hexdigest()[:16], "ticker": ticker, "feature_date": latest.feature_date.iloc[0], "issued_at": issued_at, "input_cutoff": latest.input_cutoff.iloc[0], "training_cutoff": resolved.target_close_at.max(), "target_date": target_date, "target_open_at": target_open_at, "target_close_at": target_close_at, "target_kind": "return", "model": model, "feature_set": feature_set, "fold": "forward", "prediction": prediction, "volatility_regime": latest.volatility_regime.iloc[0], "sentiment_missing": latest.sentiment_missing.iloc[0], "post_count": latest.post_count.iloc[0], "dataset_type": "forward"})
        model_dir = Path(output) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, model_dir / f"{ticker}_{model}_{feature_set}.joblib")
    result = pd.DataFrame(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "forecast.csv", index=False)
    metadata = {"forecasts": records, "training_rows_by_ticker": {str(row["ticker"]): int(built.loc[(built.ticker == row["ticker"]) & built.target_return.notna() & (pd.to_datetime(built.target_close_at, utc=True) <= as_of)].shape[0]) for row in records}, "feature_columns": columns, "target_definition": "next session open-to-close simple return", "actuals_included": False, "training_rule": "Only target closes at or before as_of are eligible.", "packages": {pkg: _version(pkg) for pkg in ["numpy", "pandas", "scikit-learn"]}, "time_assumptions": "Inputs are available through feature-session close; issue is 15 minutes later; as_of must be before next NYSE open.", "input_hash": hashlib.sha256(pd.util.hash_pandas_object(built, index=True).values.tobytes()).hexdigest(), "python": platform.python_version(), "packages": {pkg: _version(pkg) for pkg in ["numpy", "pandas", "scikit-learn"]}, "code_revision": _git_state(Path(__file__).resolve().parents[1])[0], "source_hashes": {**{name: _sha_file(Path(__file__).with_name(name)) for name in ("features.py", "research.py", "sequence.py", "cli.py")}, "lstm_model.py": _sha_file(Path(__file__).resolve().parents[1] / "lstm" / "lstm_model.py"), "main.py": _sha_file(Path(__file__).resolve().parents[1] / "main.py")}, "model": model, "feature_set": feature_set, "seed": seed}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return result
