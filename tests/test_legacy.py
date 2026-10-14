import importlib
import numpy as np
import pandas as pd
import pytest

from sentiment_stock.sequence import sequence_windows


def test_legacy_actuals_are_not_shifted_and_keep_matching_rows():
    import linear_pipeline
    expected = pd.Series([10., 20., 30.], index=[4, 8, 9])
    predictions = np.array([11., 18., 34.])
    actual, predicted = linear_pipeline.align_actual_predictions(expected, predictions)
    pd.testing.assert_series_equal(actual, expected)
    assert predicted.tolist() == [11., 18., 34.]


def test_lstm_regression_head_is_unbounded_and_batch_first():
    torch = pytest.importorskip("torch")
    from lstm.lstm_model import LSTM
    model = LSTM(input_size=2, hidden_size=3, num_layers=1, output_dim=1, drop=0, batch_first=True).eval()
    with torch.no_grad():
        model.seq[-1].weight.zero_()
        model.seq[-1].bias.fill_(1.5)
    x = torch.ones((2, 4, 2))
    assert model(x).shape == (2, 1)
    assert (model(x) > 1).all()
    sample = x[:1]
    alone = model(sample)
    batch = model(torch.cat([x[1:], sample], dim=0))[-1:]
    torch.testing.assert_close(alone, batch)


def test_legacy_private_config_imports_are_actionable_or_lazy():
    import main
    assert hasattr(main, "train_models")


def test_legacy_split_is_chronological_and_reserves_latest_test_block():
    import main
    data = pd.DataFrame({"value": range(10)}, index=pd.date_range("2024-01-01", periods=10))
    fit, validation, test = main.chronological_splits(data)
    assert fit.index.max() < validation.index.min() < test.index.min()
    assert test.index.max() == data.index.max()


def test_legacy_company_list_accepts_example_and_private_config_names():
    from types import SimpleNamespace
    import main
    assert main._company_names(SimpleNamespace(company_names=["only-private"])) == ["only-private"]
    assert main._company_names(SimpleNamespace(COMPANIES=["example"])) == ["example"]
