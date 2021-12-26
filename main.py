from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import config
from data_preprocessing import featurization, inference_featurization
import yaml
from data_serializer import save_model, load_model
from lstm import lstm_trainer


def inference():
    for company_name in config.company_names:
        data = inference_featurization(company_name)
        for model in config.models:
            data['Prediction'] = load_model(company_name, model).predict(data.values.reshape(-1, 2)).reshape(-1, 1)
            data['Prediction'] = data['Prediction'].shift(1)
            data = data.dropna()
            print(f"{company_name}: MAE = {mean_absolute_error(data['Close'], data['Prediction'])}, MAP = {mean_absolute_percentage_error(data['Close'], data['Prediction'])}")


def train_models():
    try:
        with open('hyperparameter.yaml', 'r') as file:
            params_list = yaml.safe_load(file)
    except Exception as e:
        print(e)

    for company_name in ["AMZN", "AAPL", "GOOG", "MSFT", "TSLA"]:
        featurization(company_name)
        df = pd.read_csv(f"data/training_data/{company_name}_data.csv", index_col=0)
        # print(f"Correlation:\n {df.corr()}")
        for model, param in zip([LinearRegression()],config.models):
            model = GridSearchCV(model, params_list[param])
            x_train, x_test, y_train, y_test = train_test_split(df[["Sentiment", "Close"]], df["Next_Close"],
                                                                train_size=0.75)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            save_model(company_name, param, model)


if __name__ == "__main__":
    # train_models()
    lstm_trainer.train(pd.read_csv(f"data/training_data/AAPL_data.csv", index_col=0))
