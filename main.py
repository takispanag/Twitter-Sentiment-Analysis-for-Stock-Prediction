from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import config
from data_preprocessing import featurization, inference_featurization
import yaml
from data_serializer import save_pickle, load_pickle
from lstm import lstm_trainer
from lstm.lstm_inference import lstm_results


def inference():
    for company_name in config.company_names:
        data = inference_featurization(company_name)
        # lstm_results(company_name, data)
        data['Prediction'] = load_pickle(company_name, "linear", "models").predict(data.values.reshape(-1, 2)).reshape(-1, 1)
        data['Prediction'] = data['Prediction'].shift(1)
        data = data.dropna()
        print(f"{company_name}: MAE = {mean_absolute_error(data['Close'], data['Prediction'])}, MAP = {mean_absolute_percentage_error(data['Close'], data['Prediction'])}")


def train_models():
    try:
        with open('hyperparameter.yaml', 'r') as file:
            params_list = yaml.safe_load(file)
    except Exception as e:
        print(e)

    for company_name in config.company_names:
        featurization(company_name)
        df = pd.read_csv(f"data/training_data/{company_name}_data.csv", index_col=0)
        # print(f"Correlation:\n {df.corr()}")
        for model, param in zip([LinearRegression()],config.models):
            model = GridSearchCV(model, params_list[param])
            x_train, x_test, y_train, y_test = train_test_split(df[["Sentiment", "Close"]], df["Next_Close"],
                                                                train_size=0.75, random_state=11)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)

            save_pickle(company_name, param, model.best_estimator_, "models")

        #train lstm
        # lstm_trainer.train(df,company_name)
        lstm_results(company_name,df)

if __name__ == "__main__":
    # train_models()
    inference()
