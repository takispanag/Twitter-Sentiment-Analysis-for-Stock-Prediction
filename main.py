from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
import pandas as pd
import config
from linear_pipeline import model_results, model_training
from lstm import lstm_trainer
from lstm.lstm_pipeline import lstm_results
from sentiment_analysis import get_historical_sentiment
from stock_prices_acquisition import stock_prices
from xgboost import XGBRegressor


def train_models():
	# Linear, XGBOOST, SVR train/test
	for company_name in config.company_names:
		company_sentiment = pd.read_csv(f"data/sentiment/{company_name}.csv", index_col=0)
		company_data = pd.read_csv(f"data/stock_prices/{company_name}_stock_price.csv", index_col=0)
		company_data['Sentiment'] = company_sentiment
		company_data['Label'] = company_data['Close'].shift(-1)
		company_data = company_data.dropna().round(5)
		# Lessen Dimensions for Non DL models
		linear_data = company_data[['Close', 'Sentiment', 'Label']]
		train_data = linear_data.iloc[:int(linear_data.shape[0] * 0.9)]  # get 90% of data for training/validation
		test_data = linear_data.iloc[int(linear_data.shape[0] * 0.9):]  # get 10% of data for prediction:)
		x_train, x_val, y_train, y_val = train_test_split(train_data.loc[:, linear_data.columns != 'Label'], train_data['Label'], train_size=0.75, random_state=11)

		models = {"Linear": LinearRegression(), "XGBoost": XGBRegressor(), "SVR": LinearSVR()}
		for model_name in models:
			model_training(x_train, y_train, company_name, model_name, models[model_name])

			model_results(company_name, x_train.sort_index(), y_train.sort_index(), "train", model_name)
			model_results(company_name, x_val.sort_index(), y_val.sort_index(), "val", model_name)
			model_results(company_name, test_data.loc[:, linear_data.columns != 'Label'], test_data['Label'], "test", model_name)
		# train lstm with mode data
		train_data = company_data.iloc[:int(company_data.shape[0] * 0.9)]  # get 90% of data for training/validation
		test_data = company_data.iloc[int(company_data.shape[0] * 0.9):]  # get 10% of data for prediction:)
		x_train, x_val, y_train, y_val = train_test_split(train_data.loc[:, company_data.columns != 'Label'], train_data['Label'], train_size=0.75, random_state=11)
		lstm_trainer.train(x_train.copy(), x_val.copy(), y_train.copy(), y_val.copy(), company_name)
		lstm_results(company_name, x_train.sort_index(), y_train.sort_index(), "train")
		lstm_results(company_name, x_val.sort_index(), y_val.sort_index(), "val")
		# test lstm
		lstm_results(company_name, test_data.loc[:, company_data.columns != 'Label'], test_data['Label'], "test")


if __name__ == "__main__":
	# get_historical_sentiment()
	# stock_prices()
	train_models()
