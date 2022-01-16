from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import pandas as pd
import config
from data_preprocessing import training_vector_creation
from data_serializer import save_pickle, load_pickle
from linear_pipeline import linear_results, linear_training
from lstm import lstm_trainer
from lstm.lstm_pipeline import lstm_results
from sentiment_analysis import get_historical_sentiment
from stock_prices_acquisition import stock_prices


def train_models():
	for company_name in config.company_names:
		training_vector_creation(company_name)
		df = pd.read_csv(f"data/training_data/{company_name}_data.csv", index_col=0)
		# linear_training(df, company_name)
		#train lstm
		train_data = df.iloc[:int(df.shape[0] * 0.9)] # get 90% of data for training/validation
		# lstm_trainer.train(train_data, company_name)
		lstm_results(company_name, train_data, "train")
		test_data = df.iloc[int(df.shape[0] * 0.9):] # get 10% of data for prediction:)
		lstm_results(company_name, test_data, "test")


if __name__ == "__main__":
	# get_historical_sentiment()
	# stock_prices()
	train_models()