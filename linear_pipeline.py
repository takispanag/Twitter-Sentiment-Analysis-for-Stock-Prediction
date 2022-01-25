import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import config
from data_preprocessing import inference_vector_creation, training_vector_creation
from data_serializer import load_pickle, save_pickle
from utils import create_folder


def model_training(df, company_name, model_name, model):
	x_train, x_test, y_train, y_test = train_test_split(df[["Sentiment", "Close"]], df["Next_Close"],
														train_size=0.75, random_state=11)
	model.fit(x_train, y_train)
	save_pickle(company_name, model_name, model, "models")
	model_results(company_name, x_train, y_train, "train", model_name)
	model_results(company_name, x_test, y_test, "test", model_name)


def model_results(company, x, y, mode, model_name):
	prediction = load_pickle(company, model_name, "models").predict(x)
	prediction = pd.DataFrame(prediction, index=y.index, columns=["Prediction"]).sort_index()
	y = y.sort_index().rename("Actual Close")
	fig, ax = plt.subplots()
	prediction.plot(kind="line", ax=ax)
	y.plot(kind="line", ax=ax)
	ax.set(xlabel='Time', ylabel='Value', title=f'{model_name} Regression Pred vs Actual {company}')
	plt.legend(loc="upper right")
	create_folder(f"charts/{mode}")
	plt.savefig(f"charts/{mode}/{model_name}_pred_{company}.png")
	# plt.show()
	print(
		f"{company}: MAE = {mean_absolute_error(y, prediction)}, MAP = {mean_absolute_percentage_error(y, prediction)}")
