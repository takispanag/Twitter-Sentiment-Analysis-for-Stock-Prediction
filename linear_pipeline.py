import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import config
from data_preprocessing import inference_vector_creation, training_vector_creation
from data_serializer import load_pickle, save_pickle


def linear_training(df, company_name):
	param = "linear"
	model = LinearRegression()
	x_train, x_test, y_train, y_test = train_test_split(df[["Sentiment", "Close"]], df["Next_Close"],
														train_size=0.75, random_state=11)
	model.fit(x_train, y_train)
	save_pickle(company_name, param, model, "models")
	linear_results(company_name, x_train, y_train, "train")
	linear_results(company_name, x_test, y_test, "test")

def linear_results(company, x, y, mode):
	prediction = load_pickle(company, "linear", "models").predict(x)
	prediction = pd.DataFrame(prediction, index=y.index, columns=["Prediction"]).sort_index()
	y = y.sort_index().rename("Actual Close")
	fig, ax = plt.subplots()
	prediction.plot(kind="line", ax=ax)
	y.plot(kind="line", ax=ax)
	ax.set(xlabel='Time', ylabel='Value', title=f'Linear Regression Pred vs Actual {company}')
	plt.legend(loc="upper right")
	plt.savefig(f"charts/{mode}/linear_pred_{company}.png")
	# plt.show()
	print(f"{company}: MAE = {mean_absolute_error(y, prediction)}, MAP = {mean_absolute_percentage_error(y, prediction)}")
