from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from data_preprocessing import inference_vector_creation
from data_serializer import load_pickle


def linear_results(company):
	data_linear = inference_vector_creation(company)
	data_linear['Prediction'] = load_pickle(company, "linear", "models").predict(data_linear.values.reshape(-1, 2)).reshape(-1, 1)
	data_linear['Prediction'] = data_linear['Prediction'].shift(1)
	data_linear = data_linear.dropna()
	fig, ax = plt.subplots()
	data_linear.plot(kind="line", y="Prediction", ax=ax)
	data_linear.plot(kind="line", y="Close", ax=ax)
	ax.set(xlabel='Time', ylabel='Value', title=f'Linear Regression Pred vs Actual {company}')
	plt.legend(loc="upper right")
	plt.savefig(f"charts/training_charts/linear_pred_{company}.png")
	plt.show()
	print(f"{company}: MAE = {mean_absolute_error(data_linear['Close'], data_linear['Prediction'])}, MAP = {mean_absolute_percentage_error(data_linear['Close'], data_linear['Prediction'])}")
