import pandas as pd
from matplotlib import pyplot as plt

from data_serializer import load_pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TimeseriesValues(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		return np.array(self.data[index])


def lstm_results(company, data):
	"""
	Feeds data to lstm model and returns the prediction
	:param company: name of company
	:param data: company data
	:return: predictions
	"""
	feature_scaler = load_pickle(company, "feature_scaler", "scalers")
	label_scaler = load_pickle(company, "label_scaler", "scalers")
	model = load_pickle(company, "lstm", "models")
	true_next_close = pd.DataFrame(data['Next_Close'])
	data = data.drop('Next_Close', axis=1)

	index = data.index
	data['Close'] = feature_scaler.transform(data['Close'].values.reshape(-1,1))
	data = data.values
	loader = DataLoader(TimeseriesValues(data), index.shape[0], shuffle=False, drop_last=True)
	for i in loader:
		i = i.reshape(-1, 1, 2)
		model.eval()
		predictions = model(i.float())

	predictions = predictions.detach().numpy().reshape(1, -1)
	true_next_close['Prediction'] = label_scaler.inverse_transform(predictions).reshape(-1, 1)
	fig, ax = plt.subplots()
	true_next_close.plot(kind="line", y="Prediction", ax=ax)
	true_next_close.plot(kind="line", y="Next_Close", ax=ax)
	ax.set(xlabel='Time', ylabel='Value', title=f'LSTM Pred vs Actual {company}')
	plt.legend(loc="upper right")
	plt.savefig(f"charts/training_charts/lstm_pred_{company}.png")
	# plt.show()

	print(f"{company} predictions= {predictions}")