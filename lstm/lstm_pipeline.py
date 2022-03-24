import pandas as pd
from matplotlib import pyplot as plt

from data_serializer import load_pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from utils import create_folder


class TimeseriesValues(Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, index):
		return np.array(self.data[index])


def lstm_results(company, data, true_next_close, mode):
	"""
	Feeds data to lstm model and returns the prediction
	:param mode: chart type
	:param company: name of company
	:param data: Features
	:param true_next_close: Labels
	:return: predictions
	"""
	feature_scaler = load_pickle(company, "feature_scaler", "scalers")
	label_scaler = load_pickle(company, "label_scaler", "scalers")
	model = load_pickle(company, "lstm", "models")
	results = pd.DataFrame()
	results["Actual Close"] = true_next_close.shift(1)
	index = data.index
	data.loc[:, data.columns != 'Sentiment'] = feature_scaler.transform(data.loc[:, data.columns != 'Sentiment'])
	data = data.values
	loader = DataLoader(TimeseriesValues(data), index.shape[0], shuffle=False, drop_last=True)

	model.eval()
	for i in loader:
		i = i.reshape(1, -1, data.shape[1])
		predictions = model(i.float())

	results["Prediction"] = label_scaler.inverse_transform(predictions.detach().numpy().reshape(1, -1)).reshape(-1, 1)

	layout = go.Layout(
		autosize=False,
		width=2560,
		height=1440
	)
	fig = go.Figure(layout=layout)
	fig.update_layout(template=pio.templates['plotly_dark'])
	for column, color in zip(results.columns, ["#32a88b", "#a83232"]):
		fig.add_trace(
			go.Scatter(x=results.index, y=results[column], mode='lines+markers', line=dict(color=color), name=column))
	create_folder(f"charts/{mode}")
	fig.write_html(f"charts/{mode}/lstm_pred_{company}.html")
	fig.write_image(f"charts/{mode}/lstm_pred_{company}.png")

	# ax = results.plot()
	# ax.set(xlabel='Time', ylabel='Value', title=f'LSTM Pred vs Actual {company}')
	# plt.legend(loc="upper right")
	# create_folder(f"charts/{mode}")
	# plt.savefig(f"charts/{mode}/lstm_pred_{company}.png")
	return results
# plt.show()
