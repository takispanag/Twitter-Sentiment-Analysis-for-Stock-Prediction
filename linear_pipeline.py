import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from data_serializer import load_pickle, save_pickle
from utils import create_folder
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

def model_training(x_train, y_train, company_name, model_name, model):
	model.fit(x_train, y_train)
	save_pickle(company_name, model_name, model, "models")


def model_results(company, x, y, mode, model_name):
	results = pd.DataFrame()
	prediction = load_pickle(company, model_name, "models").predict(x)
	results["Actual Close"] = y.shift(1).dropna()
	results["Prediction"] = pd.Series(prediction, index=y.index)

	metrics = pd.DataFrame()


	metrics["mae"] = [mean_absolute_error(results["Actual Close"], results["Prediction"])]
	metrics["mape"] = [mean_absolute_percentage_error(results["Actual Close"], results["Prediction"])]
	metrics["mse"] = [mean_squared_error(results["Actual Close"], results["Prediction"])]
	metrics["r2"] = [r2_score(results["Actual Close"], results["Prediction"])]

	create_folder(f"data/metrics/{mode}")
	metrics.to_csv(f"data/metrics/{mode}/{model_name}_metrics_{company}.csv")

	layout = go.Layout(
		autosize=False,
		width=2560,
		height=1440
	)
	fig = go.Figure(layout=layout)
	fig.update_layout(template=pio.templates['plotly_dark'])
	for column,color in zip(results.columns,["#32a88b","#a83232"]):
		fig.add_trace(go.Scatter(x=results.index, y=results[column], mode='lines+markers', line=dict(color=color), name=column))
	create_folder(f"charts/{mode}")
	fig.write_html(f"charts/{mode}/{model_name}_pred_{company}.html")
	fig.write_image(f"charts/{mode}/{model_name}_pred_{company}.png")
