import pickle
from utils import create_folder

def save_model(company_name, model_type, model):
    create_folder("models")
    with open(f"models/{company_name}_{model_type}.pickle", 'wb') as file:
        pickle.dump(model.best_estimator_, file)  # Get best hyperparameters and save the model

def load_model(company_name,model_type):
    create_folder("models")
    return pickle.load(open(f"models/{company_name}_{model_type}.pickle", 'rb'))
