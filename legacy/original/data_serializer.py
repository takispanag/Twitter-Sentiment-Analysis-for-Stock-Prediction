import pickle
from utils import create_folder


def save_pickle(company_name, model_type, model, path):
    create_folder(path)
    with open(f"{path}/{company_name}_{model_type}.pickle", 'wb') as file:
        pickle.dump(model, file)


def load_pickle(company_name, model_type, path):
    create_folder(path)
    return pickle.load(open(f"{path}/{company_name}_{model_type}.pickle", 'rb'))
