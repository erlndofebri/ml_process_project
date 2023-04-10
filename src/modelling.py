import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report


import yaml
import joblib
import json
import copy
import hashlib

# utils
import util as utils

# Load dataset
def load_train_feng(params: dict):
    # Load train set
    x_train = utils.pickle_load(config["train_feng_set_path"][0])
    y_train = utils.pickle_load(config["train_feng_set_path"][1])

    return x_train, y_train

def load_valid(params: dict):
    # Load valid set
    x_valid = utils.pickle_load(config["valid_feng_set_path"][0])
    y_valid = utils.pickle_load(config["valid_feng_set_path"][1])

    return x_valid, y_valid

def load_test(params: dict):
    # Load tets set
    x_test = utils.pickle_load(config["test_feng_set_path"][0])
    y_test = utils.pickle_load(config["test_feng_set_path"][1])
    
    return x_test, y_test

# Training
def train_model(x_train, y_train, x_valid, y_valid):
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    y_pred = rfc.predict(x_valid)
    print(classification_report(y_valid, y_pred))

    return rfc

if __name__ == "__main__":
    print('Start Modelling Phase')

    # 1. Load configuration file
    config = utils.load_config()

    # 2. load dataset
    x_train, y_train = load_train_feng(config)
    x_valid, y_valid = load_valid(config)
    x_test, y_test = load_test(config)

    # 3. train model
    rfc = train_model(x_train, y_train, x_valid, y_valid)

    # 4. Dump model
    utils.pickle_dump(rfc, config["production_model_path"])

    print("End Process")