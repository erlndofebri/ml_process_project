import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
# data dumb store
import joblib
# to locate yaml file
import yaml
# to locate directore
import os

import util as utils
import copy

# 1. Load configuration file
config = utils.load_config()

# 2. fungsi read data csv
def read_data(config_data):
    # Return raw dataset
    return pd.read_csv(config["dataset_path"])

# 3. handling pdays and balance columns function
def handling_negative(data, col1, col2):
    data[col1] = np.where(data[col1] < 0, -1, data[col1])
    data[col2] = np.where(data[col2] < 0, -1, data[col2])
    return data

# 4. data defence
def check_data(input_data, params):
    # check data types
    assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."
    assert input_data.select_dtypes("int").columns.to_list() == params["int32_columns"], "an error occurs in int32 column(s)."

    # check range of data
    assert set(input_data.job).issubset(set(params["range_job"])), "an error occurs in job range."
    assert set(input_data.marital).issubset(set(params["range_marital"])), "an error occurs in marital range."
    assert set(input_data.education).issubset(set(params["range_education"])), "an error occurs in education range."
    assert set(input_data.default).issubset(set(params["range_default"])), "an error occurs in default range."
    assert set(input_data.housing).issubset(set(params["range_housing"])), "an error occurs in housing range."
    assert set(input_data.loan).issubset(set(params["range_loan"])), "an error occurs in loan range."
    assert set(input_data.contact).issubset(set(params["range_contact"])), "an error occurs in contact range."
    assert set(input_data.month).issubset(set(params["range_month"])), "an error occurs in month range."
    assert set(input_data.poutcome).issubset(set(params["range_poutcome"])), "an error occurs in poutcome range."
    assert input_data.age.between(params["range_age"][0], params["range_age"][1]).sum() == len(input_data), "an error occurs in age range."
    assert input_data.balance.between(params["range_balance"][0], params["range_balance"][1]).sum() == len(input_data), "an error occurs in balance range."
    assert input_data.day.between(params["range_day"][0], params["range_day"][1]).sum() == len(input_data), "an error occurs in day range."
    assert input_data.duration.between(params["range_duration"][0], params["range_duration"][1]).sum() == len(input_data), "an error occurs in duration range."
    assert input_data.campaign.between(params["range_campaign"][0], params["range_campaign"][1]).sum() == len(input_data), "an error occurs in campaign range."
    assert input_data.pdays.between(params["range_pdays"][0], params["range_pdays"][1]).sum() == len(input_data), "an error occurs in pdays range."
    assert input_data.previous.between(params["range_previous"][0], params["range_previous"][1]).sum() == len(input_data), "an error occurs in previous range."


# 5. split data
def split_data(input_data: pd.DataFrame, config: dict):
    # Split predictor and label
    x = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()

    # 1st split train and test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y
    )

    # 2nd split test and valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    print("Start initiate Data Pipeline Process")
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    data = read_data(config)

    # 3. handling pdays and balance columns function
    data = handling_negative(data, 
                             config['predictors'][10], #pdays column
                             config['predictors'][4]) #balance column

    # 4. data defense
    check_data(data, config)

    # 5. Splitting train, valid, and test set
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(data, config)

    # 6. Save train, valid and test set
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])

    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])

    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

    utils.pickle_dump(data, config["dataset_cleaned_path"])

    print("End of Processss")