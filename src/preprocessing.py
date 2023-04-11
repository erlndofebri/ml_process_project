import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# data dumb store
import joblib
# to locate yaml file
import yaml
# to locate directore
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# utils
import util as utils

# load dataset
def load_dataset(config_data: dict):
    x_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])

    x_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])

    x_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    train_set = pd.concat([x_train, y_train], axis = 1)
    valid_set = pd.concat([x_valid, y_valid], axis = 1)
    test_set = pd.concat([x_test, y_test], axis = 1)

    return train_set, valid_set, test_set 

# 3.1 convert negative to null
def negative_to_null(data, col1, col2):
    data[col1] = np.where(data[col1] < 0, np.nan, data[col1])
    data[col2] = np.where(data[col2] < 0, np.nan, data[col2])
    return data

# 3.2 Function for checking null values
def check_null(data):
    # check null 
    check_result = data.isnull().any().sum() 
    return check_result

# 3.3 impute values
def imputer(data, col1, col2, value_imputer1, value_imputer2):
    # impute with value above
    data[col1] = data[col1].fillna(value_imputer1)
    data[col2] = data[col2].fillna(value_imputer2)
    
    return check_null(data), data

# 3. Handling missing values
def handling_missing_values(data,
                            col1,
                            col2,
                            value_imputer1,
                            value_imputer2):
    
    # 3.1 convert negative to null
    data = negative_to_null(data, col1, col2)
    
    # 3.3 check null & impute value
    check_result, data = imputer(data, col1, col2, 
                                 value_imputer1, value_imputer2)
    
    return check_result, data

# drop features with high cardinality function
def drop_feature(data, feature):
    new_data = data.drop(feature, axis = 1)
    return new_data

# one hot encoding
def encoding_cat_feature(data, fit=False, encoder=None):
    # get copy and reset_index
    data_copy = data.copy().reset_index(drop=True)
    target_col = data_copy['deposit']
    data_copy = data_copy.drop('deposit', axis=1)
    
    # category features
    cat_features = data_copy.select_dtypes(include='object').columns
    
    if fit:
        # Ohe initialization
        ohe = OneHotEncoder(handle_unknown='ignore', drop='first')
        
        # fit transform
        ohe.fit(data_copy[cat_features])
        encoder = ohe
        encoded_df = pd.DataFrame(ohe.transform(data_copy[cat_features]).toarray())
    else:
        # use existing encoder object to transform
        encoded_df = pd.DataFrame(encoder.transform(data_copy[cat_features]).toarray())

    # rename columns
    encoded_df.columns = encoder.get_feature_names_out(cat_features)
    
    # drop original cat feature
    dropped_data = data_copy.drop(cat_features, axis=1)
    
    #merge one-hot encoded columns back with original DataFrame
    final_df = dropped_data.join([encoded_df, target_col])
    
    return encoder, final_df

# Random Over Sampling
def ros_resample(data, config):
    x_ros, y_ros = RandomOverSampler(random_state = 33).fit_resample(data.drop(columns = config['label']),
                                                                    data[config['label']])
    data_resample = pd.concat([x_ros, y_ros], axis = 1)
    
    return data_resample

# Label Encoding
def label_encoder_fit(config):
    le_deposit = LabelEncoder()
    le_deposit.fit(config['label_deposit_new'])
    return le_deposit

if __name__ == "__main__":
    print('Start Preprocessing Phase')
    # 1. Load configuration file
    config = utils.load_config()

    # 2. load dataset
    train_set, valid_set, test_set = load_dataset(config)


    # 3.1 Handling missing values training set
    check_result_train, train_set = handling_missing_values(data = train_set,
                                                    col1 = config['predictors'][10], #pdays
                                                    col2 = config['predictors'][4], # balance
                                                    value_imputer1 = config['balance_imputation'],
                                                    value_imputer2 = config['pdays_imputation'])
    
    # 3.2 Handling missing values validation set
    check_result_valid, valid_set = handling_missing_values(data = valid_set,
                                                    col1 = config['predictors'][10],
                                                    col2 = config['predictors'][4],
                                                    value_imputer1 = config['balance_imputation'],
                                                    value_imputer2 = config['pdays_imputation'])
    
    # 3.3 Handling missing values test set
    check_result_test, test_set = handling_missing_values(data = test_set,
                                                    col1 = config['predictors'][10],
                                                    col2 = config['predictors'][4],
                                                    value_imputer1 = config['balance_imputation'],
                                                    value_imputer2 = config['pdays_imputation'])
    
    # 4. drop features with high cardinality (day, month, and job)
    train_set = drop_feature(train_set, config['drop_list'])
    valid_set = drop_feature(valid_set, config['drop_list'])
    test_set = drop_feature(test_set, config['drop_list'])

    # 5.1 Encoding training set
    encoder, train_set_fin = encoding_cat_feature(data = train_set,
                                                  fit = True) 
    # 5.2 Encoding valid set
    _, valid_set_fin = encoding_cat_feature(data = valid_set,
                                        encoder = encoder)
    
    # 5.3 Encoding test set
    _, test_set_fin = encoding_cat_feature(data = test_set,
                                       encoder = encoder)
    
    # 6. resampling ros training set
    train_set_ros = ros_resample(data = train_set_fin,
                        config = config)
    
    # 7. label encoding
    le_deposit = label_encoder_fit(config)

    # 7.1 dataset transform target label encoding
    train_set_ros[config['label']] = le_deposit.transform(train_set_ros[config['label']])

    # 7.2 data valid transform target label encoding
    valid_set_fin[config['label']] = le_deposit.transform(valid_set_fin[config['label']])

    # 7.3 data test transform target label encoding
    test_set_fin[config['label']] = le_deposit.transform(test_set_fin[config['label']])

    # 8. dumb data

    # x_train set ros
    utils.pickle_dump(train_set_ros[config["predictors_ohe"]],
                    config["train_feng_set_path"][0])
    # y_train set ros
    utils.pickle_dump(train_set_ros[config["label"]],
                    config["train_feng_set_path"][1])

    # x_valid set
    utils.pickle_dump(valid_set_fin[config["predictors_ohe"]],
                    config["valid_feng_set_path"][0])
    # y_valid set
    utils.pickle_dump(valid_set_fin[config["label"]],
                    config["valid_feng_set_path"][1])
    # x_test set
    utils.pickle_dump(test_set_fin[config["predictors_ohe"]],
                    config["test_feng_set_path"][0])
    # y_test set
    utils.pickle_dump(test_set_fin[config["label"]],
                    config["test_feng_set_path"][1])
    
    print('Process End')