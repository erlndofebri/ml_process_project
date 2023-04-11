from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
import util as utils
import data_pipeline as data_pipeline
import preprocessing as preprocessing


config = utils.load_config()
model_data = utils.pickle_load(config["production_model_path"])
le_encoder = utils.pickle_load(config["le_encoder_path"])
ohe_encoder = utils.pickle_load(config["ohe_stasiun_path"])

class api_data(BaseModel):
    age : int
    marital : object
    education : object
    default : object
    balance : int
    housing : object
    loan : object
    contact : object
    duration : int
    campaign : int
    pdays : int
    previous : int
    poutcome : int
    day : int
    job : object
    month : object


app = FastAPI()

@app.post("/predict/")
def predict(data: api_data):
    # convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    # check range data
    try:
        data_pipeline.check_data(data, config)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    
    # drop high cardinality features (day, month, and job)
    data = preprocessing.drop_feature(data, 
                                      config['drop_list'])


    # one hot encoding
    data = preprocessing.encoding_cat_feature(data = data,
                                              encoder = encoder)



    # predict data
    y_pred = model_data.predict(data)

    # inverse transform
    y_pred = list(le_encoder.inverse_transform(y_pred))[0]

    return {"res":y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)