import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import yaml
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
import os
import mlflow

from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/ramaiahme/ProductQuality.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="ramaiahme"
os.environ['MLFLOW_TRACKING_PASSWORD']="75390f862c21ee48ed31835444767713104badb8"

## load the paramerters from params.yaml

params=yaml.safe_load(open("params.yaml"))["train"]

EXPERIMENT_NAME = "Product_Quality"
mlflow.set_experiment(EXPERIMENT_NAME)

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["QualityRating"])
    y=data['QualityRating']

    mlflow.set_tracking_uri("https://dagshub.com/ramaiahme/ProductQuality.mlflow")

    #load the model 
    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    rmse = root_mean_squared_error(y,predictions)
    mae = mean_absolute_error(y,predictions)
    r2 = r2_score(y,predictions)
    
    metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
         }
        
    ##log matrics
    mlflow.log_metrics(metrics)

if __name__=="__main__":
    evaluate(params["data"],params['model'])