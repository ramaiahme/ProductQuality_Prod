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

EXPERIMENT_NAME = "Product_Quality"
mlflow.set_experiment(EXPERIMENT_NAME)

def hyperparameter_tuning(X_train,y_train,param_grid):
    rf=RandomForestRegressor()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train,y_train)
    return grid_search

## load the paramerters from params.yaml

params=yaml.safe_load(open("params.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["QualityRating"])
    y=data['QualityRating']

    mlflow.set_tracking_uri("https://dagshub.com/ramaiahme/ProductQuality.mlflow")
        
    #start the MLFLOW run
    with mlflow.start_run():    
        #split the dataset into tracking and test sets
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
        signature = infer_signature(X_train,y_train)

        param_grid = {
            'max_depth': [5,10,None,1],
            "n_estimators": [100, 150, 200,250]
        }
    
        grid_search = hyperparameter_tuning(X_train,y_train,param_grid)

                  # Log all runs from grid search
        for i, (params, score) in enumerate(
                zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'])
            ):
            with mlflow.start_run(run_name=f"RandomForest_run_{i}", nested=True) as run:
            
                mlflow.log_params(params)

        #get the best model
        best_model=grid_search.best_estimator_

        best_model.set_params(**params) #set the parameters of the current model to be loged in this run
        best_model.fit(X_train,y_train)
    
        #predict and evaluate the model

        y_pred=best_model.predict(X_test)

        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
         }
        

        ##log additional matrics
        mlflow.log_metrics(metrics)
        mlflow.log_param("best_n_estimators",grid_search.best_estimator_.n_estimators)
        mlflow.log_param("best_max_depth",grid_search.best_estimator_.max_depth)
    

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        input_example = np.array(X_train.iloc[[0]])
    
        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model",input_example=input_example)
        else:
            mlflow.sklearn.log_model(best_model,"model",signature=signature,input_example=input_example)

        #create the directory to save the model
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
    
        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))
    
        print(f"Model saved to {model_path}")
    
if __name__=="__main__":
    train(params['data'],params['model'],params['random_state'],params['n_estimators'],params['max_depth'])