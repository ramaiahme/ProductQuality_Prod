import pandas as pd
import numpy as np
import pickle
import yaml
import logging
import os
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# MLflow Tracking URI

os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/ramaiahme/ProductQuality.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="ramaiahme"
os.environ['MLFLOW_TRACKING_PASSWORD']="75390f862c21ee48ed31835444767713104badb8"

## load the paramerters from params.yaml

params=yaml.safe_load(open("params.yaml"))["predict"]

EXPERIMENT_NAME = "Product_Quality"
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info(f"MLflow experiment '{EXPERIMENT_NAME}' set.")

def load_model(model_path):
    """Loads the trained model."""
    try:
        logger.info(f"Attempting to load model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded successfully from {model_path}.")
        return model
    except FileNotFoundError as fnf_error:
        logger.error(f"Error: Model file not found at path: {model_path}. Details:{fnf_error}", exc_info = True)
        raise
    except Exception as e:
        logger.error(f"Error while loading model: {e}", exc_info = True)
        raise


def make_prediction(model, input_data):
    """Makes a prediction using the loaded model."""
    try:
        logger.info(f"Input data received: {input_data}")
        # Convert input data to numpy array
        input_array = np.array([input_data])
        logger.info(f"Input array created: {input_array}")
        # make predictions
        prediction = model.predict(input_array)
        logger.info(f"Prediction made successfully. {prediction}")
        return prediction[0] #only return the first value in the list
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info = True)
        return None


def predict_product_quality(model_path, input_data):
    logger.info(f"Starting predict_product_quality with model_path: {model_path}, input_data: {input_data}")
    mlflow.set_tracking_uri("https://dagshub.com/ramaiahme/ProductQuality.mlflow")

    try:
        #load the model 
        model=load_model(model_path)
        logger.info("Model loaded successfully.")

        with mlflow.start_run(run_name="Prediction") as run:
            
            # Create input example with a numpy array (required for schema detection)
            input_example = np.array([input_data])
            mlflow.sklearn.log_model(model, "model", input_example=input_example)
            logger.info("Model logged with mlflow")

            predictions=make_prediction(model, input_data)
            logger.info(f"Raw prediction: {predictions}")

            if predictions is not None:
                logger.info(f"The predicted Quality Rating is: {predictions:.2f}")
                return predictions
            else:
                logger.error("Prediction failed.")
                return None
    except Exception as e:
       logger.error(f"Error in predict_product_quality: {e}", exc_info = True)
       return None

if __name__=="__main__":
  
   # Example data for prediction (replace with actual user input)
    input_data = [
        25.0,
        10.0,
        250.0,
        60.0,
        150.0,
    ]
    
    # Call the function
    predicted_quality = predict_product_quality(params['model'],input_data)
    print(f"The predicted Quality Rating from main block is: {predicted_quality:.2f}")