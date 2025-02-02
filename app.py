from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Any
from src import prediction
import os
import yaml
import logging
import mlflow

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# MLflow Tracking URI
os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/ramaiahme/ProductQuality.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="ramaiahme"
os.environ['MLFLOW_TRACKING_PASSWORD']="75390f862c21ee48ed31835444767713104badb8"

# MLflow Experiment name (should be same as training)
EXPERIMENT_NAME = "Quality_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info(f"MLflow experiment '{EXPERIMENT_NAME}' set.")

# Load Parameters
params = yaml.safe_load(open("params.yaml"))["predict"]

app = FastAPI()

templates = Jinja2Templates(directory=".")

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
  """Render the HTML form."""
  return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_quality(
    request: Request,
    temperature: float = Form(...),
    pressure: float = Form(...),
    temperature_pressure: float = Form(...),
    material_fusion_metric: float = Form(...),
    material_transformation_metric: float = Form(...),
):
    """Takes user input, makes prediction, renders the results as HTML."""
    try:
        input_data = [
            temperature,
            pressure,
            temperature_pressure,
            material_fusion_metric,
            material_transformation_metric,
         ]

        # Make prediction
        #predicted_quality = prediction(params["model"], input_data)
        predicted_quality = prediction.predict_product_quality(params["model"], input_data)


        if predicted_quality is not None:
            return templates.TemplateResponse("index.html", {
                                                            "request": request,
                                                            "prediction": f"Predicted Quality: {predicted_quality}"
                                                        })
        else:
             return templates.TemplateResponse("index.html", {
                                                            "request": request,
                                                            "error": "Error: could not get a prediction from the model."
                                                        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return templates.TemplateResponse("index.html", {
                                                "request": request,
                                                "error": str(e)
                                            })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")