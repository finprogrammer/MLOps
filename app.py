import os
import sys
import certifi
from dotenv import load_dotenv
import pymongo
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# Load .env if present (harmless in container; main source will be Docker -e)
load_dotenv()

# --- MongoDB client (uses secret MONGODB_URL_KEY) ---
ca = certifi.where()
mongo_db_url = os.getenv("MONGODB_URL_KEY")

client = None
database = None
collection = None
if mongo_db_url:
    try:
        client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
        database = client[DATA_INGESTION_DATABASE_NAME]
        collection = database[DATA_INGESTION_COLLECTION_NAME]
        if DATA_INGESTION_COLLECTION_NAME not in database.list_collection_names():
            logging.warning(
                f"Collection '{DATA_INGESTION_COLLECTION_NAME}' not found in DB '{DATA_INGESTION_DATABASE_NAME}'. Creating empty collection."
            )
            database.create_collection(DATA_INGESTION_COLLECTION_NAME)
    except Exception as e:
        logging.exception("Failed to connect to MongoDB; app will still start.")
else:
    logging.warning("MONGODB_URL_KEY is not set; DB-dependent features may fail.")

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
templates = Jinja2Templates(directory="./templates")


@app.get("/", tags=["misc"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["misc"])
async def health():
    # Simple health signal + whether essential env is present
    return JSONResponse(
        {
            "status": "ok",
            "has_mongo_url": bool(mongo_db_url),
            "mlflow_uri_set": bool(os.getenv("MLFLOW_TRACKING_URI")),
        }
    )


@app.get("/train", tags=["pipeline"])
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict", tags=["inference"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Load preprocessing + model
        preprocessor = load_object("final_model/preprocessor.pkl")
        final_model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        # Inference
        y_pred = network_model.predict(df)
        df["predicted_column"] = y_pred

        # Save prediction CSV for debugging/inspection
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        # HTML table preview
        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    # IMPORTANT: Run on 0.0.0.0:8080 to match Docker + CI/CD mapping
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
