import os
import sys
import certifi
from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from urllib.parse import urlparse

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

# Project imports
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import (
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DATABASE_NAME,
)

# -------------------------
# Config / constants
# -------------------------
load_dotenv()  # harmless in container; main config from env

# S3 URIs provided by you (can also be supplied via env if you prefer)
S3_MODEL_URI = os.getenv(
    "S3_MODEL_URI",
    "s3://networktrial/artifact/09_01_2025_18_51_40/model_trainer/trained_model/model.pkl",
)
S3_PREPROC_URI = os.getenv(
    "S3_PREPROC_URI",
    "s3://networktrial/artifact/09_01_2025_18_51_40/data_transformation/transformed_object/preprocessing.pkl",
)

LOCAL_MODEL_DIR = "final_model"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.pkl")
LOCAL_PREPROC_PATH = os.path.join(LOCAL_MODEL_DIR, "preprocessor.pkl")

# Optional: Mongo (won’t block startup if it fails)
MONGO_DB_URL = os.getenv("MONGODB_URL_KEY")
ca = certifi.where()

# -------------------------
# Helpers
# -------------------------
def parse_s3_uri(s3_uri: str):
    """Return (bucket, key) from s3://bucket/key uri."""
    u = urlparse(s3_uri)
    if u.scheme != "s3" or not u.netloc or not u.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = u.netloc
    key = u.path.lstrip("/")
    return bucket, key


def download_from_s3(s3_uri: str, local_path: str):
    """Download a single object from S3 to local_path (idempotent)."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    bucket, key = parse_s3_uri(s3_uri)

    # Skip if already present
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        logging.info(f"[S3] Using cached file: {local_path}")
        return

    logging.info(f"[S3] Downloading {s3_uri} -> {local_path}")
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, key, local_path)
        logging.info(f"[S3] Download complete: {local_path}")
    except (BotoCoreError, ClientError) as e:
        raise NetworkSecurityException(
            f"Failed to download {s3_uri}: {e}", sys
        ) from e


def ensure_artifacts_available():
    """Ensure preprocessor + model are available locally (download from S3 if missing)."""
    download_from_s3(S3_PREPROC_URI, LOCAL_PREPROC_PATH)
    download_from_s3(S3_MODEL_URI, LOCAL_MODEL_PATH)


try:
    ensure_artifacts_available()
except Exception as e:
    # We still start the app so /health shows a clear signal, but /predict will fail until fixed
    logging.exception("Failed to prepare artifacts from S3.")

# Optional Mongo wiring (won’t crash the app if fails)
try:
    if MONGO_DB_URL:
        import pymongo

        client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
        database = client[DATA_INGESTION_DATABASE_NAME]
        if DATA_INGESTION_COLLECTION_NAME not in database.list_collection_names():
            logging.warning(
                f"Collection '{DATA_INGESTION_COLLECTION_NAME}' not found. Creating."
            )
            database.create_collection(DATA_INGESTION_COLLECTION_NAME)
    else:
        logging.warning("MONGODB_URL_KEY is not set; /train may fail.")
except Exception:
    logging.exception("Mongo init failed; continuing without DB.")

app = FastAPI()#Creates the ASGI app object
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
templates = Jinja2Templates(directory="./app/templates")


# -------------------------
# Routes
# -------------------------
@app.get("/", tags=["misc"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["misc"])
async def health():
    return JSONResponse(
        {
            "status": "ok",
            "model_present": os.path.exists(LOCAL_MODEL_PATH),
            "preprocessor_present": os.path.exists(LOCAL_PREPROC_PATH),
            "mlflow_uri_set": bool(os.getenv("MLFLOW_TRACKING_URI")),
            "mongo_set": bool(MONGO_DB_URL),
        }
    )


@app.get("/train", tags=["pipeline"])
async def train_route():
    """Left intact; may fail if Mongo isn’t reachable. Doesn’t affect prediction artifacts."""
    try:
        tp = TrainingPipeline()
        tp.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


@app.post("/predict", tags=["inference"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        # Defensive: ensure artifacts exist (in case container restarted with empty volume)
        ensure_artifacts_available()

        df = pd.read_csv(file.file)

        preprocessor = load_object(LOCAL_PREPROC_PATH)
        final_model = load_object(LOCAL_MODEL_PATH)
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

        y_pred = network_model.predict(df)
        df["predicted_column"] = y_pred

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped", index=False)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    import uvicorn
    # IMPORTANT: 0.0.0.0:8080 (matches Docker + GitHub Actions health check)
    uvicorn.run(app, host="0.0.0.0", port=8080)
