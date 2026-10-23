"""FastAPI service: serves the @champion model from the MLflow registry.

Run from the project root:
    uv run uvicorn football.api:app --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, create_model

from football.config import load_config

logger = logging.getLogger("football.api")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "xa_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# The columns the model was trained on. Checked against the loaded model at startup.
FEATURES = load_config().model.selected_features


def build_player_model(features: list[str]) -> type[BaseModel]:
    """One Pydantic field per feature.

    Column names contain spaces and symbols ('xG+/-', 'Min%'), so each field gets a
    safe Python name (f0, f1, ...) and the real column name as its alias.
    """
    fields = {f"f{i}": (float, Field(alias=name)) for i, name in enumerate(features)}
    # extra="forbid": a column that is not in the list is an error, not ignored.
    return create_model("Player", __config__=ConfigDict(extra="forbid"), **fields)


Player = build_player_model(FEATURES)


class PredictRequest(BaseModel):
    instances: list[Player] = Field(min_length=1)


class PredictResponse(BaseModel):
    predictions: list[float]
    model_version: str


def load_champion():
    """Return (model, version) for the registry alias. Kept separate so tests can replace it."""
    mlflow.set_tracking_uri(TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    return model, version


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not on every request."""
    model, version = load_champion()

    # Refuse to start if the API schema and the loaded model disagree.
    model_columns = model.metadata.get_input_schema().input_names()
    if set(model_columns) != set(FEATURES):
        raise RuntimeError(
            f"Config features do not match model {MODEL_NAME} v{version}: "
            f"only in config={sorted(set(FEATURES) - set(model_columns))}, "
            f"only in model={sorted(set(model_columns) - set(FEATURES))}"
        )

    app.state.model = model
    app.state.version = version
    app.state.columns = model_columns
    logger.info(
        "Loaded %s version %s (%d features)", MODEL_NAME, version, len(model_columns)
    )
    yield


app = FastAPI(title="xA prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info(request: Request):
    state = request.app.state
    return {
        "name": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": state.version,
        "n_features": len(state.columns),
        "features": state.columns,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    state = request.app.state

    # by_alias=True puts the real column names back as the keys.
    rows = [player.model_dump(by_alias=True) for player in body.instances]
    df = pd.DataFrame(rows)[state.columns].astype("float64")

    predictions = state.model.predict(df)
    return PredictResponse(
        predictions=[float(p) for p in predictions],
        model_version=str(state.version),
    )
