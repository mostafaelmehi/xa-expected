"""Train the final xA model with the frozen config, log it to MLflow, register it."""

import logging
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor

from football.config import load_config
from football.modeling.evaluate import ModelEvaluator
from football.pipeline import run_pipeline

logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "xa_prediction"
MODEL_NAME = "xa_model"


def train_final():
    config = load_config()
    X_train, X_test, y_train, y_test = run_pipeline()

    # Only the frozen feature list goes into the model (and later into the API).
    features = config.model.selected_features

    model = RandomForestRegressor(
        random_state=config.model.random_state,
        n_jobs=-1,
        **config.model.rf_best_params,
    )
    model.fit(X_train[features], y_train)

    metrics = ModelEvaluator().evaluate(model, X_test[features], y_test)
    return config, model, metrics, features, X_train


def log_to_mlflow(config, model, metrics, features, X_train) -> str:
    """Log one run, register the model, tag the new version as @challenger."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_sample = X_train[features].head(5)

    with mlflow.start_run():
        mlflow.log_params(config.model.rf_best_params)
        mlflow.log_param("n_features", len(features))
        mlflow.log_param("random_state", config.model.random_state)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_dict({"features": features}, "features.json")

        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_sample, model.predict(X_sample)),
            input_example=X_sample,
            registered_model_name=MODEL_NAME,
            # We trained this model ourselves, so its tree nodes are safe to trust.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "challenger", version)
    return version


def promote_if_better(version: str, metrics: dict) -> bool:
    """Make this version @champion if it is at least as good as the current champion."""
    client = MlflowClient()

    try:
        current = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        # No champion yet: the first registered model takes the role.
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        logger.info("No champion yet: version %s is now @champion", version)
        return True

    current_r2 = client.get_run(current.run_id).data.metrics["test_r2"]
    if metrics["r2"] >= current_r2:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        logger.info(
            "Version %s (R2 %.4f) replaces champion (R2 %.4f)",
            version,
            metrics["r2"],
            current_r2,
        )
        return True

    logger.info(
        "Version %s (R2 %.4f) is worse than champion (R2 %.4f): not promoted",
        version,
        metrics["r2"],
        current_r2,
    )
    return False


def main():
    config, model, metrics, features, X_train = train_final()
    print(f"\nTrained on {len(features)} features | test = {metrics}")

    version = log_to_mlflow(config, model, metrics, features, X_train)
    print(f"Registered {MODEL_NAME} version {version} with alias @challenger")

    promoted = promote_if_better(version, metrics)
    print(f"Promoted to @champion: {promoted}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    main()
