import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute regression metrics for any fitted model.

    The target is never scaled, so MAE and RMSE are in real xA units.
    """

    def evaluate(self, model, X, y) -> dict:
        """Return R2, MAE and RMSE of the model on the given data."""
        y_pred = model.predict(X)
        metrics = {
            "r2": r2_score(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        }
        logger.info("Metrics: %s", metrics)
        return metrics
