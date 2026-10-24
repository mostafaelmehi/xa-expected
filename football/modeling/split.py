import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from football.config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class DataSplitter:
    """Separate the target from the features and split into train and test sets.

    Everything that learns from the data (scaler, feature selection, models)
    must be fitted on the train part only, so the split happens first.
    """

    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        self.data_config = data_config
        self.model_config = model_config

    def split(self, df: pd.DataFrame):
        """Return X_train, X_test, y_train, y_test."""
        target = self.data_config.target

        # The target must never stay inside the features.
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.model_config.test_size,
            random_state=self.model_config.random_state,
        )
        logger.info("Train: %d rows, Test: %d rows", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test
