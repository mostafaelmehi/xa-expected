import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold

from football.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train the models. Only training data is ever passed in here."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def train_linear(self, X_train: pd.DataFrame, y_train) -> LinearRegression:
        """Fit a plain linear regression."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info("Linear regression trained on %d rows", len(X_train))
        return model

    def train_random_forest(self, X_train: pd.DataFrame, y_train) -> GridSearchCV:
        """Grid-search a random forest with K-fold CV on the training data only.

        The returned GridSearchCV is already refitted on the full training set
        with the best parameters, so it can be used directly for predict().
        """
        cv = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        grid = GridSearchCV(
            RandomForestRegressor(random_state=self.config.random_state),
            self.config.rf_param_grid,
            cv=cv,
            scoring="r2",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        logger.info("RF best params: %s (CV R2 = %.4f)", grid.best_params_, grid.best_score_)
        return grid