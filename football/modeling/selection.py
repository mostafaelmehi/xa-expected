import logging

import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Hybrid feature selection: mutual-information ranking, then forward SFS.

    Stateful: the selected features are learned in fit(), so fit() must only
    ever see the training data. transform() then applies the same list to any data.
    """

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        # Step 1 (filter): rank features by mutual information with the target.
        mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        ranked_columns = X.columns[np.argsort(mi_scores)[::-1]]
        X_ranked = X[ranked_columns]

        # Step 2 (wrapper): forward selection with a linear model and K-fold CV.
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        sfs = SFS(
            LinearRegression(),
            k_features="best",
            forward=True,
            floating=False,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
        )
        sfs.fit(X_ranked, y)

        self.selected_features_ = list(sfs.k_feature_names_)
        logger.info("Selected %d of %d features", len(self.selected_features_), X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Same columns, same order, for both train and test.
        return X[self.selected_features_]