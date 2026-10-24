import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class PositionFeatures(BaseEstimator, TransformerMixin):
    """Turn the 'Pos' text column into numeric features.

    Adds:
      - haybird: 1 if the player has more than one listed position
      - Main_Pos_<X>: one-hot columns for the first listed position
    and drops the original 'Pos' column.

    Stateless: nothing is learned from the data, so it is safe to run before the split.
    """

    def __init__(self, main_positions=("DF", "MF", "FW")):
        self.main_positions = main_positions

    def fit(self, X, y=None):
        # Nothing to learn; required by the sklearn transformer API.
        return self

    def transform(self, X):
        df = X.copy()

        # A comma in 'Pos' means several positions, e.g. "MF,FW".
        df["haybird"] = df["Pos"].str.contains(",").astype(int)

        # The first listed position is treated as the main one.
        main = df["Pos"].str.split(",").str[0]

        # Fixed categories keep the same columns even if a position is missing in a subset.
        main = pd.Categorical(main, categories=list(self.main_positions))
        dummies = pd.get_dummies(main, prefix="Main_Pos").astype(int)
        dummies.index = df.index

        df = pd.concat([df.drop(columns=["Pos"]), dummies], axis=1)
        logger.info("Added position features, shape now %s", df.shape)
        return df
