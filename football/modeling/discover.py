"""One-off discovery: which features to keep and which RF params to use.

Run once and copy the printed block into football/config.py.
Re-run only when the data or the cleaning rules change.
"""

import logging

from football.config import load_config
from football.modeling.evaluate import ModelEvaluator
from football.modeling.selection import FeatureSelector
from football.modeling.train import ModelTrainer
from football.pipeline import run_pipeline


def main():
    config = load_config()
    X_train, X_test, y_train, y_test = run_pipeline()

    # 1. Feature selection on the train set only
    selector = FeatureSelector(
        cv_folds=config.model.cv_folds, random_state=config.model.random_state
    )
    features = selector.fit(X_train, y_train).selected_features_

    # 2. Grid search on the selected features, train set only
    grid = ModelTrainer(config.model).train_random_forest(X_train[features], y_train)

    # 3. Test score, for information only
    metrics = ModelEvaluator().evaluate(grid, X_test[features], y_test)

    print("\n" + "=" * 60)
    print(f"{len(features)} features | CV R2 = {grid.best_score_:.4f} | test = {metrics}")
    print("=" * 60)
    print("Copy this into ModelConfig in football/config.py:\n")
    print(f"    selected_features: list[str] = field(default_factory=lambda: {features!r})")
    print(f"    rf_best_params: dict = field(default_factory=lambda: {grid.best_params_!r})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
