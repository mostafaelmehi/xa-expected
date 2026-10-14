import numpy as np
import pandas as pd

from football.modeling.selection import FeatureSelector


def make_data(n: int = 120):
    """Target depends only on 'a' and 'b'; the other columns are pure noise."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(n, 5)), columns=["a", "b", "noise1", "noise2", "noise3"]
    )
    y = 3 * X["a"] + 2 * X["b"] + rng.normal(scale=0.1, size=n)
    return X, y


def test_selector_keeps_the_informative_features():
    X, y = make_data()
    selector = FeatureSelector(cv_folds=3, random_state=42).fit(X, y)

    assert "a" in selector.selected_features_
    assert "b" in selector.selected_features_


def test_transform_returns_only_selected_columns_in_the_same_order():
    X, y = make_data()
    selector = FeatureSelector(cv_folds=3, random_state=42).fit(X, y)

    out = selector.transform(X)

    assert list(out.columns) == selector.selected_features_
    assert len(out) == len(X)


def test_transform_works_on_new_data_without_refitting():
    X, y = make_data()
    selector = FeatureSelector(cv_folds=3, random_state=42).fit(X.iloc[:100], y.iloc[:100])
    selected_before = list(selector.selected_features_)

    # The "test" rows are only transformed, never used to choose features.
    out = selector.transform(X.iloc[100:])

    assert list(out.columns) == selected_before
    assert selector.selected_features_ == selected_before