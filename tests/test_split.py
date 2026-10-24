import pandas as pd

from football.config import DataConfig, ModelConfig
from football.modeling.split import DataSplitter


def make_data(n: int = 100) -> pd.DataFrame:
    """Fake data with a target column and two features."""
    return pd.DataFrame({"xA": range(n), "KP": range(n, 2 * n), "Sh": range(2 * n, 3 * n)})


def test_split_removes_target_from_features():
    splitter = DataSplitter(DataConfig(), ModelConfig())
    X_train, X_test, y_train, y_test = splitter.split(make_data())

    assert "xA" not in X_train.columns
    assert "xA" not in X_test.columns


def test_split_sizes_follow_config():
    splitter = DataSplitter(DataConfig(), ModelConfig(test_size=0.2))
    X_train, X_test, y_train, y_test = splitter.split(make_data(100))

    assert len(X_test) == 20
    assert len(X_train) == 80
    assert len(y_train) == len(X_train)


def test_split_has_no_overlap_between_train_and_test():
    splitter = DataSplitter(DataConfig(), ModelConfig())
    X_train, X_test, _, _ = splitter.split(make_data())

    # No row may appear in both sets, otherwise the test would not be unseen.
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_is_reproducible():
    splitter = DataSplitter(DataConfig(), ModelConfig(random_state=42))
    first = splitter.split(make_data())[0].index
    second = splitter.split(make_data())[0].index

    assert first.equals(second)
