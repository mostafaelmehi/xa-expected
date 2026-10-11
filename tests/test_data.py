import pandas as pd

from football.config import DataConfig
from football.dataset import DataPreprocessor


def make_players() -> pd.DataFrame:
    """A tiny fake dataset: one goalkeeper, one low-minutes player, two valid players."""
    return pd.DataFrame(
        {
            "Pos": ["GK", "DF", "MF", "FW"],
            "Min": [900, 1000, 100, 2000],
            "xA": [0.1, 0.5, 0.2, 3.0],
            "xAG": [0.1, 0.6, 0.3, 3.1],
            "G/SoT": [None, 0.3, None, 0.5],
            "On-Off": [0.1, None, 0.2, 0.4],
        }
    )


def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.target == "xA"
    assert "GK" in cfg.excluded_positions
    assert cfg.min_minutes == 270


def test_filter_rows_removes_goalkeepers_and_low_minutes():
    out = DataPreprocessor(DataConfig()).filter_rows(make_players())
    assert out["Pos"].tolist() == ["DF", "FW"]


def test_drop_listed_columns_removes_leakage_but_keeps_target():
    out = DataPreprocessor(DataConfig()).drop_listed_columns(make_players())
    assert "xAG" not in out.columns
    assert "xA" in out.columns


def test_handle_missing_values_fills_ratios_and_drops_rows():
    cfg = DataConfig(zero_fill_columns=["G/SoT"], dropna_columns=["On-Off"])
    out = DataPreprocessor(cfg).handle_missing_values(make_players())
    assert len(out) == 3  # the row with a missing On-Off is dropped
    assert out.isna().sum().sum() == 0
