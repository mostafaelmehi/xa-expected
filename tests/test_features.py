import pandas as pd

from football.features import PositionFeatures


def make_positions() -> pd.DataFrame:
    """Tiny fake data: single position, hybrid position, and a missing FW."""
    return pd.DataFrame({"Pos": ["DF", "MF,FW", "MF"], "Min": [900, 1200, 600]})


def test_position_features_adds_haybird_and_main_pos():
    out = PositionFeatures().transform(make_positions())

    assert "Pos" not in out.columns                       # original column dropped
    assert out["haybird"].tolist() == [0, 1, 0]           # comma means hybrid
    assert out["Main_Pos_MF"].tolist() == [0, 1, 1]       # first listed position wins
    assert out["Main_Pos_DF"].tolist() == [1, 0, 0]


def test_position_features_keeps_all_columns_when_a_position_is_missing():
    out = PositionFeatures().transform(make_positions())

    # No player has FW as the main position, but the column must still exist.
    assert out["Main_Pos_FW"].tolist() == [0, 0, 0]


def test_position_features_does_not_change_input():
    df = make_positions()
    PositionFeatures().transform(df)

    assert "Pos" in df.columns                            # the input stays untouched