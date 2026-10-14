from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class DataConfig:
    dataset_name: str = "hubertsidorowicz/football-players-stats-2024-2025"
    file_name: str = "players_data-2024_2025.csv"
    target: str = "xA"
    excluded_positions: List[str] = field(default_factory=lambda: ["GK"])
    min_minutes: int = 270
    # Defensive statistics: not relevant for explaining chance creation.
    defensive_columns: List[str] = field(
        default_factory=lambda: [
            "Tkl",
            "TklW",
            "Def 3rd",
            "Mid 3rd",
            "Att 3rd",
            "Att_stats_defense",
            "Tkl%",
            "Lost",
            "Blocks",
            "Blocks_stats_defense",
            "Sh_stats_defense",
            "Pass",
            "Int",
            "Tkl+Int",
            "Clr",
            "Err",
            "Recov",
            "Fls",
            "Tkld",
            "Tkld%",
        ]
    )

    # Identifiers: not performance features (e.g. 'Player' has ~2000 unique values).
    identifier_columns: List[str] = field(
        default_factory=lambda: ["Player", "Nation", "Squad", "Comp"]
    )

    # Leakage: columns that contain or nearly equal the target (xA).
    leakage_columns: List[str] = field(
        default_factory=lambda: [
            "xAG",
            "Ast",
            "G+A",
            "G+A-PK",
            "xG+xAG",
            "npxG+xAG",
            "A-xAG",
        ]
    )

    # Dropped on purpose: high share of missing values, or demographics (bias risk).
    low_value_columns: List[str] = field(
        default_factory=lambda: ["Mn/Sub", "Mn/Start", "Age", "Born"]
    )
    # Ratio columns that are NaN when the denominator is 0 or when FBref's minimum
    # threshold is not met. NaN is replaced by 0 ("no activity"), not treated as an error.
    zero_fill_columns: List[str] = field(
        default_factory=lambda: [
            "G/SoT",
            "G/Sh",
            "npxG/Sh",
            "SoT%",
            "Dist",
            "Succ%",
            "Won%",
            "Cmp%",
        ]
    )

    # Rows with a missing value in these columns are dropped (very few rows).
    dropna_columns: List[str] = field(default_factory=lambda: ["On-Off"])


@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

@dataclass
class FeatureConfig:
    # Main positions used to build one-hot columns (Main_Pos_DF, ...).
    main_positions: List[str] = field(default_factory=lambda: ["DF", "MF", "FW"])

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)



def load_config() -> Config:
    config = Config()
    if os.getenv("RANDOM_STATE"):
        config.model.random_state = int(os.getenv("RANDOM_STATE"))
    if os.getenv("MIN_MINUTES"):
        config.data.min_minutes = int(os.getenv("MIN_MINUTES"))
    return config
