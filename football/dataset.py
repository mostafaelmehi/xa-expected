import logging
from pathlib import Path

import kagglehub
import pandas as pd

from football.config import DataConfig

logger = logging.getLogger(__name__)


class DataAcquisition:
    """Download the dataset from Kaggle, load it, and validate its schema."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.file_path: Path | None = None

    def download_dataset(self) -> Path:
        """Download the dataset via kagglehub and return the path to the CSV file.

        kagglehub caches the download, so repeated calls do not re-download.
        """
        path = kagglehub.dataset_download(self.config.dataset_name)
        self.file_path = Path(path) / self.config.file_name

        # Fail early with a clear message if the file name in the config is wrong.
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found after download: {self.file_path}")

        logger.info("Dataset ready at: %s", self.file_path)
        return self.file_path

    def load_data(self) -> pd.DataFrame:
        """Read the downloaded CSV into a DataFrame."""
        if self.file_path is None:
            raise ValueError("Dataset not downloaded. Call download_dataset() first.")

        df = pd.read_csv(self.file_path)
        logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])
        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Check that the columns the pipeline depends on are present.

        Required: the target, the position (to exclude goalkeepers),
        and minutes played (for the minimum-minutes filter).
        """
        required = {self.config.target, "Pos", "Min"}
        missing = required - set(df.columns)
        if missing:
            logger.error("Missing columns: %s", missing)
            return False
        return True

    def run(self) -> pd.DataFrame:
        """Run the full acquisition step: download -> load -> validate."""
        self.download_dataset()
        df = self.load_data()
        if not self.validate_data(df):
            raise ValueError("Data validation failed: missing required columns")
        return df


class DataPreprocessor:
    """Clean the raw data (rows first, columns in later steps)."""

    def __init__(self, config: DataConfig):
        self.config = config

    def filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop excluded positions (goalkeepers) and players with few minutes."""
        # Excluded positions come from the config, so changing them
        # never requires touching this code.
        before = len(df)
        df = df[~df["Pos"].isin(self.config.excluded_positions)]
        logger.info("Removed %d rows (excluded positions)", before - len(df))

        # Players with very few minutes have unstable per-90 and ratio statistics.
        before = len(df)
        df = df[df["Min"] >= self.config.min_minutes]
        logger.info(
            "Removed %d rows (Min < %d)", before - len(df), self.config.min_minutes
        )

        # Return an independent copy so later steps can modify it
        # without triggering SettingWithCopyWarning.
        return df.copy()

    def drop_redundant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that are exact copies of an earlier column (the first is kept).

        The data was merged from several FBref tables, so identifier columns such as
        'Nation' appear again as 'Nation_stats_shooting', 'Nation_stats_passing', etc.
        """
        to_drop: set[str] = set()
        columns = list(df.columns)

        for i, col in enumerate(columns):
            if col in to_drop:
                continue
            for other in columns[i + 1 :]:
                if other not in to_drop and df[col].equals(df[other]):
                    to_drop.add(other)

        logger.info("Removed %d duplicate columns", len(to_drop))
        return df.drop(columns=sorted(to_drop))

    def drop_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that contain only missing values (goalkeeper-only statistics)."""
        empty = df.columns[df.isna().all()]
        logger.info("Removed %d empty columns: %s", len(empty), list(empty))
        return df.drop(columns=empty)

    def drop_listed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop the columns that the config lists as defensive, identifier, leakage or low-value."""
        listed = (
            self.config.defensive_columns
            + self.config.identifier_columns
            + self.config.leakage_columns
            + self.config.low_value_columns
        )
        present = [col for col in listed if col in df.columns]
        absent = sorted(set(listed) - set(present))
        # Prefix rule (e.g. every 'Rk*' column), added on top of the explicit lists.
        present += [
            col
            for col in df.columns
            if col.startswith(tuple(self.config.drop_prefixes)) and col not in present
        ]

        # The notebook used errors='ignore' here; we keep that behaviour
        # but log what was not found so a typo in the config does not go unnoticed.
        if absent:
            logger.warning("Listed columns not found (ignored): %s", absent)

        logger.info("Removed %d listed columns", len(present))
        return df.drop(columns=present)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill undefined ratios with 0 and drop the few rows still missing key values."""
        df = df.copy()

        # A ratio is NaN when its denominator is 0 (e.g. G/SoT with no shots on target)
        # or the player is below FBref's minimum threshold. Both mean "no activity".
        cols = self.config.zero_fill_columns
        df[cols] = df[cols].fillna(0)

        before = len(df)
        df = df.dropna(subset=self.config.dropna_columns)
        logger.info(
            "Removed %d rows with missing %s",
            before - len(df),
            self.config.dropna_columns,
        )

        logger.info("Remaining missing values: %d", int(df.isna().sum().sum()))
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all cleaning steps in order and return the cleaned DataFrame."""
        df = self.filter_rows(df)
        # Must run before the name-based drops: it keeps the first copy of each
        # duplicated column, so the plain names (e.g. 'Nation', 'Tkl') survive.
        df = self.drop_redundant_columns(df)
        df = self.drop_empty_columns(df)
        df = self.drop_listed_columns(df)
        df = self.handle_missing_values(df)
        logger.info("Cleaning done: %d rows, %d columns", df.shape[0], df.shape[1])
        return df
