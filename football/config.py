from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    dataset_name: str = "hubertsidorowicz/football-players-stats-2024-2025"
    file_name: str = "players_data-2024_2025.csv"
    target: str = "xA"
    excluded_positions: List[str] = field(default_factory=lambda: ["GK"])
    min_minutes: int = 270
