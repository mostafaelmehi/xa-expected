from football.config import DataConfig


def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.target == "xA"
    assert "GK" in cfg.excluded_positions
    assert cfg.min_minutes == 270
