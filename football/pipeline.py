import logging

from football.config import load_config
from football.dataset import DataAcquisition, DataPreprocessor
from football.features import PositionFeatures
from football.modeling.split import DataSplitter

logger = logging.getLogger(__name__)


def run_pipeline():
    config = load_config()

    # 1. Data: download -> load -> validate, then clean
    df = DataAcquisition(config.data).run()
    df = DataPreprocessor(config.data).run(df)

    # 2. Position features (stateless, so safe before the split)
    df = PositionFeatures(tuple(config.features.main_positions)).transform(df)

    # 3. Split: everything that learns from data happens after this line
    X_train, X_test, y_train, y_test = DataSplitter(config.data, config.model).split(df)

    logger.info("X_train %s | X_test %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    run_pipeline()
