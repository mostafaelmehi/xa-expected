import pytest
from fastapi.testclient import TestClient

from football import api


class FakeSchema:
    def __init__(self, names):
        self.names = names

    def input_names(self):
        return list(self.names)


class FakeModel:
    """Stands in for the MLflow model: same attributes the API uses, no MLflow needed."""

    def __init__(self, names=None):
        self.metadata = type(
            "Meta",
            (),
            {"get_input_schema": lambda _: FakeSchema(names or api.FEATURES)},
        )()

    def predict(self, df):
        # Deterministic: 0.1 * KP per row, so the tests can check exact values.
        return df["KP"].to_numpy() * 0.1


def use_model(monkeypatch, model):
    """Make the API start with this model instead of loading one from MLflow."""
    monkeypatch.setattr(api, "load_champion", lambda: (model, "7"))


@pytest.fixture
def client(monkeypatch):
    use_model(monkeypatch, FakeModel())
    with TestClient(api.app) as c:  # "with" runs the startup (lifespan)
        yield c


def player(**overrides):
    row = {name: 1.0 for name in api.FEATURES}
    row.update(overrides)
    return row


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info_reports_version_and_features(client):
    info = client.get("/model-info").json()
    assert info["version"] == "7"
    assert info["n_features"] == len(api.FEATURES)


def test_predict_returns_one_prediction_per_player(client):
    body = {"instances": [player(KP=10.0), player(KP=20.0)]}
    response = client.post("/predict", json=body)

    assert response.status_code == 200
    assert response.json()["predictions"] == pytest.approx([1.0, 2.0])
    assert response.json()["model_version"] == "7"


def test_missing_feature_is_rejected(client):
    row = player()
    del row["KP"]
    response = client.post("/predict", json={"instances": [row]})

    assert response.status_code == 422
    assert "KP" in str(response.json()["detail"])


def test_unknown_feature_is_rejected(client):
    response = client.post("/predict", json={"instances": [player(bogus=1.0)]})
    assert response.status_code == 422


def test_non_numeric_value_is_rejected(client):
    response = client.post("/predict", json={"instances": [player(KP="abc")]})
    assert response.status_code == 422


def test_empty_instances_is_rejected(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_startup_fails_if_model_and_config_disagree(monkeypatch):
    # The model lacks one feature that the config expects.
    use_model(monkeypatch, FakeModel(names=api.FEATURES[:-1]))

    with pytest.raises(RuntimeError, match="do not match"), TestClient(api.app):
        pass
