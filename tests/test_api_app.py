"""
test_api_app.py

API route tests for forecast serving.
"""

from fastapi.testclient import TestClient

from pipeline.api import app as api_app


class DummyForecastService:
    """
    Test double for API route behavior.
    """

    def __init__(self):

        self.loaded = True

    def is_model_loaded(self):

        return self.loaded

    def get_model_name(self):

        return "DummyForecaster"

    def get_metadata(self):

        return {
            "model_name": "DummyForecaster",
            "artifact_path": "memory://dummy",
            "metadata": {
                "RMSE": 10.0
            }
        }

    def load_model(self):

        self.loaded = True

    def forecast(
        self,
        forecast_horizon
    ):

        return [1.0] * forecast_horizon


def test_health_endpoint(
    monkeypatch
):

    monkeypatch.setattr(
        api_app,
        "forecast_service",
        DummyForecastService()
    )

    client = TestClient(
        api_app.app
    )

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "RetailCast API"


def test_ready_endpoint(
    monkeypatch
):
    service = DummyForecastService()
    monkeypatch.setattr(
        api_app,
        "forecast_service",
        service
    )

    client = TestClient(
        api_app.app
    )

    # Test Ready (model loaded)
    response = client.get(
        "/ready"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["service"] == "RetailCast API"

    # Test Not Ready (model not loaded)
    service.loaded = False
    response_not_ready = client.get(
        "/ready"
    )
    assert response_not_ready.status_code == 503



def test_forecast_endpoint(
    monkeypatch
):

    monkeypatch.setattr(
        api_app,
        "forecast_service",
        DummyForecastService()
    )

    client = TestClient(
        api_app.app
    )

    response = client.post(
        "/forecast",
        json={
            "store_id": 1,
            "department_id": 1,
            "forecast_horizon": 4
        }
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["model_name"] == "DummyForecaster"
    assert payload["forecast"] == [
        1.0,
        1.0,
        1.0,
        1.0
    ]
