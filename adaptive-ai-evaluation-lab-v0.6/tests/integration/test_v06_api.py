from fastapi.testclient import TestClient

from app.main import app


def test_rl_environment_spec():
    with TestClient(app) as client:
        response = client.get("/api/rl/environment")
        assert response.status_code == 200
        body = response.json()
        assert body["version"] == "0.6.0"
        assert body["observation_dimensions"] == 13


def test_dashboard_and_models_end_to_end():
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/dashboard").status_code == 200
        models = client.get("/api/models")
        assert models.status_code == 200
        assert len(models.json()) >= 1
