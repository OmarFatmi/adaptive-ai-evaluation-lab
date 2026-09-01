import os
os.environ["DATABASE_URL"]="sqlite:///./test_adaptive_lab.db"
from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def test_health(): assert client.get("/api/health").status_code==200
def test_models_and_experiment():
 with client:
  models=client.get("/api/models").json(); assert models
  body={"name":"test","model_ids":[models[0]["id"]],"categories":["arithmetic"],"budget":2,"policy":"ucb1","seed":2}
  r=client.post("/api/experiments",json=body); assert r.status_code==200
