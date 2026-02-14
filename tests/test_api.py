from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Churn API is running"
