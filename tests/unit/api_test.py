import sys
import pytest
from fastapi.testclient import TestClient
from main import app

test_client = TestClient(app=app)


def test_welcome_page():
    response = test_client.get("/")
    assert response.status_code == 200
    json_content = response.json()
    assert json_content == ["welcome here, this is a REST API"]


def test_high_salary(census_record_high_salary):
    # Send a POST request to the /items/ endpoint
    response = test_client.post("/predict", json=census_record_high_salary)
    assert (
        response.status_code == 200
    ), f"Resposnse: actual status code {response.status_code}, response body: {response.text}"
    json_content = response.json()
    assert json_content == [1]


def test_low_salary(census_record_low_salary):
    # Send a POST request to the /items/ endpoint
    response = test_client.post("/predict", json=census_record_low_salary)
    assert (
        response.status_code == 200
    ), f"Resposnse: actual status code {response.status_code}, response body: {response.text}"
    json_content = response.json()
    assert json_content == [0]


if __name__ == "__main__":
    # The `-v` flag is for verbose mode
    pytest_args = ["-v"]
    pytest_args += sys.argv[1:]
    pytest.main(pytest_args)
