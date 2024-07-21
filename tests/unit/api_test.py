import sys
import pytest
from fastapi.testclient import TestClient
from main import app

test_client = TestClient(app=app)

def test_welcome_page():
    return_value = test_client.get('/')
    assert return_value.status_code == 200
    assert return_value.text == '["welcome here, this is a REST API"]'

def test_predict(census_record):    
    # Send a POST request to the /items/ endpoint
    response = test_client.post("/predict", json=census_record)
    assert response.status_code == 200, f"Resposnse: actual status code {response.status_code}, response body: {response.text}"

if __name__ == "__main__":
    # The `-v` flag is for verbose mode
    pytest_args = ["-v"]
    pytest_args += sys.argv[1:]
    pytest.main(pytest_args)
