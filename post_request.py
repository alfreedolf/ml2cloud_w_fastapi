import ast
import json
import logging
import requests
import argparse

DEFAULT_PAYLOAD = {
    "age": 33,
    "workclass": "private",
    "fnlgt": 45785,
    "education": "masters",
    "education-num": 14,
    "marital-status": "never-married",
    "occupation": "prof-specialty",
    "relationship": "not-in-family",
    "race": "white",
    "sex": "female",
    "capital-gain": 15000,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "united-states",
    "model": "default",
    "serialized-encoder": "default",
    "serialized-lb": "default"
}

DEFAULT_URL = "http://localhost:80/predict"

# "model": "test",
# "serialized_encoder": "test",
# "serialized_lb": "test"


def send_post(endpoint_url: str, payload: str) -> int:

    # Headers (optional)
    headers = {"Content-Type": "application/json"}

    if payload == "":
        payload: dict = DEFAULT_PAYLOAD
    else:
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as e:
            logging.error("Error: Invalid JSON format - %s", e)
            return -1

    # Sending the POST request
    response = requests.post(endpoint_url, json=payload, headers=headers)

    # Printing the response
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
    return response.status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=False, default=DEFAULT_URL, help="URL of the FastAPI POST endpoint")
    parser.add_argument("--payload", type=str, required=False, default="", help="URL of the FastAPI POST endpoint")
    args = parser.parse_args()

    send_post(args.url, args.payload)
