# Put the code for your API here.
from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
import pandas as pd
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import inference


# set categorical data
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Declare the data object with its components and their type.
class CensusRecord(BaseModel):
    age: str = Field(..., example="John Doe")
    workclass: str = Field(..., example="state-gov")
    fnlgt: int = Field(..., example=123456)
    education: str = Field(..., example="masters")
    education_num: int = Field(..., example=9)
    marital_status: str = Field(..., example="never-married")
    occupation: str = Field(..., example="sales")
    relationship: str = Field(..., example="wife")
    race: str = Field(..., example="black")
    sex: str = Field(..., example="male")
    capital_gain: int = Field(..., example=0)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="Italy")
    salary: int = Field(None, example="<=50k")

# Save items from POST method in the memory
items = {}

# Initialize FastAPI instance
app = FastAPI()


# A GET that returns a welcome message
@app.get("/")
async def welcome():
    return {"welcome here, this is a REST API"}


# A POST method that does inference
@app.post("/predict")
async def do_inference(record: CensusRecord):
    X, y, encoder, lb = process_data(
        X=pd.DataFrame(record.dict),
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=False,
    )    
    predictions = inference(X=X, model="model_10-fold_sex-male.pkl")
    return predictions
