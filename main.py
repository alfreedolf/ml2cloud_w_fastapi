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
class CensusSalaryRequest(BaseModel):
    age: int = Field(..., example=43)
    workclass: str = Field(..., example="private")
    fnlgt: int = Field(..., example=174524)
    education: str = Field(..., example="10th")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="never-married", alias="marital-status")
    occupation: str = Field(..., example="prof-specialty")
    relationship: str = Field(..., example="husband")
    race: str = Field(..., example="white")
    sex: str = Field(..., example="male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="united-states", alias="native-country")
    salary: str = Field(None, example="<=50k")
    model: str = Field(None, example="default")
    serialized_encoder:str = Field(None, example="default")
    serialized_lb:str = Field(None, example="default")

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
async def do_inference(request: CensusSalaryRequest):
    
    request_dict = request.dict(by_alias=True)
    model = request_dict.pop('model', 'default')
    serialized_encoder = request_dict.pop('serialized_encoder', 'default')
    serialized_lb = request_dict.pop('serialized_lb', 'default')
        
    X, y, encoder, lb = process_data(
        X=pd.DataFrame(request_dict, index=[0]),
        categorical_features=CATEGORICAL_FEATURES,
        training=False,
        label="salary",
        serialized_encoder=serialized_encoder,
        serialized_lb=serialized_lb
    )    
    predictions = inference(X=X, model=model)
    return predictions.tolist()
