# Put the code for your API here.
from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel


# Declare the data object with its components and their type.
class CensusRecord(BaseModel):
    age: str
    workclass: str
    fnlgt: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    salary: int

# Save items from POST method in the memory
items = {}

# Initialize FastAPI instance
app = FastAPI()


# A GET that returns a welcome message
@app.get("/")
async def welcome():
    return {"welcome here, this is a REST API"}


# A GET that returns a welcome message
@app.post("/predict")
async def inference(record: CensusRecord):
    pass
