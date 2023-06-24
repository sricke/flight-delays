import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

model_name = "flight_delay.json"
model = XGBClassifier()
model.load_model(os.path.join("models", model_name))

app = FastAPI()

# assign vocab for features between api model and prediction model values
vocab = {
            "month": "MES",
            "day": "DIANOM",
            "flight_type": "TIPOVUELO",
            "airline": "OPERA",
            "city_origin": "SIGLAORI", 
            "city_destiny": "SIGLADES", 
            "high_season": "Temp-A",
            "time_day": "Per-D"
        }

# class for api model feature names and types
class FlightInfo(BaseModel):
    month: str
    day: str
    flight_type: str
    airline: str
    city_origin: str
    city_destiny: str
    high_season: bool
    time_day: str

# home root
@app.get("/")
def root():
    return {"message": "Flight Delay Model"}

# root for predicting delay instance
@app.post("/predict")
def predict_delay(flight_info: FlightInfo):
    data = {}
    for key, value in flight_info.dict().items():
        feature = vocab[key] # get column name
        data[feature] = [value] # assign value
    df = pd.DataFrame(data).astype("category") # make sure value types
    prediction = model.predict_proba(df) 
    return prediction.item(0)