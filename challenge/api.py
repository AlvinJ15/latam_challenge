import fastapi
import numpy as np
import pandas as pd
from fastapi import HTTPException

from pydantic import BaseModel, validator

from challenge.model import DelayModel

app = fastapi.FastAPI()
model = DelayModel()


class FlightPredictionItem(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("TIPOVUELO")
    def validate_tipovuelo(cls, value):
        if value not in {"N", "I"}:
            raise HTTPException(status_code=400, detail="value exceeded limit")
        return value

    @validator("MES")
    def validate_mes(cls, value):
        if value < 1 or value > 12:
            raise HTTPException(status_code=400, detail="value exceeded limit")
        return value


class PredictionRequest(BaseModel):
    flights: list[FlightPredictionItem]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(request_body: PredictionRequest) -> dict:
    data = []
    for flight in request_body.flights:
        data.append({
            "OPERA": flight.OPERA,
            "TIPOVUELO": flight.TIPOVUELO,
            "MES": flight.MES
        })

    df = pd.DataFrame(data)
    predictions = model.predict(model.preprocess(df))
    predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

    # Devolver las predicciones en la respuesta
    return {"predict": predictions}
