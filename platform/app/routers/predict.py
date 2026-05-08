"""POST /predict — single prediction endpoint.

1. Validate request against PredictRequest Pydantic model
2. Convert request fields to DataFrame in correct column order
3. Add pdays_never_contacted sentinel flag
4. Pass through pipeline.predict_proba()
5. Apply operating threshold
6. Return PredictResponse

On bad input: FastAPI automatically returns structured 422.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request

from app.dependencies import get_model, get_threshold
from app.schemas.predict_request import PredictRequest
from app.schemas.predict_response import PredictResponse

router = APIRouter()

FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
]


@router.post("/", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    request: Request,
    model=Depends(get_model),
    threshold: float = Depends(get_threshold),
) -> PredictResponse:
    row = pd.DataFrame([{
        "age": body.age,
        "job": body.job,
        "marital": body.marital,
        "education": body.education,
        "default": body.default,
        "housing": body.housing,
        "loan": body.loan,
        "contact": body.contact,
        "month": body.month,
        "day_of_week": body.day_of_week,
        "campaign": body.campaign,
        "pdays": body.pdays,
        "previous": body.previous,
        "poutcome": body.poutcome,
        "emp.var.rate": body.emp_var_rate,
        "cons.price.idx": body.cons_price_idx,
        "cons.conf.idx": body.cons_conf_idx,
        "euribor3m": body.euribor3m,
        "nr.employed": body.nr_employed,
    }])
    row["pdays_never_contacted"] = (row["pdays"] == 999).astype(int)
    row = row[FEATURE_COLUMNS + ["pdays_never_contacted"]]

    proba = float(model.predict_proba(row)[0, 1])
    prediction = int(proba >= threshold)

    state = request.app.state
    if hasattr(state, "drift_accumulator"):
        state.drift_accumulator.append({
            "age": body.age,
            "campaign": body.campaign,
            "pdays": body.pdays,
            "previous": body.previous,
            "emp.var.rate": body.emp_var_rate,
            "cons.price.idx": body.cons_price_idx,
            "cons.conf.idx": body.cons_conf_idx,
            "euribor3m": body.euribor3m,
            "nr.employed": body.nr_employed,
            "job": body.job,
            "marital": body.marital,
            "education": body.education,
            "default": body.default,
            "housing": body.housing,
            "loan": body.loan,
            "contact": body.contact,
            "month": body.month,
            "day_of_week": body.day_of_week,
            "poutcome": body.poutcome,
            "proba": proba,
        })

    return PredictResponse(prediction=prediction, probability=proba)
