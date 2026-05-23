"""Pydantic request/response schemas."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TransactionFeatures(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    merchant_category: int = Field(..., ge=0, le=50)
    distance_from_home: float = Field(..., ge=0)
    distance_from_last_transaction: float = Field(..., ge=0)
    ratio_to_median_purchase_price: float = Field(..., gt=0)
    repeat_retailer: int = Field(..., ge=0, le=1)
    used_chip: int = Field(..., ge=0, le=1)
    used_pin_number: int = Field(..., ge=0, le=1)
    online_order: int = Field(..., ge=0, le=1)

    model_config = {"json_schema_extra": {
        "example": {
            "amount": 149.99,
            "hour": 14,
            "day_of_week": 2,
            "merchant_category": 5,
            "distance_from_home": 12.5,
            "distance_from_last_transaction": 3.2,
            "ratio_to_median_purchase_price": 1.8,
            "repeat_retailer": 1,
            "used_chip": 1,
            "used_pin_number": 0,
            "online_order": 0,
        }
    }}


class PredictionRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Optional transaction identifier")
    features: TransactionFeatures


class BatchPredictionRequest(BaseModel):
    transactions: List[PredictionRequest] = Field(..., max_length=1000)


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    confidence: str
    model_version: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_detected: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    stage: str
    metrics: Dict[str, float]
    features: List[str]
    threshold: float
