"""
schemas.py

Request and response contracts for the forecasting API.

"""

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from pipeline.config.settings import (
    FORECAST_HORIZON,
    MAX_FORECAST_HORIZON
)


class ForecastRequest(BaseModel):
    """
    Forecast request payload.

    Store and department are optional metadata today because
    the served champion artifact already represents one trained
    time series scope, such as company, store, or store-department.
    """

    model_config = ConfigDict(
        protected_namespaces=()
    )

    forecast_horizon: int = Field(
        default=FORECAST_HORIZON,
        ge=1,
        le=MAX_FORECAST_HORIZON
    )

    store_id: int | None = Field(
        default=None,
        ge=1
    )

    department_id: int | None = Field(
        default=None,
        ge=1
    )


class ForecastResponse(BaseModel):
    """
    Forecast response payload.
    """

    model_config = ConfigDict(
        protected_namespaces=()
    )

    model_name: str

    forecast_horizon: int

    forecast: list[float]

    store_id: int | None = None

    department_id: int | None = None


class HealthResponse(BaseModel):
    """
    API health response.
    """

    model_config = ConfigDict(
        protected_namespaces=()
    )

    status: str

    model_loaded: bool

    model_name: str | None = None


class ModelMetadataResponse(BaseModel):
    """
    Served model metadata.
    """

    model_config = ConfigDict(
        protected_namespaces=()
    )

    model_name: str

    artifact_path: str

    metadata: dict[str, Any]


class InventoryOptimizeRequest(BaseModel):
    forecast_demands: list[float]
    historical_sales_std: float
    lead_time_weeks: float = 2.0
    service_level: float = 0.95
    holding_cost_unit_year: float = 1.5
    setup_cost_order: float = 50.0


class InventoryOptimizeResponse(BaseModel):
    average_forecasted_demand: float
    safety_stock: float
    reorder_point: float
    economic_order_quantity: float
    parameters: dict[str, Any]


class RiskClassifyRequest(BaseModel):
    current_inventory: float
    reorder_point: float
    safety_stock: float
    total_forecasted_demand: float


class RiskClassifyResponse(BaseModel):
    stockout_risk: dict[str, str]
    overstock_risk: dict[str, str]
    metrics: dict[str, Any]


class LLMRecommendationRequest(BaseModel):
    store_id: int
    department_id: int | None = None
    horizon: int
    average_historical: float
    average_forecast: float
    total_forecast: float
    trend_direction: str
    change_pct: float


class LLMRecommendationResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )
    raw_insights: str
    verified: bool
    model_used: str


class ChatMessage(BaseModel):
    role: str
    content: str


class RetailChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500
    )
    business_context: dict[str, Any]
    conversation_history: list[ChatMessage] = Field(
        default_factory=list
    )


class RetailChatResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )
    answer: str
    verified: bool
    model_used: str
    detected_intent: str
    referenced_entity: str
    analysis: dict[str, Any]
    conversation_history: list[ChatMessage]


class MetricsResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )
    total_requests: int
    requests_by_endpoint: dict[str, int]
    model_loaded: bool
    active_model_name: str | None
    latencies: dict[str, float] = Field(default_factory=dict)

