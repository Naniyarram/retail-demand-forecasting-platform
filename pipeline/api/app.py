"""
app.py

FastAPI application for serving champion forecasts.

Run locally:
    uvicorn pipeline.api.app:app --host 0.0.0.0 --port 8000

"""

from contextlib import asynccontextmanager
import time
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pipeline.api.forecast_service import ForecastService
from pipeline.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ModelMetadataResponse,
    InventoryOptimizeRequest,
    InventoryOptimizeResponse,
    RiskClassifyRequest,
    RiskClassifyResponse,
    LLMRecommendationRequest,
    LLMRecommendationResponse,
    RetailChatRequest,
    RetailChatResponse,
    MetricsResponse
)
from pipeline.inventory.optimization import optimize_inventory
from pipeline.inventory.risk import classify_risk
from pipeline.utils.conversational_assistant import ConversationalRetailAssistant
from pipeline.utils.llm_client import HFLLMClient


forecast_service = ForecastService()
llm_client = HFLLMClient()
chat_assistant = ConversationalRetailAssistant(
    llm_client=llm_client
)

# System Metrics
system_metrics = {
    "total_requests": 0,
    "requests_by_endpoint": {
        "/health": 0,
        "/model": 0,
        "/forecast": 0,
        "/inventory/optimize": 0,
        "/inventory/risk": 0,
        "/decision/recommendations": 0,
        "/decision/chat": 0,
        "/monitoring/metrics": 0
    },
    "latencies": {
        "/health": 0.0,
        "/model": 0.0,
        "/forecast": 0.0,
        "/inventory/optimize": 0.0,
        "/inventory/risk": 0.0,
        "/decision/recommendations": 0.0,
        "/decision/chat": 0.0,
        "/monitoring/metrics": 0.0
    }
}


def track_metric(endpoint: str):
    system_metrics["total_requests"] += 1
    if endpoint in system_metrics["requests_by_endpoint"]:
        system_metrics["requests_by_endpoint"][endpoint] += 1


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the champion model when the API starts.
    Model loading is best-effort: if the artifact is missing or
    incompatible the server still starts and returns 503 on forecast calls.
    """
    try:
        forecast_service.load_model()
        print("[INFO] Champion model loaded successfully.")
    except Exception as exc:
        # Covers FileNotFoundError, ModuleNotFoundError (stale pickle),
        # numpy compatibility errors between environments, etc.
        print(
            f"[WARNING] Champion model could not be pre-loaded: {exc}. "
              "The API will start without a cached model. "
            "Run /model or /forecast to trigger lazy loading, "
            "or regenerate the artifact with: python run_experiments.py"
        )
    yield


app = FastAPI(
    title="Retail Demand Forecasting API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Measure response latency of API endpoints in milliseconds.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    path = request.url.path
    # Expose and record latency
    if path in system_metrics["latencies"]:
        system_metrics["latencies"][path] = round(process_time * 1000, 2)
        
    return response


@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check() -> HealthResponse:
    """
    Return API and model health.
    """
    track_metric("/health")
    return HealthResponse(
        status="ok",
        model_loaded=forecast_service.is_model_loaded(),
        model_name=(
            forecast_service.get_model_name()
            if forecast_service.is_model_loaded()
            else None
        )
    )


@app.get(
    "/model",
    response_model=ModelMetadataResponse
)
def model_metadata() -> ModelMetadataResponse:
    """
    Return served model metadata.
    """
    track_metric("/model")
    if not forecast_service.is_model_loaded():
        try:
            forecast_service.load_model()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=str(exc)
            ) from exc

    return ModelMetadataResponse(
        **forecast_service.get_metadata()
    )


@app.post(
    "/forecast",
    response_model=ForecastResponse
)
def forecast(
    request: ForecastRequest
) -> ForecastResponse:
    """
    Generate a demand forecast.
    """
    track_metric("/forecast")
    try:
        predictions = forecast_service.forecast(
            forecast_horizon=request.forecast_horizon
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    return ForecastResponse(
        model_name=forecast_service.get_model_name(),
        forecast_horizon=request.forecast_horizon,
        forecast=predictions,
        store_id=request.store_id,
        department_id=request.department_id
    )


@app.post(
    "/inventory/optimize",
    response_model=InventoryOptimizeResponse
)
def optimize_inventory_endpoint(
    request: InventoryOptimizeRequest
) -> InventoryOptimizeResponse:
    """
    Perform safety stock, ROP, and EOQ calculations.
    """
    track_metric("/inventory/optimize")
    try:
        results = optimize_inventory(
            forecast_demands=request.forecast_demands,
            historical_sales_std=request.historical_sales_std,
            lead_time_weeks=request.lead_time_weeks,
            service_level=request.service_level,
            holding_cost_unit_year=request.holding_cost_unit_year,
            setup_cost_order=request.setup_cost_order
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    return InventoryOptimizeResponse(**results)


@app.post(
    "/inventory/risk",
    response_model=RiskClassifyResponse
)
def classify_risk_endpoint(
    request: RiskClassifyRequest
) -> RiskClassifyResponse:
    """
    Classify stockout and overstock risk levels.
    """
    track_metric("/inventory/risk")
    results = classify_risk(
        current_inventory=request.current_inventory,
        reorder_point=request.reorder_point,
        safety_stock=request.safety_stock,
        total_forecasted_demand=request.total_forecasted_demand
    )
    return RiskClassifyResponse(**results)


@app.post(
    "/decision/recommendations",
    response_model=LLMRecommendationResponse
)
def generate_recommendations(
    request: LLMRecommendationRequest
) -> LLMRecommendationResponse:
    """
    Generate natural language retail insights and recommendations using Llama 3.1 8B.
    """
    track_metric("/decision/recommendations")
    forecast_data = {
        "store_id": request.store_id,
        "department_id": request.department_id,
        "horizon": request.horizon,
        "average_historical": request.average_historical,
        "average_forecast": request.average_forecast,
        "total_forecast": request.total_forecast,
        "trend_direction": request.trend_direction,
        "change_pct": request.change_pct
    }
    
    results = llm_client.generate_retail_insights(forecast_data)
    return LLMRecommendationResponse(**results)


@app.post(
    "/decision/chat",
    response_model=RetailChatResponse
)
def chat_with_retail_assistant(
    request: RetailChatRequest
) -> RetailChatResponse:
    """
    Answer conversational business questions using forecast and inventory context.
    """
    track_metric("/decision/chat")
    history = [
        message.model_dump()
        for message in request.conversation_history
    ]
    try:
        result = chat_assistant.answer_question(
            question=request.question,
            business_context=request.business_context,
            conversation_history=history
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    return RetailChatResponse(**result)


@app.get(
    "/monitoring/metrics",
    response_model=MetricsResponse
)
def get_metrics() -> MetricsResponse:
    """
    Expose serving usage and model loading status metrics.
    """
    track_metric("/monitoring/metrics")
    return MetricsResponse(
        total_requests=system_metrics["total_requests"],
        requests_by_endpoint=system_metrics["requests_by_endpoint"],
        model_loaded=forecast_service.is_model_loaded(),
        active_model_name=(
            forecast_service.get_model_name()
            if forecast_service.is_model_loaded()
            else None
        ),
        latencies=system_metrics["latencies"]
    )
