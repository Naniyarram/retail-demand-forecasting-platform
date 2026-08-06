"""
FastAPI application serving champion forecasts.

Run locally:
    uvicorn pipeline.api.app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
import time
from typing import Dict, Any

from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pipeline.api.forecast_service import ForecastService
from pipeline.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    ReadyResponse,
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
        "/ready": 0,
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
        "/ready": 0.0,
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
    Loads champion model at startup.
    Fails gracefully if the artifact is missing or incompatible so that the 
    server can still start (forecast requests will return 503).
    """
    try:
        forecast_service.load_model()
        print("[INFO] Champion model loaded successfully.")
    except Exception as exc:
        # Handles missing files, stale pickles, or numpy environment mismatches.
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
    """Measures endpoint latency in milliseconds."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    path = request.url.path
    # Record latency
    if path in system_metrics["latencies"]:
        system_metrics["latencies"][path] = round(process_time * 1000, 2)
        
    return response


@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check() -> HealthResponse:
    """Liveness probe. Quick check if the container is running."""
    track_metric("/health")
    return HealthResponse(
        status="healthy",
        service="RetailCast API",
        version="1.0.0",
        environment="production",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get(
    "/ready",
    response_model=ReadyResponse
)
def readiness_check() -> ReadyResponse:
    """Readiness probe. Checks if the forecasting model is loaded and ready."""
    track_metric("/ready")
    is_loaded = forecast_service.is_model_loaded()
    if not is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet"
        )
    return ReadyResponse(
        status="ready",
        model_loaded=True,
        service="RetailCast API"
    )


@app.get(
    "/model",
    response_model=ModelMetadataResponse
)
def model_metadata() -> ModelMetadataResponse:
    """Returns active model metadata."""
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
    """Generates demand forecast predictions for the requested horizon."""
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
    """Calculates safety stock, reorder point (ROP), and economic order quantity (EOQ)."""
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
    """Classifies risk level for stockouts and overstocks."""
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
    """Generates natural language retail insights and recommendations using the LLM."""
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
    """Handles chat queries using forecast and inventory business context."""
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
    """Exposes usage metrics, status, and latencies for monitoring."""
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
