"""
optimization.py

Inventory optimization engine for safety stock, ROP, and EOQ calculations.


"""

import math
from typing import Dict, Any


def get_z_score(service_level: float) -> float:
    """
    Get the Z-score for a given service level using a standard normal distribution lookup.
    Supports common service levels in retail.
    """
    if service_level >= 0.999:
        return 3.09
    if service_level >= 0.99:
        return 2.33
    if service_level >= 0.98:
        return 2.05
    if service_level >= 0.95:
        return 1.645
    if service_level >= 0.90:
        return 1.28
    if service_level >= 0.85:
        return 1.04
    if service_level >= 0.80:
        return 0.84
    return 1.0  # Default fallback


def calculate_safety_stock(
    demand_std: float,
    lead_time_weeks: float,
    service_level: float = 0.95
) -> float:
    """
    Calculate safety stock: SS = Z * std_d * sqrt(L).
    """
    z = get_z_score(service_level)
    return round(z * demand_std * math.sqrt(lead_time_weeks), 2)


def calculate_reorder_point(
    average_demand: float,
    lead_time_weeks: float,
    safety_stock: float
) -> float:
    """
    Calculate Reorder Point: ROP = (average_demand * lead_time_weeks) + safety_stock.
    """
    return round((average_demand * lead_time_weeks) + safety_stock, 2)


def calculate_eoq(
    average_weekly_demand: float,
    holding_cost_per_unit_year: float,
    setup_cost_per_order: float
) -> float:
    """
    Calculate Economic Order Quantity (EOQ): EOQ = sqrt((2 * D * S) / H).
    D is the annual demand (weekly demand * 52).
    """
    annual_demand = average_weekly_demand * 52
    if holding_cost_per_unit_year <= 0:
        return 0.0
    eoq = math.sqrt((2 * annual_demand * setup_cost_per_order) / holding_cost_per_unit_year)
    return round(eoq, 2)


def optimize_inventory(
    forecast_demands: list[float],
    historical_sales_std: float,
    lead_time_weeks: float = 2.0,
    service_level: float = 0.95,
    holding_cost_unit_year: float = 1.5,
    setup_cost_order: float = 50.0
) -> Dict[str, Any]:
    """
    Generate optimized inventory settings based on demand forecast array.
    """
    if not forecast_demands:
        raise ValueError("Forecast demands list cannot be empty.")
    
    avg_demand = sum(forecast_demands) / len(forecast_demands)
    safety_stock = calculate_safety_stock(
        demand_std=historical_sales_std,
        lead_time_weeks=lead_time_weeks,
        service_level=service_level
    )
    rop = calculate_reorder_point(
        average_demand=avg_demand,
        lead_time_weeks=lead_time_weeks,
        safety_stock=safety_stock
    )
    eoq = calculate_eoq(
        average_weekly_demand=avg_demand,
        holding_cost_per_unit_year=holding_cost_unit_year,
        setup_cost_per_order=setup_cost_order
    )

    return {
        "average_forecasted_demand": round(avg_demand, 2),
        "safety_stock": safety_stock,
        "reorder_point": rop,
        "economic_order_quantity": eoq,
        "parameters": {
            "lead_time_weeks": lead_time_weeks,
            "service_level": service_level,
            "holding_cost_unit_year": holding_cost_unit_year,
            "setup_cost_order": setup_cost_order
        }
    }
