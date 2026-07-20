"""
business_analytics.py

Small business calculation layer for the RetailCast assistant.

The goal is to compute practical retail facts before the LLM writes an
answer. This keeps the assistant grounded in numbers and makes the logic
easy to explain in interviews.

"""

from __future__ import annotations

import re
from typing import Any


def build_business_analysis(
    question: str,
    context: dict[str, Any],
    intent: str
) -> dict[str, Any]:
    """
    Compute business metrics and scenario facts for a user question.
    """

    contexts = separate_business_contexts(
        question=question,
        context=context
    )
    question_inputs = contexts["current_user_inputs"]
    forecast_context = contexts["forecast_metrics"]
    inventory_context = contexts["inventory_metrics"]

    forecast_values = [
        float(value)
        for value in forecast_context.get("forecast_values", [])
        if _is_number(value)
    ]
    average_forecast = _number(
        forecast_context.get("average_forecast"),
        _mean(forecast_values)
    )
    average_historical = _number(
        forecast_context.get("average_historical"),
        0.0
    )
    total_forecast = _number(
        forecast_context.get("total_forecast"),
        sum(forecast_values)
    )
    horizon = int(
        _number(
            forecast_context.get("horizon"),
            len(forecast_values) or 1
        )
    )

    current_inventory = _number_or_none(
        inventory_context.get("current_inventory")
    )
    lead_time_weeks = _number(
        inventory_context.get("lead_time_weeks"),
        2.0
    )
    service_level = _number(
        inventory_context.get("service_level"),
        0.95
    )
    holding_cost = _number(
        inventory_context.get("holding_cost_unit_year"),
        1.5
    )
    safety_stock = _number_or_none(
        inventory_context.get("safety_stock")
    )
    reorder_point = _number_or_none(
        inventory_context.get("reorder_point")
    )

    inventory_gap = None
    coverage_weeks = None
    coverage_days = None
    stockout_probability = None
    fill_rate = None
    if current_inventory is not None and average_forecast > 0:
        inventory_gap = round(
            total_forecast - current_inventory,
            2
        )
        coverage_weeks = round(
            current_inventory / average_forecast,
            2
        )
        coverage_days = round(
            coverage_weeks * 7,
            1
        )
        if reorder_point is not None:
            stockout_probability = _stockout_probability_proxy(
                current_inventory=current_inventory,
                reorder_point=reorder_point,
                safety_stock=safety_stock or 0.0
            )
        elif inventory_gap > 0:
            stockout_probability = 0.7
        else:
            stockout_probability = 0.15
        fill_rate = _fill_rate_proxy(
            current_inventory=current_inventory,
            total_forecast=total_forecast
        )

    revenue_growth = None
    revenue_growth_pct = None
    if average_historical > 0:
        revenue_growth = round(
            total_forecast - (average_historical * horizon),
            2
        )
        revenue_growth_pct = round(
            ((average_forecast - average_historical) / average_historical) * 100,
            2
        )

    calculated_reorder_point = reorder_point
    if calculated_reorder_point is None and average_forecast > 0:
        calculated_reorder_point = round(
            (average_forecast * lead_time_weeks) + (safety_stock or 0.0),
            2
        )

    kpis = {
        "forecast_horizon_weeks": horizon,
        "average_forecast": round(average_forecast, 2),
        "total_forecast": round(total_forecast, 2),
        "revenue_growth": revenue_growth,
        "revenue_growth_pct": revenue_growth_pct,
        "current_inventory": current_inventory,
        "inventory_gap": inventory_gap,
        "coverage_weeks": coverage_weeks,
        "coverage_days": coverage_days,
        "lead_time_weeks": round(lead_time_weeks, 4),
        "lead_time_days": round(lead_time_weeks * 7, 1),
        "safety_stock": safety_stock,
        "reorder_point": calculated_reorder_point,
        "fill_rate": fill_rate,
        "service_level": round(service_level, 4),
        "stockout_probability": stockout_probability,
        "holding_cost_estimate": _holding_cost_estimate(
            current_inventory=current_inventory,
            holding_cost=holding_cost,
            horizon=horizon
        ),
        "inventory_turnover_proxy": _inventory_turnover_proxy(
            total_forecast=total_forecast,
            current_inventory=current_inventory
        )
    }

    rankings = _build_rankings(
        context=context
    )
    scenario = _extract_what_if_scenario(
        question=question,
        context=context,
        average_forecast=average_forecast,
        total_forecast=total_forecast,
        current_inventory=current_inventory,
        lead_time_weeks=lead_time_weeks
    )
    if scenario:
        _add_scenario_metrics(
            scenario=scenario,
            current_inventory=current_inventory,
            lead_time_weeks=lead_time_weeks
        )

    response_mode = _classify_response_mode(
        question=question,
        intent=intent
    )

    return {
        "intent": intent,
        "response_mode": response_mode,
        "contexts": contexts,
        "kpis": kpis,
        "rankings": rankings,
        "scenario": scenario,
        "question_inputs": question_inputs,
        "recommendation": _build_recommendation(
            intent=intent,
            kpis=kpis,
            risk=contexts["inventory_metrics"].get("risk", {}),
            scenario=scenario,
            lead_time_weeks=lead_time_weeks
        ),
        "confidence": _confidence_label(
            forecast_values=forecast_values,
            inventory_available=current_inventory is not None,
            intent=intent
        )
    }


def separate_business_contexts(
    question: str,
    context: dict[str, Any]
) -> dict[str, Any]:
    """
    Build isolated context groups using the required priority order.

    Current user inputs override dashboard values. When the user provides
    a complete inventory scenario, unrelated revenue and historical metrics
    are excluded to prevent context contamination.
    """

    question_inputs = _extract_question_inputs(
        question
    )
    inventory = context.get("inventory", {})
    risk = context.get("risk", {})
    risk_metrics = risk.get("metrics", {})
    has_fresh_inventory_scenario = (
        "current_inventory" in question_inputs
        and "forecast_demand" in question_inputs
    )

    forecast_metrics = {
        "horizon": 1 if has_fresh_inventory_scenario else context.get("horizon"),
        "average_forecast": question_inputs.get(
            "forecast_demand",
            context.get("average_forecast")
        ),
        "total_forecast": question_inputs.get(
            "forecast_demand",
            context.get("total_forecast")
        ),
        "forecast_values": (
            [question_inputs["forecast_demand"]]
            if "forecast_demand" in question_inputs
            else context.get("forecast_values", [])
        )
    }
    if not has_fresh_inventory_scenario:
        forecast_metrics["average_historical"] = context.get("average_historical")
        forecast_metrics["trend_direction"] = context.get("trend_direction")
        forecast_metrics["change_pct"] = context.get("change_pct")

    inventory_metrics = {
        "current_inventory": question_inputs.get(
            "current_inventory",
            context.get("current_inventory", risk_metrics.get("current_inventory"))
        ),
        "lead_time_weeks": question_inputs.get(
            "lead_time_weeks",
            inventory.get("parameters", {}).get(
                "lead_time_weeks",
                context.get("lead_time_weeks", 2.0)
            )
        ),
        "service_level": inventory.get("parameters", {}).get(
            "service_level",
            context.get("service_level", 0.95)
        ),
        "holding_cost_unit_year": inventory.get("parameters", {}).get(
            "holding_cost_unit_year",
            context.get("holding_cost_unit_year", 1.5)
        ),
        "risk": {} if has_fresh_inventory_scenario else risk
    }
    if not has_fresh_inventory_scenario:
        inventory_metrics["safety_stock"] = inventory.get("safety_stock")
        inventory_metrics["reorder_point"] = inventory.get("reorder_point")
        inventory_metrics["economic_order_quantity"] = inventory.get(
            "economic_order_quantity"
        )

    revenue_metrics = {}
    if not has_fresh_inventory_scenario:
        revenue_metrics = {
            "average_historical": context.get("average_historical"),
            "average_forecast": context.get("average_forecast"),
            "total_forecast": context.get("total_forecast"),
            "items": context.get("items", [])
        }

    return {
        "priority_order": [
            "current_user_inputs",
            "forecast_metrics",
            "inventory_metrics",
            "conversation_memory",
            "historical_context"
        ],
        "fresh_scenario": has_fresh_inventory_scenario,
        "current_user_inputs": question_inputs,
        "forecast_metrics": forecast_metrics,
        "inventory_metrics": inventory_metrics,
        "revenue_metrics": revenue_metrics,
        "business_kpis": context.get("business_kpis", {})
    }


def format_analysis_for_prompt(
    analysis: dict[str, Any]
) -> str:
    """
    Convert computed metrics to compact text for LLM grounding.
    """

    lines = [
        "Computed Business Analysis",
        f"- Response Mode: {analysis.get('response_mode', 'business')}"
    ]
    for key, value in analysis.get("kpis", {}).items():
        if value is not None:
            lines.append(
                f"- {key.replace('_', ' ').title()}: {value}"
            )

    recommendation = analysis.get("recommendation", {})
    for key in ["observation", "impact", "risk", "recommendation", "confidence"]:
        value = recommendation.get(key)
        if value:
            lines.append(
                f"- {key.title()}: {value}"
            )

    scenario = analysis.get("scenario")
    if scenario:
        lines.append(
            f"- What If Scenario: {scenario['description']}"
        )
        lines.append(
            f"- Scenario Impact: {scenario['impact']}"
        )
        if scenario.get("inventory_gap") is not None:
            lines.append(
                f"- Scenario Inventory Gap: {scenario['inventory_gap']}"
            )

    question_inputs = analysis.get("question_inputs", {})
    for key, value in question_inputs.items():
        lines.append(
            f"- Current User Input {key.replace('_', ' ').title()}: {value}"
        )

    ranking = analysis.get("rankings", {}).get("top_revenue_contributor")
    if ranking:
        lines.append(
            f"- Top Revenue Contributor: {ranking['name']} "
            f"with {ranking['contribution_pct']}% contribution"
        )

    return "\n".join(lines)


def _classify_response_mode(
    question: str,
    intent: str
) -> str:
    """
    Select direct, calculated, or business-recommendation response style.
    """

    q = question.lower()

    business_terms = [
        "what action",
        "what should",
        "should we",
        "recommend",
        "why",
        "risk",
        "what if",
        "if "
    ]
    if any(term in q for term in business_terms):
        return "business"

    analytical_terms = [
        "gap",
        "calculate",
        "difference",
        "coverage",
        "how many",
        "probability",
        "turnover",
        "holding cost"
    ]
    if any(term in q for term in analytical_terms):
        return "analytical"

    retrieval_terms = [
        "what is",
        "what's",
        "show",
        "give me",
        "tell me"
    ]
    if any(term in q for term in retrieval_terms):
        return "retrieval"

    if intent in {"inventory_action", "inventory_risk", "business_reasoning"}:
        return "business"
    if intent in {"comparison", "growth_analysis", "ranking_analysis"}:
        return "analytical"
    return "retrieval"


def _add_scenario_metrics(
    scenario: dict[str, Any],
    current_inventory: float | None,
    lead_time_weeks: float
) -> None:
    """
    Add validated inventory metrics to a what-if scenario.
    """

    new_total = _number_or_none(
        scenario.get("new_total_forecast")
    )
    if new_total is None or new_total <= 0:
        return

    scenario["lead_time_days"] = round(
        lead_time_weeks * 7,
        1
    )
    scenario["daily_demand"] = round(
        new_total / max(lead_time_weeks * 7, 1),
        2
    )

    if current_inventory is not None:
        scenario["inventory_gap"] = round(
            new_total - current_inventory,
            2
        )
        scenario["coverage_days"] = round(
            current_inventory / scenario["daily_demand"],
            1
        )
        scenario["fill_rate"] = _fill_rate_proxy(
            current_inventory=current_inventory,
            total_forecast=new_total
        )
        scenario["stockout_risk"] = (
            "High"
            if scenario["inventory_gap"] > 0
            else "Low"
        )


def _build_rankings(
    context: dict[str, Any]
) -> dict[str, Any]:
    """
    Rank optional product/category records when provided by the UI or API.
    """

    items = context.get("items", [])
    if not items:
        return {}

    valid_items = []
    total_revenue = 0.0
    for item in items:
        revenue = _number(
            item.get("forecast_revenue", item.get("total_forecast")),
            0.0
        )
        if revenue <= 0:
            continue
        total_revenue += revenue
        valid_items.append(
            {
                "name": str(item.get("name", item.get("id", "Unknown"))),
                "forecast_revenue": revenue,
                "growth_pct": _number(item.get("growth_pct"), 0.0),
            }
        )

    if not valid_items:
        return {}

    by_revenue = sorted(
        valid_items,
        key=lambda item: item["forecast_revenue"],
        reverse=True
    )
    by_growth = sorted(
        valid_items,
        key=lambda item: item["growth_pct"],
        reverse=True
    )
    top_revenue = by_revenue[0]
    top_growth = by_growth[0]

    return {
        "top_revenue_contributor": {
            "name": top_revenue["name"],
            "forecast_revenue": round(top_revenue["forecast_revenue"], 2),
            "contribution_pct": round((top_revenue["forecast_revenue"] / total_revenue) * 100, 2)
        },
        "top_growth_item": {
            "name": top_growth["name"],
            "growth_pct": round(top_growth["growth_pct"], 2)
        }
    }


def _extract_what_if_scenario(
    question: str,
    context: dict[str, Any],
    average_forecast: float,
    total_forecast: float,
    current_inventory: float | None,
    lead_time_weeks: float
) -> dict[str, Any] | None:
    """
    Parse simple what-if scenarios from the user's natural language question.
    """

    q = question.lower()
    if "what if" not in q and not q.strip().startswith("if "):
        return None

    pct_match = re.search(
        r"(demand|sales|inventory|lead time).{0,30}(increase|increases|decrease|decreases|drop|drops|double|doubles|rise|rises|fall|falls)\w*.*?(\d+(?:\.\d+)?)\s*%",
        q
    )
    doubles_lead_time = "lead time" in q and ("double" in q or "doubles" in q)

    if pct_match:
        metric = pct_match.group(1)
        direction = pct_match.group(2)
        pct = float(pct_match.group(3))
        sign = -1 if direction in {"decrease", "drop", "drops", "fall", "falls"} else 1
        multiplier = 1 + (sign * pct / 100)

        if metric in {"demand", "sales"}:
            new_total = round(total_forecast * multiplier, 2)
            scenario_gap = (
                round(new_total - current_inventory, 2)
                if current_inventory is not None
                else None
            )
            return {
                "type": "demand_change",
                "description": f"{metric.title()} changes by {sign * pct:+.1f}%.",
                "impact": f"Projected total demand changes from {total_forecast:,.2f} to {new_total:,.2f}.",
                "new_total_forecast": new_total,
                "incremental_demand": round(new_total - total_forecast, 2),
                "inventory_gap": scenario_gap
            }

        if metric == "inventory" and current_inventory is not None:
            new_inventory = round(current_inventory * multiplier, 2)
            gap = round(total_forecast - new_inventory, 2)
            return {
                "type": "inventory_change",
                "description": f"Inventory changes by {sign * pct:+.1f}%.",
                "impact": f"Inventory moves from {current_inventory:,.2f} to {new_inventory:,.2f}; forecast gap becomes {gap:,.2f}.",
                "new_inventory": new_inventory,
                "inventory_gap": gap
            }

    if doubles_lead_time:
        new_lead_time = round(lead_time_weeks * 2, 2)
        new_lead_time_demand = round(average_forecast * new_lead_time, 2)
        return {
            "type": "lead_time_change",
            "description": f"Lead time doubles from {lead_time_weeks:g} to {new_lead_time:g} weeks.",
            "impact": f"Lead-time demand increases to {new_lead_time_demand:,.2f}, so reorder planning must be raised.",
            "new_lead_time_weeks": new_lead_time,
            "lead_time_demand": new_lead_time_demand
        }

    return None


def _build_recommendation(
    intent: str,
    kpis: dict[str, Any],
    risk: dict[str, Any],
    scenario: dict[str, Any] | None,
    lead_time_weeks: float
) -> dict[str, str]:
    """
    Build an explainable recommendation structure.
    """

    growth_pct = kpis.get("revenue_growth_pct")
    stockout = risk.get("stockout_risk", {}).get("level", "Not calculated")
    inventory_gap = kpis.get("inventory_gap")
    coverage_days = kpis.get("coverage_days")

    observation = (
        f"Forecasted demand is {growth_pct:+.1f}% versus the historical baseline."
        if growth_pct is not None
        else "The current scenario provides forecast and inventory values without a historical comparison."
    )
    if scenario:
        observation = scenario["description"]
        if scenario.get("inventory_gap") is not None:
            inventory_gap = scenario["inventory_gap"]
        if scenario.get("coverage_days") is not None:
            coverage_days = scenario["coverage_days"]
        if scenario.get("stockout_risk"):
            stockout = scenario["stockout_risk"]

    impact = "Revenue and replenishment plans should be aligned with the forecast."
    if inventory_gap is not None:
        if inventory_gap > 0:
            impact = f"Current inventory is short of forecasted demand by {inventory_gap:,.2f}."
        else:
            impact = f"Current inventory covers forecasted demand with a surplus of {abs(inventory_gap):,.2f}."

    if stockout == "Not calculated" and inventory_gap is not None:
        stockout = "High" if inventory_gap > 0 else "Low"

    risk_text = f"Stockout risk is {stockout}."
    if coverage_days is not None:
        risk_text = f"Inventory coverage is about {coverage_days:g} days and stockout risk is {stockout}."

    if scenario and scenario.get("inventory_gap") is not None and scenario["inventory_gap"] > 0:
        action = (
            f"Place a replenishment order for at least {scenario['inventory_gap']:,.0f} units "
            f"before the {round(lead_time_weeks * 7):g}-day lead-time window, then add safety stock if service-level targets require it."
        )
    elif intent in {"inventory_action", "inventory_risk"}:
        if inventory_gap is not None and inventory_gap > 0:
            action = "Place a replenishment order and review the reorder point before the demand window."
        elif inventory_gap is not None:
            action = "Avoid aggressive purchasing and monitor sell-through to prevent excess inventory."
        else:
            action = "Calculate current inventory coverage before changing replenishment quantities."
    elif scenario:
        action = "Use the scenario impact to update replenishment and staffing plans before execution."
    else:
        action = "Track the forecast weekly and align inventory, staffing, and promotions to the trend."

    return {
        "observation": observation,
        "impact": impact,
        "risk": risk_text,
        "recommendation": action,
        "confidence": _recommendation_confidence(
            stockout=stockout,
            inventory_gap=inventory_gap,
            scenario=scenario
        )
    }


def _stockout_probability_proxy(
    current_inventory: float,
    reorder_point: float,
    safety_stock: float
) -> float:
    """
    Practical stockout probability proxy for demo decision support.
    """

    if current_inventory <= safety_stock:
        return 0.9
    if current_inventory <= reorder_point:
        return 0.7
    if current_inventory <= reorder_point * 1.25:
        return 0.4
    return 0.15


def _extract_question_inputs(
    question: str
) -> dict[str, float]:
    """
    Extract current user-provided numbers so they override older context.
    """

    q = question.lower()
    inputs: dict[str, float] = {}

    inventory_match = re.search(
        r"(?:current\s+)?inventory\s+(?:is|=|of)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        q
    )
    if inventory_match:
        inputs["current_inventory"] = _number(
            inventory_match.group(1).replace(",", ""),
            0.0
        )

    forecast_match = re.search(
        r"forecast(?:ed)?\s+demand\s+(?:is|=|of)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        q
    )
    if forecast_match:
        inputs["forecast_demand"] = _number(
            forecast_match.group(1).replace(",", ""),
            0.0
        )

    lead_time_match = re.search(
        r"lead\s*time\s+(?:is|=|of)?\s*(\d+(?:\.\d+)?)\s*(day|days|week|weeks)",
        q
    )
    if lead_time_match:
        lead_value = _number(
            lead_time_match.group(1),
            0.0
        )
        unit = lead_time_match.group(2)
        inputs["lead_time_weeks"] = (
            round(lead_value / 7, 4)
            if unit in {"day", "days"}
            else lead_value
        )

    return inputs


def _fill_rate_proxy(
    current_inventory: float,
    total_forecast: float
) -> float:
    """
    Estimate demand coverage as a simple fill-rate proxy.
    """

    if total_forecast <= 0:
        return 1.0
    return round(
        min(current_inventory / total_forecast, 1.0),
        4
    )


def _holding_cost_estimate(
    current_inventory: float | None,
    holding_cost: float,
    horizon: int
) -> float | None:
    if current_inventory is None:
        return None
    return round(
        current_inventory * holding_cost * (horizon / 52),
        2
    )


def _inventory_turnover_proxy(
    total_forecast: float,
    current_inventory: float | None
) -> float | None:
    if current_inventory is None or current_inventory <= 0:
        return None
    return round(
        total_forecast / current_inventory,
        3
    )


def _confidence_label(
    forecast_values: list[float],
    inventory_available: bool,
    intent: str
) -> str:
    if len(forecast_values) >= 4 and inventory_available:
        return "High"
    if len(forecast_values) >= 2 or intent not in {"inventory_action", "inventory_risk"}:
        return "Medium"
    return "Low"


def _recommendation_confidence(
    stockout: str,
    inventory_gap: float | None,
    scenario: dict[str, Any] | None
) -> str:
    if scenario:
        return "Medium"
    if inventory_gap is not None and stockout in {"High", "Critical", "Medium"}:
        return "High"
    if inventory_gap is not None:
        return "Medium"
    return "Low"


def _mean(
    values: list[float]
) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _number(
    value: Any,
    default: float
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _number_or_none(
    value: Any
) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_number(
    value: Any
) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
