"""
risk.py

Risk classification engine for stock-out and overstock risk assessments.


"""

from typing import Dict, Any


def classify_risk(
    current_inventory: float,
    reorder_point: float,
    safety_stock: float,
    total_forecasted_demand: float
) -> Dict[str, Any]:
    """
    Classifies stock-out and overstock risks based on current inventory levels,
    safety stock, reorder point, and forecasted demand.
    """
    # 1. Stock-out Risk
    if current_inventory < safety_stock:
        stockout_risk = "Critical"
        stockout_desc = "Current stock is below safety stock limits. Immediate reorder required."
    elif current_inventory < reorder_point:
        stockout_risk = "High"
        stockout_desc = "Current stock is below the reorder point. Trigger restocking order."
    elif current_inventory < reorder_point * 1.25:
        stockout_risk = "Medium"
        stockout_desc = "Current stock is slightly above ROP. Monitor depletion rates closely."
    else:
        stockout_risk = "Low"
        stockout_desc = "Sufficient inventory buffer. Stockout risk is minimal."

    # 2. Overstock Risk
    maximum_recommended_level = total_forecasted_demand + safety_stock
    
    if current_inventory > maximum_recommended_level * 1.3:
        overstock_risk = "High"
        overstock_desc = "Inventory levels exceed maximum forecasted demand and safety buffer by 30%. High holding costs."
    elif current_inventory > maximum_recommended_level:
        overstock_risk = "Medium"
        overstock_desc = "Inventory levels exceed recommended forecast buffer. Hold off on additional orders."
    else:
        overstock_risk = "Low"
        overstock_desc = "Inventory levels are well aligned with projected demand."

    return {
        "stockout_risk": {
            "level": stockout_risk,
            "description": stockout_desc
        },
        "overstock_risk": {
            "level": overstock_risk,
            "description": overstock_desc
        },
        "metrics": {
            "current_inventory": current_inventory,
            "reorder_point": reorder_point,
            "safety_stock": safety_stock,
            "total_forecasted_demand": total_forecasted_demand,
            "max_recommended_stock": round(maximum_recommended_level, 2)
        }
    }
