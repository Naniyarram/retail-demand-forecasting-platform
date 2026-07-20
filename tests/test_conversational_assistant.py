"""
test_conversational_assistant.py

Tests for RetailCast's conversational analytics assistant.
"""

import pytest

from pipeline.utils.conversational_assistant import ConversationalRetailAssistant


class DummyLLMClient:
    model_name = "test-llm"

    def __init__(self, answer):
        self.answer = answer

    def generate_text(self, prompt, system_prompt, max_tokens=450, temperature=0.35):
        return self.answer

    def _clean_response(self, text, prompt):
        return text.strip()


@pytest.fixture
def business_context():
    return {
        "store_id": 1,
        "department_id": 1,
        "horizon": 12,
        "average_historical": 12000.0,
        "average_forecast": 13500.0,
        "total_forecast": 162000.0,
        "trend_direction": "Increase of +12.5%",
        "change_pct": 12.5,
        "forecast_values": [13000.0, 13200.0, 13500.0, 13900.0],
        "inventory": {
            "safety_stock": 4000.0,
            "reorder_point": 31000.0,
            "economic_order_quantity": 18000.0,
            "parameters": {
                "lead_time_weeks": 2.0,
                "service_level": 0.95,
                "holding_cost_unit_year": 1.5
            }
        },
        "risk": {
            "stockout_risk": {
                "level": "Medium",
                "description": "Inventory is close to the reorder point."
            },
            "overstock_risk": {
                "level": "Low",
                "description": "No overstock signal."
            },
            "metrics": {
                "current_inventory": 1000.0,
                "reorder_point": 31000.0,
                "safety_stock": 4000.0,
                "total_forecasted_demand": 162000.0
            }
        },
        "items": [
            {
                "name": "Department 1",
                "forecast_revenue": 162000.0,
                "growth_pct": 12.5
            },
            {
                "name": "Department 2",
                "forecast_revenue": 90000.0,
                "growth_pct": 4.0
            }
        ]
    }


def test_conversational_assistant_returns_verified_llm_answer(business_context):
    answer = (
        "Forecasted demand is increasing for Department 1, with projected weekly "
        "sales of about 13,500 versus 12,000 historically. The inventory plan "
        "should account for this growth by reviewing safety stock and reorder "
        "point levels before the next demand window."
    )
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient(answer)
    )

    result = assistant.answer_question(
        question="Should we increase inventory?",
        business_context=business_context,
        conversation_history=[]
    )

    assert result["verified"] is True
    assert result["model_used"] == "test-llm"
    assert result["detected_intent"] == "inventory_action"
    assert len(result["conversation_history"]) == 2


def test_conversational_assistant_resolves_follow_up_context(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )
    history = [
        {
            "role": "user",
            "content": "How will Department 1 perform next month?"
        },
        {
            "role": "assistant",
            "content": "Department 1 is expected to grow."
        },
    ]

    result = assistant.answer_question(
        question="Why is it increasing?",
        business_context=business_context,
        conversation_history=history
    )

    assert result["verified"] is False
    assert result["referenced_entity"] == "Department 1"
    assert "Department 1" in result["answer"]
    assert "12.5%" in result["answer"]


def test_conversational_assistant_fallback_handles_stockout_question(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )

    result = assistant.answer_question(
        question="Which products may experience stockouts?",
        business_context=business_context,
        conversation_history=[]
    )

    assert result["verified"] is False
    assert result["detected_intent"] == "inventory_risk"
    assert "stockout risk is Medium" in result["answer"]
    assert "reorder point" in result["answer"].lower()


def test_conversational_assistant_rejects_eoq_stock_level_comparison(business_context):
    answer = (
        "The forecasted sales are increasing, so inventory should be monitored. "
        "The current inventory level is above the EOQ, meaning the store has "
        "more than the optimal stock quantity. Reorder point and safety stock "
        "should be reviewed against demand."
    )
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient(answer)
    )

    result = assistant.answer_question(
        question="Should we increase inventory?",
        business_context=business_context,
        conversation_history=[]
    )

    assert result["verified"] is False
    assert result["model_used"] == "Rule-Based Conversational Fallback"


def test_conversational_assistant_calculates_inventory_gap(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )

    result = assistant.answer_question(
        question="Inventory is 1000 units and forecast demand is 1400 units. What is the gap?",
        business_context={
            **business_context,
            "average_forecast": 1400.0,
            "total_forecast": 1400.0,
            "forecast_values": [1400.0],
            "risk": {
                **business_context["risk"],
                "metrics": {
                    **business_context["risk"]["metrics"],
                    "current_inventory": 1000.0,
                    "total_forecasted_demand": 1400.0
                }
            }
        },
        conversation_history=[
            {
                "role": "assistant",
                "content": "Earlier demand was much lower."
            }
        ]
    )

    assert result["analysis"]["kpis"]["inventory_gap"] == 400.0
    assert "400.00" in result["answer"]
    assert result["analysis"]["response_mode"] == "analytical"
    assert result["model_used"] == "Business Calculation Engine"
    assert "forecast demand minus current inventory" in result["answer"]


def test_conversational_assistant_handles_what_if_demand_increase(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )

    result = assistant.answer_question(
        question="What if demand increases by 30%?",
        business_context=business_context,
        conversation_history=[]
    )

    assert result["analysis"]["scenario"]["type"] == "demand_change"
    assert result["analysis"]["scenario"]["new_total_forecast"] == 210600.0
    assert "210,600.00" in result["answer"]


def test_conversational_assistant_ranks_revenue_contribution(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )

    result = assistant.answer_question(
        question="Which category contributes the most to forecasted revenue?",
        business_context=business_context,
        conversation_history=[]
    )

    contributor = result["analysis"]["rankings"]["top_revenue_contributor"]
    assert contributor["name"] == "Department 1"
    assert contributor["contribution_pct"] == 64.29
    assert "Department 1" in result["answer"]


def test_conversational_assistant_rejects_overstated_confidence(business_context):
    answer = (
        "Observation: Demand changes by 30%. Impact: Forecast demand increases, "
        "so inventory planning should be updated. Risk: Stockout risk may rise. "
        "Recommendation: Update replenishment planning. Confidence: High"
    )
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient(answer)
    )

    result = assistant.answer_question(
        question="What if demand increases by 30%?",
        business_context=business_context,
        conversation_history=[]
    )

    assert result["verified"] is False
    assert result["model_used"] == "Rule-Based Conversational Fallback"
    assert "**Confidence:** Medium" in result["answer"]


def test_conversational_assistant_uses_current_question_numbers(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )

    result = assistant.answer_question(
        question=(
            "If Product A demand increases by 20%, current inventory is 1000 units, "
            "forecasted demand is 1400 units, and lead time is 7 days, what action should we take?"
        ),
        business_context=business_context,
        conversation_history=[]
    )

    assert result["referenced_entity"] == "Product A"
    assert result["analysis"]["question_inputs"]["current_inventory"] == 1000.0
    assert result["analysis"]["question_inputs"]["forecast_demand"] == 1400.0
    assert result["analysis"]["question_inputs"]["lead_time_weeks"] == 1.0
    assert result["analysis"]["scenario"]["new_total_forecast"] == 1680.0
    assert result["analysis"]["scenario"]["inventory_gap"] == 680.0
    assert "680" in result["answer"]
    assert "7-day" in result["answer"]


def test_fresh_scenario_excludes_stale_revenue_and_inventory_context(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )

    result = assistant.answer_question(
        question=(
            "If Product A demand increases by 20%, current inventory is 1000 units, "
            "forecasted demand is 1400 units, and lead time is 7 days, what action should we take?"
        ),
        business_context={
            **business_context,
            "average_historical": 999999.0,
            "total_forecast": 888888.0,
            "current_inventory": 70000.0
        },
        conversation_history=[
            {
                "role": "assistant",
                "content": "Previous revenue was $9 million and growth was 80%."
            }
        ]
    )

    contexts = result["analysis"]["contexts"]
    assert contexts["fresh_scenario"] is True
    assert contexts["revenue_metrics"] == {}
    assert contexts["forecast_metrics"]["total_forecast"] == 1400.0
    assert contexts["inventory_metrics"]["current_inventory"] == 1000.0
    assert "999,999" not in result["answer"]
    assert "9 million" not in result["answer"]


def test_retrieval_question_returns_direct_forecast_answer(business_context):
    assistant = ConversationalRetailAssistant(
        llm_client=DummyLLMClient("")
    )

    result = assistant.answer_question(
        question="What is the forecasted demand?",
        business_context=business_context,
        conversation_history=[]
    )

    assert result["analysis"]["response_mode"] == "retrieval"
    assert result["model_used"] == "Business Calculation Engine"
    assert result["verified"] is True
    assert "162,000.00" in result["answer"]
