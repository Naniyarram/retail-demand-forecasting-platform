"""
conversational_assistant.py

Conversational retail analytics assistant for RetailCast.

This module turns forecast, inventory, and risk metrics into natural
language answers. It keeps the design intentionally simple:

1. Detect the user's business intent.
2. Add recent conversation history for follow-up questions.
3. Ask the LLM for a grounded answer.
4. Validate the answer and fall back to deterministic business logic
   if the LLM is unavailable.
"""

from __future__ import annotations

import re
from typing import Any

from pipeline.utils.business_analytics import (
    build_business_analysis,
    format_analysis_for_prompt
)
from pipeline.utils.llm_client import HFLLMClient


class ConversationalRetailAssistant:
    """
    Answer business questions about RetailCast forecast and inventory data.
    """

    MAX_MEMORY_MESSAGES = 8

    def __init__(
        self,
        llm_client: HFLLMClient | None = None
    ):
        self.llm_client = llm_client or HFLLMClient()

    def answer_question(
        self,
        question: str,
        business_context: dict[str, Any],
        conversation_history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        """
        Generate a contextual answer for a business user's question.
        """

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        history = self._clean_history(
            conversation_history or []
        )
        intent = self._detect_intent(
            clean_question
        )
        subject = self._resolve_subject(
            question=clean_question,
            context=business_context,
            history=history
        )
        analysis = build_business_analysis(
            question=clean_question,
            context=business_context,
            intent=intent
        )

        if analysis["response_mode"] in {"retrieval", "analytical"}:
            answer = self._build_direct_answer(
                question=clean_question,
                subject=subject,
                analysis=analysis
            )
            updated_history = self._append_history(
                history=history,
                question=clean_question,
                answer=answer
            )
            return {
                "answer": answer,
                "verified": True,
                "model_used": "Business Calculation Engine",
                "detected_intent": intent,
                "referenced_entity": subject,
                "analysis": analysis,
                "conversation_history": updated_history
            }

        prompt = self._build_prompt(
            question=clean_question,
            context=business_context,
            history=history,
            intent=intent,
            subject=subject,
            analysis=analysis
        )

        try:
            raw_answer = self.llm_client.generate_text(
                prompt=prompt,
                system_prompt=(
                    "You are RetailCast's conversational retail analytics "
                    "assistant. Programmatic calculations in the prompt are "
                    "authoritative. Copy those calculated values exactly and "
                    "do not recalculate them. Answer only from the current "
                    "prioritized context. If data is unavailable, say what is "
                    "missing and give the next best analytical step."
                ),
                max_tokens=450,
                temperature=0.35
            )
            answer = self.llm_client._clean_response(
                text=raw_answer,
                prompt=prompt
            )
            verified = self._verify_answer(
                answer=answer,
                question=clean_question,
                context=business_context,
                analysis=analysis
            )
            model_used = self.llm_client.model_name
        except Exception:
            answer = ""
            verified = False
            model_used = "Rule-Based Conversational Fallback"

        if not verified:
            answer = self._fallback_answer(
                question=clean_question,
                context=business_context,
                analysis=analysis,
                intent=intent,
                subject=subject
            )
            model_used = "Rule-Based Conversational Fallback"

        updated_history = self._append_history(
            history=history,
            question=clean_question,
            answer=answer
        )

        return {
            "answer": answer,
            "verified": verified,
            "model_used": model_used,
            "detected_intent": intent,
            "referenced_entity": subject,
            "analysis": analysis,
            "conversation_history": updated_history
        }

    def _clean_history(
        self,
        history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Keep only recent user/assistant messages with plain text content.
        """

        cleaned = []
        for message in history[-self.MAX_MEMORY_MESSAGES:]:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                cleaned.append(
                    {
                        "role": role,
                        "content": content
                    }
                )
        return cleaned

    def _detect_intent(
        self,
        question: str
    ) -> str:
        """
        Map a natural language question to a simple business intent.
        """

        q = question.lower()

        if any(word in q for word in ["stockout", "out of stock", "risk"]):
            return "inventory_risk"
        if any(word in q for word in ["inventory", "stock", "reorder", "safety"]):
            return "inventory_action"
        if any(word in q for word in ["grow", "growth", "increase", "decline", "decrease"]):
            return "growth_analysis"
        if any(word in q for word in ["top", "highest", "best", "contribute"]):
            return "ranking_analysis"
        if any(word in q for word in ["compare", "last month", "previous"]):
            return "comparison"
        if any(word in q for word in ["why", "reason", "driver"]):
            return "business_reasoning"

        return "general_forecast"

    def _resolve_subject(
        self,
        question: str,
        context: dict[str, Any],
        history: list[dict[str, str]]
    ) -> str:
        """
        Resolve pronouns such as "it" to the active store/department context.
        """

        explicit_product = re.search(
            r"\b(product|sku|department)\s+([a-zA-Z0-9_-]+)",
            question,
            flags=re.IGNORECASE
        )
        if explicit_product:
            return f"{explicit_product.group(1).title()} {explicit_product.group(2)}"

        q = question.lower()
        if any(token in q for token in ["it", "that", "this", "same product"]):
            for message in reversed(history):
                match = re.search(
                    r"\b(Product|SKU|Department)\s+([a-zA-Z0-9_-]+)",
                    message["content"],
                    flags=re.IGNORECASE
                )
                if match:
                    return f"{match.group(1).title()} {match.group(2)}"

        store_id = context.get("store_id", "selected")
        department_id = context.get("department_id")
        if department_id:
            return f"Store {store_id}, Department {department_id}"
        return f"Store {store_id}, all departments"

    def _build_prompt(
        self,
        question: str,
        context: dict[str, Any],
        history: list[dict[str, str]],
        intent: str,
        subject: str,
        analysis: dict[str, Any]
    ) -> str:
        """
        Build the grounded prompt sent to the LLM.
        """

        history_text = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in history[-6:]
        ) or "No previous conversation."

        analysis_text = format_analysis_for_prompt(
            analysis
        )
        contexts = analysis.get("contexts", {})
        fresh_scenario = contexts.get("fresh_scenario", False)
        forecast_metrics = contexts.get("forecast_metrics", {})
        inventory_metrics = contexts.get("inventory_metrics", {})
        revenue_metrics = contexts.get("revenue_metrics", {})
        scenario = analysis.get("scenario") or {}
        authoritative_facts = self._authoritative_facts(
            analysis
        )

        return (
            "RetailCast Prioritized Business Context\n"
            f"- Active subject: {subject}\n"
            f"- User intent: {intent}\n"
            f"- Fresh user scenario: {fresh_scenario}\n"
            f"- Current user inputs: {contexts.get('current_user_inputs', {})}\n"
            f"- Forecast metrics: {forecast_metrics}\n"
            f"- Inventory metrics: {inventory_metrics}\n"
            f"- Revenue metrics: {revenue_metrics}\n\n"
            "Authoritative Calculated Facts\n"
            f"{authoritative_facts}\n\n"
            f"{analysis_text}\n\n"
            "Recent conversation\n"
            f"{history_text}\n\n"
            f"User question: {question}\n\n"
            "Answer requirements:\n"
            "- Prioritize current user inputs and computed business analysis before conversation memory.\n"
            "- Ignore conversation values that conflict with current user inputs.\n"
            "- Do not mention revenue or historical growth for a fresh inventory scenario unless the user asks for them.\n"
            "- Use the post-scenario demand and post-scenario inventory gap, not the base demand gap.\n"
            "- Copy every authoritative calculated fact exactly; do not replace it with your own calculation.\n"
            "- Answer using this structure: Observation, Impact, Risk, Recommendation, Confidence.\n"
            "- Use the forecast and inventory numbers when relevant.\n"
            "- Explain the business reason, not only the metric.\n"
            "- Include one clear recommended action when the question asks for action.\n"
            "- Treat reorder point, safety stock, and EOQ as planning parameters, "
            "not as current inventory.\n"
            "- Do not compare current inventory directly against EOQ; EOQ is an "
            "order-size recommendation, not a stock-level target.\n"
            "- Do not recommend exact inventory increase quantities or percentages "
            "unless they are directly provided in the context.\n"
            "- If the user asks a what-if question, answer using the scenario impact above.\n"
            "- Do not invent product-level data that is not in the context."
        )

    def _authoritative_facts(
        self,
        analysis: dict[str, Any]
    ) -> str:
        """
        Highlight calculations the LLM must copy without modification.
        """

        scenario = analysis.get("scenario") or {}
        kpis = analysis.get("kpis", {})
        facts = []

        if scenario:
            mappings = [
                ("Post-scenario demand", scenario.get("new_total_forecast")),
                ("Post-scenario inventory gap", scenario.get("inventory_gap")),
                ("Post-scenario coverage days", scenario.get("coverage_days")),
                ("Lead time days", scenario.get("lead_time_days")),
                ("Post-scenario stockout risk", scenario.get("stockout_risk")),
            ]
        else:
            mappings = [
                ("Forecast demand", kpis.get("total_forecast")),
                ("Inventory gap", kpis.get("inventory_gap")),
                ("Coverage days", kpis.get("coverage_days")),
                ("Lead time days", kpis.get("lead_time_days")),
                ("Stockout probability", kpis.get("stockout_probability")),
            ]

        for label, value in mappings:
            if value is not None:
                facts.append(
                    f"- {label}: {value}"
                )

        return "\n".join(facts) or "- No authoritative calculation available."

    def _verify_answer(
        self,
        answer: str,
        question: str,
        context: dict[str, Any],
        analysis: dict[str, Any] | None = None
    ) -> bool:
        """
        Validate that the LLM answer is useful and grounded enough for the demo.
        """

        if not answer or len(answer.strip()) < 80:
            return False

        lower_answer = answer.lower()
        if not any(term in lower_answer for term in ["sales", "demand", "forecast", "inventory", "stock"]):
            return False

        if analysis and not self._confidence_is_consistent(
            answer=answer,
            analysis=analysis
        ):
            return False

        if analysis and not self._required_calculations_are_present(
            answer=answer,
            analysis=analysis
        ):
            return False

        if (
            analysis
            and analysis.get("contexts", {}).get("fresh_scenario")
            and any(term in lower_answer for term in ["revenue", "historical growth"])
            and not any(term in question.lower() for term in ["revenue", "historical"])
        ):
            return False

        if any(term in question.lower() for term in ["inventory", "stock", "reorder"]):
            if not any(term in lower_answer for term in ["inventory", "stock", "reorder", "safety"]):
                return False
            if (
                "current inventory" in lower_answer
                and context.get("risk", {}).get("metrics", {}).get("current_inventory") is None
            ):
                return False
            if self._compares_current_inventory_to_eoq(
                lower_answer
            ):
                return False
            if self._has_unsupported_inventory_quantity(
                answer=answer,
                context=context
            ):
                return False

        forecast_mean = context.get("average_forecast")
        if forecast_mean is not None:
            rounded_value = str(round(float(forecast_mean)))
            has_number = rounded_value in answer.replace(",", "")
            has_context_word = any(term in lower_answer for term in ["forecast", "projected", "expected"])
            return has_number or has_context_word

        return True

    def _required_calculations_are_present(
        self,
        answer: str,
        analysis: dict[str, Any]
    ) -> bool:
        """
        Confirm that critical programmatic calculations appear in the answer.
        """

        scenario = analysis.get("scenario") or {}
        required_values = []
        for key in [
            "new_total_forecast",
            "inventory_gap",
            "lead_time_days"
        ]:
            value = scenario.get(key)
            if value is not None:
                required_values.append(
                    float(value)
                )

        if not required_values:
            return True

        normalized_answer = answer.replace(",", "")
        for value in required_values:
            displays = {
                str(round(value, 2)),
                str(round(value, 1)),
                str(round(value))
            }
            if not any(display in normalized_answer for display in displays):
                return False

        return True

    def _build_direct_answer(
        self,
        question: str,
        subject: str,
        analysis: dict[str, Any]
    ) -> str:
        """
        Return direct, deterministic answers for retrieval and calculations.
        """

        q = question.lower()
        kpis = analysis.get("kpis", {})
        rankings = analysis.get("rankings", {})

        if "gap" in q or "difference" in q:
            gap = kpis.get("inventory_gap")
            if gap is None:
                return (
                    "The inventory gap cannot be calculated because current "
                    "inventory or forecast demand is missing."
                )
            direction = "shortfall" if gap > 0 else "surplus"
            return (
                f"The inventory {direction} for {subject} is "
                f"{abs(gap):,.2f} units. "
                "Calculation: forecast demand minus current inventory."
            )

        if "coverage" in q:
            coverage_days = kpis.get("coverage_days")
            if coverage_days is None:
                return (
                    "Inventory coverage cannot be calculated because current "
                    "inventory or forecast demand is missing."
                )
            return (
                f"{subject} has approximately {coverage_days:g} days of "
                "inventory coverage at the current forecast demand rate."
            )

        if any(term in q for term in ["contribute", "top", "highest"]):
            contributor = rankings.get("top_revenue_contributor")
            if not contributor:
                return "Revenue contribution cannot be ranked because item-level data is unavailable."
            return (
                f"{contributor['name']} contributes the most forecasted revenue "
                f"at {contributor['contribution_pct']}%, representing "
                f"${contributor['forecast_revenue']:,.2f}."
            )

        if "forecast" in q or "demand" in q:
            total_forecast = kpis.get("total_forecast")
            return (
                f"The forecasted demand for {subject} is "
                f"{total_forecast:,.2f} units across the selected horizon."
            )

        return self._format_recommendation(
            observation=analysis["recommendation"]["observation"],
            impact=analysis["recommendation"]["impact"],
            risk=analysis["recommendation"]["risk"],
            recommendation=analysis["recommendation"]["recommendation"],
            confidence=analysis["recommendation"]["confidence"]
        )

    def _confidence_is_consistent(
        self,
        answer: str,
        analysis: dict[str, Any]
    ) -> bool:
        """
        Reject answers that overstate computed confidence.
        """

        expected = str(
            analysis.get("recommendation", {}).get(
                "confidence",
                analysis.get("confidence", "")
            )
        ).lower()
        if not expected:
            return True

        match = re.search(
            r"confidence\s*[:\-]\s*(high|medium|low)",
            answer,
            flags=re.IGNORECASE
        )
        if not match:
            return True

        levels = {
            "low": 1,
            "medium": 2,
            "high": 3
        }
        actual = match.group(1).lower()
        return levels.get(actual, 0) <= levels.get(expected, 0)

    def _compares_current_inventory_to_eoq(
        self,
        lower_answer: str
    ) -> bool:
        """
        Reject answers that compare stock level directly to EOQ.
        """

        comparison_words = (
            "above",
            "below",
            "greater",
            "less",
            "higher",
            "lower",
            "optimal quantity",
            "optimal stock"
        )
        compares_inventory_and_eoq = (
            re.search(r"current inventory.{0,100}\beoq\b", lower_answer)
            or re.search(r"\beoq\b.{0,100}current inventory", lower_answer)
        )
        if not compares_inventory_and_eoq:
            return False

        return any(
            word in lower_answer
            for word in comparison_words
        )

    def _has_unsupported_inventory_quantity(
        self,
        answer: str,
        context: dict[str, Any]
    ) -> bool:
        """
        Reject precise inventory increases that are not grounded in context.
        """

        lower_answer = answer.lower()
        precise_action = re.search(
            r"\b(increase|raise|add|reduce|decrease)\b.{0,45}\bby\s+\$?[\d,]+(?:\.\d+)?\s*(units?|%|percent)?",
            lower_answer
        )
        if not precise_action:
            return False

        allowed_values = self._context_numbers(
            context
        )
        answer_numbers = {
            self._normalize_number(match)
            for match in re.findall(r"\$?[\d,]+(?:\.\d+)?", answer)
        }
        unsupported_numbers = {
            number
            for number in answer_numbers
            if number is not None and number not in allowed_values
        }

        return bool(unsupported_numbers)

    def _context_numbers(
        self,
        context: dict[str, Any]
    ) -> set[float]:
        """
        Collect numeric facts that are allowed to appear in the answer.
        """

        values: list[Any] = [
            context.get("horizon"),
            context.get("average_historical"),
            context.get("average_forecast"),
            context.get("total_forecast"),
            context.get("change_pct"),
            *context.get("forecast_values", []),
        ]
        values.extend(
            context.get("inventory", {}).values()
        )
        values.extend(
            context.get("risk", {}).get("metrics", {}).values()
        )

        normalized = set()
        for value in values:
            try:
                normalized.add(
                    round(float(value), 2)
                )
            except (TypeError, ValueError):
                continue

        return normalized

    def _normalize_number(
        self,
        value: str
    ) -> float | None:
        """
        Convert a displayed number to the comparison format.
        """

        try:
            return round(
                float(
                    value.replace("$", "").replace(",", "")
                ),
                2
            )
        except ValueError:
            return None

    def _fallback_answer(
        self,
        question: str,
        context: dict[str, Any],
        analysis: dict[str, Any],
        intent: str,
        subject: str
    ) -> str:
        """
        Deterministic answer used when live LLM generation is unavailable.
        """

        avg_hist = float(context.get("average_historical", 0))
        avg_fc = float(context.get("average_forecast", 0))
        total_fc = float(context.get("total_forecast", 0))
        change_pct = float(context.get("change_pct", 0))
        horizon = context.get("horizon", "selected")
        trend = "increase" if change_pct > 0 else "decline" if change_pct < 0 else "stable pattern"
        recommendation = analysis.get("recommendation", {})
        kpis = analysis.get("kpis", {})
        rankings = analysis.get("rankings", {})
        scenario = analysis.get("scenario")

        if scenario:
            return self._format_recommendation(
                observation=scenario["description"],
                impact=scenario["impact"],
                risk=recommendation.get("risk", "Scenario risk should be reviewed before execution."),
                recommendation=recommendation.get("recommendation", "Update the operating plan using the scenario result."),
                confidence=recommendation.get("confidence", analysis.get("confidence", "Medium"))
            )

        top_contributor = rankings.get("top_revenue_contributor")
        if intent == "ranking_analysis" and top_contributor:
            return self._format_recommendation(
                observation=(
                    f"{top_contributor['name']} is the top revenue contributor "
                    f"with {top_contributor['contribution_pct']}% of forecasted revenue."
                ),
                impact=(
                    f"Its projected revenue is ${top_contributor['forecast_revenue']:,.2f}, "
                    "so changes in this item have the largest effect on total performance."
                ),
                risk="A miss in this contributor can materially affect the forecast plan.",
                recommendation="Prioritize availability, pricing checks, and promotion planning for this contributor.",
                confidence=analysis.get("confidence", "Medium")
            )

        if intent in {"inventory_risk", "inventory_action"}:
            return self._format_recommendation(
                observation=recommendation.get(
                    "observation",
                    f"For {subject}, demand shows a {trend} of {change_pct:+.1f}%."
                ),
                impact=recommendation.get(
                    "impact",
                    f"Average forecasted weekly sales are ${avg_fc:,.2f} versus ${avg_hist:,.2f} historically."
                ),
                risk=recommendation.get(
                    "risk",
                    "Inventory risk is not calculated yet."
                ),
                recommendation=recommendation.get(
                    "recommendation",
                    "Calculate inventory coverage before changing replenishment quantities."
                ),
                confidence=recommendation.get(
                    "confidence",
                    analysis.get("confidence", "Low")
                )
            )

        if intent in {"growth_analysis", "business_reasoning"}:
            return self._format_recommendation(
                observation=(
                    f"{subject} is expected to show a {trend} of {change_pct:+.1f}% "
                    f"over the next {horizon} weeks."
                ),
                impact=(
                    f"Forecasted weekly sales are ${avg_fc:,.2f}, compared with "
                    f"${avg_hist:,.2f} historically; projected revenue is ${total_fc:,.2f}."
                ),
                risk=recommendation.get("risk", "Demand changes may affect service levels and inventory planning."),
                recommendation=recommendation.get(
                    "recommendation",
                    "Align inventory, staffing, and promotions with the expected demand pattern."
                ),
                confidence=analysis.get("confidence", "Medium")
            )

        if intent == "comparison":
            return self._format_recommendation(
                observation=(
                    f"Compared with the historical weekly average of ${avg_hist:,.2f}, "
                    f"the forecast expects ${avg_fc:,.2f} per week."
                ),
                impact=(
                    f"This is a {change_pct:+.1f}% change, producing ${total_fc:,.2f} "
                    "across the selected horizon."
                ),
                risk=recommendation.get("risk", "Planning risk depends on inventory coverage and forecast variance."),
                recommendation="Use the gap to decide whether replenishment should increase, decrease, or stay steady.",
                confidence=analysis.get("confidence", "Medium")
            )

        return self._format_recommendation(
            observation=(
                f"For {subject}, RetailCast forecasts average weekly sales of ${avg_fc:,.2f} "
                f"over the next {horizon} weeks."
            ),
            impact=(
                f"Total forecasted revenue is ${total_fc:,.2f}, a {change_pct:+.1f}% "
                "change versus the historical baseline."
            ),
            risk=recommendation.get("risk", "Operational risk depends on inventory availability and demand variance."),
            recommendation=recommendation.get(
                "recommendation",
                "Align stock levels, staffing, and promotions with the expected demand pattern."
            ),
            confidence=analysis.get("confidence", "Medium")
        )

    def _format_recommendation(
        self,
        observation: str,
        impact: str,
        risk: str,
        recommendation: str,
        confidence: str
    ) -> str:
        """
        Format deterministic assistant output for business users.
        """

        return (
            f"**Observation:** {observation}\n\n"
            f"**Impact:** {impact}\n\n"
            f"**Risk:** {risk}\n\n"
            f"**Recommendation:** {recommendation}\n\n"
            f"**Confidence:** {confidence}"
        )

    def _append_history(
        self,
        history: list[dict[str, str]],
        question: str,
        answer: str
    ) -> list[dict[str, str]]:
        """
        Add the latest turn and keep only recent messages.
        """

        updated = [
            *history,
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        return updated[-self.MAX_MEMORY_MESSAGES:]
