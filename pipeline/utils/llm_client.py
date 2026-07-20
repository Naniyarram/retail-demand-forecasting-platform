"""
llm_client.py

Hugging Face LLM client for retail business insights.

The client tries Hugging Face's OpenAI-compatible router first,
then falls back to the classic text-generation inference endpoint.
If the network, token, or model is unavailable, it returns a clear
rule-based backup report instead of crashing the app.

"""

import os
from pathlib import Path
from typing import Any

import requests


def _load_dotenv() -> None:
    """Pull HF_API_TOKEN from a local .env when it is not already set."""
    if os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        return

    # Try Streamlit Cloud secrets first (works when running on Streamlit Cloud)
    try:
        import streamlit as st
        token = st.secrets.get("HF_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        if token:
            os.environ["HF_API_TOKEN"] = token
            return
    except Exception:
        # Not running inside Streamlit, or secrets not configured — that's fine
        pass

    # Fall back to a local .env file for local development
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()



class HFLLMClient:
    """
    Generate retail recommendations from forecast metrics.
    """

    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    BACKUP_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

    def __init__(
        self,
        token: str | None = None,
        model_name: str = DEFAULT_MODEL
    ):

        self.token = (
            token
            or os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

        self.model_name = model_name

        self.headers = {
            "Authorization": f"Bearer {self.token}"
        } if self.token else {}

    def generate_retail_insights(
        self,
        forecast_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate retail analysis and inventory recommendations.
        """

        prompt = self._build_prompt(
            forecast_data
        )

        last_error = None

        for model_name in self._model_candidates():

            try:

                raw_text = self._generate_with_router(
                    model_name=model_name,
                    prompt=prompt
                )

                raw_text = self._clean_response(
                    text=raw_text,
                    prompt=prompt
                )

                if self._verify_output(
                    raw_text,
                    forecast_data
                ):

                    return {
                        "raw_insights": raw_text,
                        "verified": True,
                        "model_used": model_name
                    }

                last_error = (
                    "Model response failed validation."
                )

            except Exception as exc:

                last_error = str(exc)

        fallback_text = self._generate_fallback_insights(
            forecast_data,
            reason=last_error or "Hugging Face generation unavailable"
        )

        return {
            "raw_insights": fallback_text,
            "verified": False,
            "model_used": "Rule-Based Analytical Fallback"
        }

    def _model_candidates(self) -> list[str]:
        """
        Return the selected model followed by backup models.
        """

        candidates = [
            self.model_name,
            *self.BACKUP_MODELS
        ]

        unique_candidates = []

        for model_name in candidates:

            if model_name not in unique_candidates:

                unique_candidates.append(
                    model_name
                )

        return unique_candidates

    def _generate_with_router(
        self,
        model_name: str,
        prompt: str
    ) -> str:
        """
        Generate text using Hugging Face's chat-completions router.
        """

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a retail demand planner. Reply with exactly "
                        "three markdown sections and mention sales figures."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": 600,
            "temperature": 0.4,
            "top_p": 0.9,
        }

        response = requests.post(
            self.ROUTER_URL,
            headers=self.headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:

            raise RuntimeError(
                f"HF router returned {response.status_code}: "
                f"{response.text[:300]}"
            )

        result = response.json()

        if not isinstance(result, dict):

            return ""

        choices = result.get(
            "choices",
            []
        )

        if not choices:

            return ""

        message = choices[0].get(
            "message",
            {}
        )

        return str(
            message.get(
                "content",
                ""
            )
        ).strip()

    def generate_text(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 450,
        temperature: float = 0.35
    ) -> str:
        """
        Generate a general business answer using the configured HF model.

        This method is used by the conversational assistant. It keeps the
        same authentication, endpoint, and model routing behavior as the
        recommendation report generator.
        """

        last_error = None

        for model_name in self._model_candidates():

            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            }

            response = requests.post(
                self.ROUTER_URL,
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                last_error = (
                    f"HF router returned {response.status_code}: "
                    f"{response.text[:300]}"
                )
                continue

            result = response.json()
            choices = result.get("choices", []) if isinstance(result, dict) else []
            if not choices:
                last_error = "HF router returned no choices."
                continue

            message = choices[0].get("message", {})
            text = str(message.get("content", "")).strip()
            if text:
                self.model_name = model_name
                return text

            last_error = "HF router returned empty content."

        raise RuntimeError(last_error or "HF text generation unavailable.")

    def _build_prompt(
        self,
        data: dict[str, Any]
    ) -> str:
        """
        Build a concise prompt for retail forecasting analysis.
        """

        store_str = f"Store {data['store_id']}"

        dept_str = (
            f"Department {data['department_id']}"
            if data.get("department_id")
            else "All Departments"
        )

        return (
            "You are an expert retail demand planner and inventory analyst.\n\n"
            "Use the forecast metrics below to write a concise business report.\n\n"
            "Sales Data Context:\n"
            f"- Target Level: {store_str}, {dept_str}\n"
            f"- Forecast Horizon: {data['horizon']} weeks\n"
            f"- Historical Weekly Sales Mean: ${data['average_historical']:,.2f}\n"
            f"- Forecasted Weekly Sales Mean: ${data['average_forecast']:,.2f}\n"
            f"- Total Forecasted Revenue: ${data['total_forecast']:,.2f}\n"
            f"- Projected Trend: {data['trend_direction']} "
            f"({data['change_pct']}% change compared to history)\n\n"
            "Return exactly these markdown sections:\n"
            "#### 1. Demand & Trend Analysis\n"
            "#### 2. Inventory & Stocking Recommendations\n"
            "#### 3. Operational & Marketing Actions\n\n"
            "Mention the provided sales numbers. Keep the tone practical, "
            "analytical, and recruiter-demo friendly."
        )

    def _clean_response(
        self,
        text: str,
        prompt: str
    ) -> str:
        """
        Remove prompt echo and extra whitespace from model output.
        """

        clean_text = text.replace(
            prompt,
            ""
        ).strip()

        return clean_text

    def _verify_output(
        self,
        text: str,
        data: dict[str, Any]
    ) -> bool:
        """
        Validate that the response is useful business analysis.
        """

        if not text or len(text) < 180:

            return False

        lower_text = text.lower()

        required_terms = [
            "demand",
            "inventory",
            "recommend"
        ]

        if not all(
            term in lower_text
            for term in required_terms
        ):

            return False

        if (
            "sales" not in lower_text
            and "revenue" not in lower_text
            and "demand" not in lower_text
        ):

            return False

        return True

    def _generate_fallback_insights(
        self,
        data: dict[str, Any],
        reason: str
    ) -> str:
        """
        Generate a deterministic backup report if HF is unavailable.
        """

        dept_str = (
            f"Department {data['department_id']}"
            if data.get("department_id")
            else "All Departments"
        )

        change_abs = abs(
            float(data["change_pct"])
        )

        trend = str(
            data["trend_direction"]
        ).lower()

        if (
            "increase" in trend
            or "upward" in trend
            or data["change_pct"] > 0
        ):

            demand_analysis = (
                f"Sales are projected to grow by {change_abs:.1f}% over "
                f"the next {data['horizon']} weeks. Forecasted weekly "
                f"sales are ${data['average_forecast']:,.2f}, compared "
                f"with the historical weekly mean of "
                f"${data['average_historical']:,.2f}."
            )

            inventory_advice = (
                f"- Increase safety stock for high-velocity items in {dept_str}.\n"
                "- Review supplier lead times before the demand peak.\n"
                "- Monitor shelf availability weekly to avoid lost sales."
            )

            actions = (
                "- Add labor coverage during high-traffic periods.\n"
                "- Promote high-margin products while demand is rising.\n"
                "- Use end-cap placement for fast-moving items."
            )

        elif (
            "decrease" in trend
            or "downward" in trend
            or data["change_pct"] < 0
        ):

            demand_analysis = (
                f"Sales are projected to decline by {change_abs:.1f}% "
                f"over the next {data['horizon']} weeks. Forecasted "
                f"weekly sales are ${data['average_forecast']:,.2f}, "
                f"below the historical weekly mean of "
                f"${data['average_historical']:,.2f}."
            )

            inventory_advice = (
                "- Reduce replenishment quantities for slow-moving items.\n"
                "- Review markdown plans to prevent excess inventory.\n"
                "- Keep only essential safety stock until demand recovers."
            )

            actions = (
                "- Run targeted promotions to protect sales volume.\n"
                "- Rebalance labor scheduling for lower traffic periods.\n"
                "- Review assortment mix and remove weak performers."
            )

        else:

            demand_analysis = (
                f"Sales are expected to remain stable near "
                f"${data['average_forecast']:,.2f} per week across the "
                f"{data['horizon']}-week horizon."
            )

            inventory_advice = (
                "- Maintain normal replenishment cycles.\n"
                "- Keep existing safety stock policies unchanged.\n"
                "- Continue weekly inventory checks."
            )

            actions = (
                "- Use the stable period for operational cleanup.\n"
                "- Keep promotions focused on reliable staple products.\n"
                "- Monitor drift signals before changing the model."
            )

        return (
            "#### 1. Demand & Trend Analysis\n"
            f"{demand_analysis}\n\n"
            "#### 2. Inventory & Stocking Recommendations\n"
            f"{inventory_advice}\n\n"
            "#### 3. Operational & Marketing Actions\n"
            f"{actions}\n\n"
            f"Note: Rule-based backup used because live HF generation failed: {reason}"
        )
