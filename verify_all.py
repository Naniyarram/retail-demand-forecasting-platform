"""
End-to-end regression and verification suite for all pipeline components.
"""

from pipeline.preprocessing.data_loader import WalmartDataLoader
from pipeline.preprocessing.aggregations import WalmartAggregator
from pipeline.api.forecast_service import ForecastService
from pipeline.inventory.optimization import optimize_inventory
from pipeline.inventory.risk import classify_risk
from pipeline.monitoring.drift_detector import DataDriftDetector
from pipeline.utils.llm_client import HFLLMClient
import pandas as pd


def main():
    passed = 0

    # LLM Client
    client = HFLLMClient()
    assert client.model_name
    print("1.  LLM client ............. PASS")
    passed += 1

    # Training Data
    loader = WalmartDataLoader()
    train = loader.load_train_data()
    assert len(train) > 400_000
    print(f"2.  Train data ({len(train):,} rows) .. PASS")
    passed += 1

    # Stores Metadata
    stores = loader.load_stores_data()
    assert len(stores) == 45
    print(f"3.  Stores ({len(stores)} entries) ..... PASS")
    passed += 1

    # Feature Flags/External Factors
    features = loader.load_features_data()
    assert len(features) > 5000
    print(f"4.  Features ({len(features):,} rows) .. PASS")
    passed += 1

    # Company Aggregation
    agg = WalmartAggregator()
    company = agg.get_company_sales(train)
    assert len(company) > 100
    print(f"5.  Company series ({len(company)} wks) PASS")
    passed += 1

    # Store Aggregation
    store1 = agg.get_store_sales(train, store_id=1)
    assert len(store1) > 100
    print(f"6.  Store 1 series ({len(store1)} wks) . PASS")
    passed += 1

    # Top Stores Revenue Ranking
    top_stores = agg.get_top_stores(train, top_n=5)
    store_ids = top_stores["Store"].tolist()
    assert len(store_ids) == 5
    print(f"7.  Top 5 stores {store_ids} ... PASS")
    passed += 1

    # Model Registry/Loader
    service = ForecastService()
    service.load_model()
    model_name = service.get_model_name()
    assert model_name in {"SARIMA", "Prophet", "XGBoost"}
    print(f"8.  Champion model ({model_name}) .. PASS")
    passed += 1

    # Forecast Generation
    preds = service.forecast(12)
    assert len(preds) == 12
    print(f"9.  Forecast (12 weeks) ...... PASS")
    passed += 1

    # Drift Detection
    detector = DataDriftDetector(alpha=0.05)
    split = int(len(company) * 0.7)
    result = detector.detect_drift(
        company.iloc[:split],
        company.iloc[split:],
        ["Weekly_Sales"],
    )
    drift_flag = result["drift_detected"]
    print(f"10. Drift detector (drift={drift_flag}) PASS")
    passed += 1

    # Leaderboard Tracking
    lb = pd.read_csv("artifacts/leaderboards/leaderboard.csv")
    champ_name = lb.iloc[0]["Model"]
    champ_mape = lb.iloc[0]["MAPE"]
    assert champ_name in {"SARIMA", "Prophet", "XGBoost"}
    print(f"11. Leaderboard ({len(lb)} models, champ={champ_name}, MAPE={champ_mape:.2f}%) PASS")
    passed += 1

    # Inventory Optimization
    inv = optimize_inventory([48e6, 49e6, 47e6], 3e6, 2.0, 0.95)
    assert inv["safety_stock"] > 0
    assert inv["reorder_point"] > 0
    assert inv["economic_order_quantity"] > 0
    print(f"12. Inventory optimizer (SS={inv['safety_stock']:,.0f}) PASS")
    passed += 1

    # Inventory Risk Assessment
    risk = classify_risk(40e6, inv["reorder_point"], inv["safety_stock"], 144e6)
    assert risk["stockout_risk"]["level"] in ("Low", "Medium", "High", "Critical")
    assert risk["overstock_risk"]["level"] in ("Low", "Medium", "High")
    print(f"13. Risk classifier (stockout={risk['stockout_risk']['level']}) PASS")
    passed += 1

    # Insights Generation
    llm_payload = {
        "store_id": 1,
        "department_id": None,
        "horizon": 12,
        "average_historical": 48e6,
        "average_forecast": 49.5e6,
        "total_forecast": 594e6,
        "trend_direction": "Upward of +3.1%",
        "change_pct": 3.1,
    }
    llm_result = client.generate_retail_insights(llm_payload)
    assert len(llm_result["raw_insights"]) > 200
    assert "model_used" in llm_result
    assert llm_result["verified"] is True, (
        f"LLM fell back to rule-based output ({llm_result['model_used']})"
    )
    print(f"14. LLM insights engine ({llm_result['model_used'][:30]}...) PASS")
    passed += 1

    print()
    print("=" * 55)
    print(f"  ALL {passed}/14 COMPONENTS VERIFIED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    main()
