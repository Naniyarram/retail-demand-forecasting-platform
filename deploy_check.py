"""
deploy_check.py

Pre-deployment validation for Streamlit Community Cloud.
Run this from the project root before pushing.
"""

from pathlib import Path


def check(label, value):
    icon = "OK" if value else "FAIL"
    print(f"  [{icon}]  {label}")
    return value


def main():
    print("\n=== Streamlit Cloud Deployment Readiness Check ===\n")

    all_ok = True

    # Imports
    print("1. Module imports")
    try:
        from pipeline.preprocessing.data_loader import WalmartDataLoader
        from pipeline.preprocessing.aggregations import WalmartAggregator
        from pipeline.api.forecast_service import ForecastService
        from pipeline.inventory.optimization import optimize_inventory
        from pipeline.inventory.risk import classify_risk
        from pipeline.monitoring.drift_detector import DataDriftDetector
        from pipeline.utils.conversational_assistant import ConversationalRetailAssistant
        from pipeline.utils.llm_client import HFLLMClient
        print("  [OK]  All pipeline imports successful")
    except ImportError as e:
        print(f"  [FAIL]  Import error: {e}")
        all_ok = False

    # Data files
    print("\n2. Data files")
    data_dir = Path("data")
    for fname in ["train.csv", "features.csv", "stores.csv"]:
        ok = check(fname, (data_dir / fname).exists())
        all_ok = all_ok and ok

    # Artifacts
    print("\n3. Artifacts")
    ok = check("artifacts/models/champion_model.pkl", Path("artifacts/models/champion_model.pkl").exists())
    all_ok = all_ok and ok
    ok = check("artifacts/models/champion_metadata.json", Path("artifacts/models/champion_metadata.json").exists())
    all_ok = all_ok and ok
    ok = check("artifacts/leaderboards/leaderboard.csv", Path("artifacts/leaderboards/leaderboard.csv").exists())
    all_ok = all_ok and ok

    # Streamlit config
    print("\n4. Streamlit configuration")
    ok = check(".streamlit/config.toml", Path(".streamlit/config.toml").exists())
    all_ok = all_ok and ok
    ok = check(".streamlit/secrets.toml.example", Path(".streamlit/secrets.toml.example").exists())
    all_ok = all_ok and ok

    # Requirements
    print("\n5. Requirements")
    req = Path("requirements.txt").read_text()
    ok = check("streamlit in requirements.txt", "streamlit" in req)
    all_ok = all_ok and ok
    ok = check("altair in requirements.txt", "altair" in req)
    all_ok = all_ok and ok
    ok = check("scipy in requirements.txt", "scipy" in req)
    all_ok = all_ok and ok
    ok = check("fastapi NOT in requirements.txt", "fastapi" not in req)
    all_ok = all_ok and ok

    # Model load test
    print("\n6. Champion model load test")
    try:
        from pipeline.api.forecast_service import ForecastService
        svc = ForecastService()
        svc.load_model()
        name = svc.get_model_name()
        preds = svc.forecast(4)
        check(f"model loaded: {name}", True)
        check(f"4-week forecast returns {len(preds)} values", len(preds) == 4)
    except Exception as e:
        check(f"model load failed: {e}", False)
        all_ok = False

    # Data load test
    print("\n7. Data load test")
    try:
        from pipeline.preprocessing.data_loader import WalmartDataLoader
        loader = WalmartDataLoader()
        train = loader.load_train_data()
        check(f"train.csv loaded ({len(train):,} rows)", len(train) > 0)
    except Exception as e:
        check(f"data load failed: {e}", False)
        all_ok = False

    print()
    if all_ok:
        print("=== ALL CHECKS PASSED — ready for Streamlit Cloud deployment ===\n")
    else:
        print("=== SOME CHECKS FAILED — review above before deploying ===\n")

    return all_ok


if __name__ == "__main__":
    import sys
    ok = main()
    sys.exit(0 if ok else 1)
