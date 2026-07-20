"""
retrain.py

Automated retraining pipeline based on data drift checks.

"""

from typing import Dict, Any, List
import pandas as pd

from pipeline.monitoring.drift_detector import DataDriftDetector
from pipeline.training.champion_pipeline import ChampionPipeline


class AutomatedRetrainer:
    """
    Automates retraining triggers based on statistical drift detection.
    """

    def __init__(self, significance_level: float = 0.05):
        self.detector = DataDriftDetector(alpha=significance_level)
        self.pipeline = ChampionPipeline()

    def run_retraining_check(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        drift_columns: List[str],
        champion_result: Dict[str, Any],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Check for drift. If detected (or if forced), retrain the champion model.
        """
        # Run drift detection
        drift_report = self.detector.detect_drift(
            baseline_df=baseline_df,
            current_df=current_df,
            columns=drift_columns
        )

        drift_detected = drift_report["drift_detected"]
        triggered = bool(drift_detected or force)
        
        pipeline_output = None
        message = "No drift detected. Retraining skipped."

        if triggered:
            message = "Retraining triggered due to "
            if force:
                message += "manual override (force=True)."
            else:
                message += "detected distribution drift."

            print(f"[INFO] {message}")
            
            # Run the Champion Pipeline on the updated/current dataset
            pipeline_output = self.pipeline.run(
                champion_result=champion_result,
                full_dataset=current_df
            )
        else:
            print(f"[INFO] {message}")

        return {
            "drift_detected": drift_detected,
            "retraining_triggered": triggered,
            "message": message,
            "drift_report": drift_report,
            "pipeline_output": pipeline_output
        }
