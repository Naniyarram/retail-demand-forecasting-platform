"""
drift_detector.py

Statistical data drift monitoring using Kolmogorov-Smirnov tests.


"""

from typing import Dict, Any, List
import pandas as pd
from scipy.stats import ks_2samp


class DataDriftDetector:
    """
    Detects distribution drift between training baseline and incoming datasets.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """
        Run Kolmogorov-Smirnov test on selected columns to identify drift.
        """
        drift_results = {}
        drift_detected = False

        for col in columns:
            if col not in baseline_df.columns:
                continue
            if col not in current_df.columns:
                continue

            # Drop missing values for clean test
            baseline_data = baseline_df[col].dropna()
            current_data = current_df[col].dropna()

            if len(baseline_data) == 0 or len(current_data) == 0:
                continue

            # Perform two-sample KS test
            stat, p_val = ks_2samp(baseline_data, current_data)
            
            # If p-value is lower than significance level, we reject null hypothesis (no drift)
            col_drift = bool(p_val < self.alpha)
            if col_drift:
                drift_detected = True

            drift_results[col] = {
                "ks_statistic": float(stat),
                "p_value": float(p_val),
                "drift_detected": col_drift
            }

        return {
            "drift_detected": drift_detected,
            "significance_level": self.alpha,
            "metrics": drift_results
        }
