"""
Statistical data drift detection using Kolmogorov-Smirnov tests.
"""

from typing import Dict, Any, List
import pandas as pd
from scipy.stats import ks_2samp


class DataDriftDetector:
    """
    Detector for feature distribution shifts between baseline and production datasets.
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
        Runs two-sample KS tests on the specified columns to find significant distribution shifts.
        """
        drift_results = {}
        drift_detected = False

        for col in columns:
            if col not in baseline_df.columns:
                continue
            if col not in current_df.columns:
                continue

            # Drop NaNs prior to testing
            baseline_data = baseline_df[col].dropna()
            current_data = current_df[col].dropna()

            if len(baseline_data) == 0 or len(current_data) == 0:
                continue

            # Run two-sample KS test
            stat, p_val = ks_2samp(baseline_data, current_data)
            
            # Reject null hypothesis of identical distributions if p-value < alpha
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
