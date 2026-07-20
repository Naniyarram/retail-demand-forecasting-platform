"""
champion_trainer.py

Champion model retraining.

"""

from typing import Dict, Any

import pandas as pd


class ChampionTrainer:
    """
    Retrains the champion model
    on the full dataset.
    """

    def __init__(self):
        pass

    def retrain_champion(
        self,
        champion_result: Dict[str, Any],
        full_dataset: pd.DataFrame
    ):
        """
        Retrain champion model using
        the entire available dataset.
        """

        model = champion_result["model_object"]

        model.fit(full_dataset)

        return model

    def get_champion_summary(
        self,
        champion_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return champion metadata.
        """

        summary = {
            "model_name": champion_result["model_name"],
            "RMSE": champion_result["RMSE"],
            "MAE": champion_result["MAE"],
            "MAPE": champion_result["MAPE"],
            "parameters": champion_result["parameters"]
        }

        return summary

    def print_champion_summary(
        self,
        champion_result: Dict[str, Any]
    ) -> None:
        """
        Print champion information.
        """

        print("\nChampion Summary")
        print("-" * 50)

        print(
            f"Model : {champion_result['model_name']}"
        )

        print(
            f"RMSE  : {champion_result['RMSE']:.4f}"
        )

        print(
            f"MAE   : {champion_result['MAE']:.4f}"
        )

        print(
            f"MAPE  : {champion_result['MAPE']:.4f}"
        )