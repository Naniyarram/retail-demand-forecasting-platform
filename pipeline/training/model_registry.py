"""
Manages model registration, aliasing, and promotion in the MLflow Model Registry.
"""

from typing import List
from typing import Dict
from typing import Optional

import mlflow
from mlflow import MlflowClient


class ModelRegistryManager:
    """
    API client wrapper for MLflow model registry tasks.
    """

    def __init__(self):

        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        registered_model_name: str
    ) -> int:
        """
        Registers a new model version in the MLflow Registry.
        """

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name
        )

        return int(
            model_version.version
        )

    def set_alias(
        self,
        model_name: str,
        version: int,
        alias: str
    ) -> None:
        """
        Assigns a custom alias (e.g. 'champion' or 'challenger') to a specific model version.
        """

        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version)
        )

    def get_model_by_alias(
        self,
        model_name: str,
        alias: str = "champion"
    ):
        """
        Retrieves model version info matching the specified alias.
        """

        return self.client.get_model_version_by_alias(
            name=model_name,
            alias=alias
        )

    def get_champion_version(
        self,
        model_name: str
    ) -> Optional[int]:
        """
        Returns the version number currently tagged as the champion, or None if not set.
        """

        try:

            version = (
                self.client
                .get_model_version_by_alias(
                    name=model_name,
                    alias="champion"
                )
            )

            return int(version.version)

        except Exception:

            return None

    def list_versions(
        self,
        model_name: str
    ) -> List[Dict]:
        """
        Queries and returns a list of all registered versions for a given model.
        """

        versions = (
            self.client.search_model_versions(
                f"name='{model_name}'"
            )
        )

        return [
            {
                "version": v.version,
                "status": v.status,
                "run_id": v.run_id
            }
            for v in versions
        ]

    def promote_to_champion(
        self,
        model_name: str,
        version: int
    ) -> None:
        """
        Updates the 'champion' alias to point to the specified model version.
        """

        self.set_alias(
            model_name=model_name,
            version=version,
            alias="champion"
        )

    def promote_to_challenger(
        self,
        model_name: str,
        version: int
    ) -> None:
        """
        Updates the 'challenger' alias to point to the specified model version.
        """

        self.set_alias(
            model_name=model_name,
            version=version,
            alias="challenger"
        )

    def rollback_champion(
        self,
        model_name: str,
        version: int
    ) -> None:
        """
        Points the champion alias back to a previous model version.
        """

        self.set_alias(
            model_name=model_name,
            version=version,
            alias="champion"
        )