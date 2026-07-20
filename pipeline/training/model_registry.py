"""
model_registry.py

MLflow Model Registry Manager.

Responsibilities
----------------
- Register Models
- Assign Aliases
- Retrieve Registered Models
- List Versions
- Rollback Champion Alias


"""

from typing import List
from typing import Dict
from typing import Optional

import mlflow
from mlflow import MlflowClient


class ModelRegistryManager:
    """
    MLflow Registry Management.
    """

    def __init__(self):

        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        registered_model_name: str
    ) -> int:
        """
        Register model in MLflow Registry.

        Returns
        -------
        int
            Registered version number.
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
        Assign alias to model version.

        Examples:
        ----------
        champion
        challenger
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
        Retrieve model version by alias.
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
        Get current champion version.
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
        List all registered versions.
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
        Promote model version
        to champion alias.
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
        Promote model version
        to challenger alias.
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
        Rollback champion alias.
        """

        self.set_alias(
            model_name=model_name,
            version=version,
            alias="champion"
        )