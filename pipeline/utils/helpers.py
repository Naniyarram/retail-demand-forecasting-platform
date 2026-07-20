"""
helpers.py

Small shared utilities used across pipeline layers.


"""

import json
from pathlib import Path
from typing import Any


def save_json(
    data: dict[str, Any],
    path: str | Path
) -> str:
    """
    Save a dictionary as pretty JSON.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            default=str
        )

    return str(output_path)


def load_json(
    path: str | Path
) -> dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    """

    input_path = Path(path)

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)
