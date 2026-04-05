"""Table and data export utilities."""

import csv
import json
from pathlib import Path
from typing import Any, Union


def save_json(data: Any, path: Union[str, Path]) -> Path:
    """
    Save data as JSON file.

    Args:
        data: Data to serialize
        path: Output file path

    Returns:
        Path object of saved file
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output


def save_measurements_csv(rows: list[dict[str, Any]], path: Union[str, Path]) -> Path:
    """
    Save measurement data as CSV file.

    Args:
        rows: List of dictionaries with measurement data
        path: Output file path

    Returns:
        Path object of saved file
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Create empty file
        output.touch()
        return output

    # Get all unique keys from all rows while preserving first-seen order.
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output
