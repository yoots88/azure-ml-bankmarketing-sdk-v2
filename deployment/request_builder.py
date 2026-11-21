"""Build safe tabular scoring requests from validated pandas data."""
import json
from pathlib import Path


def build_request(dataframe, target_column, row_count=2):
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column {target_column!r} is missing.")
    features = dataframe.drop(columns=[target_column]).head(row_count)
    if features.empty:
        raise ValueError("No feature rows are available for the request.")
    return {
        "input_data": {
            "columns": features.columns.tolist(),
            "index": list(range(len(features))),
            "data": features.astype(object).where(features.notna(), None).values.tolist(),
        }
    }


def validate_request(payload, forbidden_columns=None):
    node = payload.get("input_data", {})
    columns = node.get("columns", [])
    data = node.get("data", [])
    if not columns or not data:
        raise ValueError("Request must contain input_data.columns and input_data.data.")
    forbidden = set(forbidden_columns or [])
    found = forbidden.intersection(columns)
    if found:
        raise ValueError(f"Forbidden columns are present: {sorted(found)}")
    if any(len(row) != len(columns) for row in data):
        raise ValueError("Request row width does not match the column count.")


def build_and_save_request(dataframe, target_column, output_file, row_count=2):
    payload = build_request(dataframe, target_column, row_count)
    validate_request(payload, [target_column])
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, output_file


if __name__ == "__main__":
    print("Import this module from Project_2_Master_Notebook_SDK_v2_Working.ipynb.")