
"""Local MLTable definition and validation helpers.

The functions in this module do not register Azure Machine Learning assets.
The master notebook owns the decision to create a cloud resource.
"""

from pathlib import Path

import mltable
import pandas as pd


def build_mltable_text(raw_data_path: str) -> str:
    """Return the canonical MLTable YAML for the uploaded CSV file."""

    if not raw_data_path or not str(raw_data_path).strip():
        raise ValueError("raw_data_path must not be empty.")

    return (
        "paths:\n"
        f"  - file: {raw_data_path}\n"
        "transformations:\n"
        "  - read_delimited:\n"
        '      delimiter: ","\n'
        '      encoding: "utf8"\n'
        "      header: all_files_same_headers\n"
        "      infer_column_types: true\n"
    )



def ensure_local_mltable_definition(
    mltable_file: Path,
    expected_text: str,
    allow_refresh: bool = False,
) -> str:
    """Create, compare, or explicitly refresh the local MLTable file.

    Ignores harmless whitespace differences.
    Returns MATCHED, CREATED, DIFFERENT, or REFRESHED.
    """

    mltable_file = Path(mltable_file)

    def normalize(text: str) -> str:
        return "\n".join(
            line.rstrip()
            for line in text.strip().splitlines()
            if line.strip()
        )

    expected_text = expected_text.strip() + "\n"

    if not mltable_file.exists():
        mltable_file.parent.mkdir(parents=True, exist_ok=True)
        mltable_file.write_text(expected_text, encoding="utf-8")
        return "CREATED"

    current_text = mltable_file.read_text(encoding="utf-8")

    if normalize(current_text) == normalize(expected_text):
        return "MATCHED"

    if not allow_refresh:
        return "DIFFERENT"

    mltable_file.write_text(expected_text, encoding="utf-8")
    return "REFRESHED"


def load_and_validate_mltable(
    mltable_folder: Path,
    target_column: str,
) -> pd.DataFrame:
    """Load the local MLTable and validate its target column."""

    folder = Path(mltable_folder)
    definition = folder / "MLTable"

    if not definition.exists():
        raise FileNotFoundError(
            f"MLTable definition not found: {definition}"
        )

    table = mltable.load(str(folder))
    dataframe = table.to_pandas_dataframe()

    if dataframe.empty:
        raise ValueError("The MLTable produced an empty DataFrame.")

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column {target_column!r} was not found. "
            f"Columns: {dataframe.columns.tolist()}"
        )

    return dataframe


if __name__ == "__main__":
    print(
        "Import this module from Project_2_Master_Notebook_SDK_v2_Working.ipynb. "
        "No Azure assets are registered by this diagnostic."
    )
