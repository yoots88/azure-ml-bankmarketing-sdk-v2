"""Pipeline step: validate and materialize the training table."""
import argparse
from pathlib import Path
import mltable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--validated_data", required=True)
    args = parser.parse_args()

    dataframe = mltable.load(args.input_data).to_pandas_dataframe()
    if dataframe.empty:
        raise ValueError("Training data is empty.")
    if args.target_column not in dataframe.columns:
        raise ValueError(f"Missing target column: {args.target_column}")

    output = Path(args.validated_data)
    output.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output / "validated.csv", index=False)
    print({"rows": len(dataframe), "columns": len(dataframe.columns)})


if __name__ == "__main__":
    main()