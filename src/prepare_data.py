"""
src/prepare_data.py

Cleans the raw UCI Bank Marketing CSV into the shape used for training
(the same transformation the classic Nanodegree starter's `clean_data`
function performs), and writes the result out as a CSV.

This is meant to run either:
  - standalone: `python prepare_data.py --input_data raw.csv --output_data clean.csv`
  - as an Azure ML SDK v2 command component (see pipeline/bankmarketing_pipeline.py)

Keeping this as a plain, dependency-light script (pandas only) avoids
pulling in azureml-* packages at data-prep time, which is one less place
for environment/version drift to bite you.
"""

import argparse
import os

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the classic Udacity `clean_data` transformation."""

    df = df.copy()

    # Drop rows with no target
    df = df.dropna(subset=["y"])

    # Binary yes/no columns -> 0/1
    df["default"] = df["default"].apply(lambda x: 1 if x == "yes" else 0)
    df["housing"] = df["housing"].apply(lambda x: 1 if x == "yes" else 0)
    df["loan"] = df["loan"].apply(lambda x: 1 if x == "yes" else 0)

    # Fill missing categoricals before one-hot encoding
    df["job"] = df["job"].fillna("unknown")
    df["marital"] = df["marital"].fillna("unknown")

    one_hot_columns = [
        "contact",
        "education",
        "job",
        "marital",
        "month",
        "day_of_week",
        "poutcome",
    ]

    dummies = [
        pd.get_dummies(df[col], prefix=col, drop_first=False)
        for col in one_hot_columns
    ]

    df = df.drop(columns=one_hot_columns)
    df = pd.concat([df] + dummies, axis=1)

    # Target: yes/no -> 1/0
    df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to the raw bankmarketing CSV.",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Path to write the cleaned CSV.",
    )
    args = parser.parse_args()

    print("Reading raw data from:", args.input_data)
    raw_df = pd.read_csv(args.input_data)
    print("Raw shape:", raw_df.shape)

    clean_df = clean_data(raw_df)
    print("Clean shape:", clean_df.shape)

    output_dir = os.path.dirname(args.output_data)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    clean_df.to_csv(args.output_data, index=False)
    print("Wrote cleaned data to:", args.output_data)


if __name__ == "__main__":
    main()
