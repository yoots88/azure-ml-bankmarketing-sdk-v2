"""Pipeline step: train a lightweight classifier and write MLflow-style files."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated_data", required=True)
    parser.add_argument("--target_column", required=True)
    parser.add_argument("--model_output", required=True)
    parser.add_argument("--metrics_output", required=True)
    args = parser.parse_args()

    dataframe = pd.read_csv(Path(args.validated_data) / "validated.csv")
    x = dataframe.drop(columns=[args.target_column])
    y = dataframe[args.target_column].astype(str).str.lower().map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target must contain yes/no values.")

    categorical = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in x.columns if c not in categorical]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(x, y)
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "training_accuracy": float(accuracy_score(y, prediction)),
        "training_auc": float(roc_auc_score(y, probability)),
        "rows": int(len(dataframe)),
    }

    model_dir = Path(args.model_output); model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_output); metrics_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()