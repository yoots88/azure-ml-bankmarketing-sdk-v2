"""
deployment/score.py

Only used by the FALLBACK path (deploy_custom_environment.py). The
recommended path (deploy.py) is a no-code MLflow deployment and does
not use this file at all.

Fix vs. the original version: AutoML pickles its sklearn pipeline with
joblib, not the plain `pickle` module. `pickle.load` can sometimes
"work" and sometimes silently choke depending on the exact objects
inside the pipeline (numpy dtypes, custom transformers, etc.) - joblib
is the format AutoML/scikit-learn actually use, so load with joblib.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

model = None


def init():
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR")

    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR is not set.")

    print("Model directory:", model_dir)

    model_path = None

    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pkl"):
                model_path = os.path.join(root, file)
                break
        if model_path:
            break

    if model_path is None:
        raise FileNotFoundError(f"No .pkl model found under {model_dir}")

    print("Loading model:", model_path)

    model = joblib.load(model_path)

    print("Model loaded successfully.")


def run(raw_data):
    try:
        data = json.loads(raw_data)

        # Support Azure ML request format: {"data": [{...}, {...}]}
        if isinstance(data, dict) and "data" in data:
            input_data = data["data"]
        else:
            input_data = data

        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            raise ValueError("Input must be a dictionary or list of records.")

        print("Input shape:", df.shape)
        print("Input columns:", list(df.columns))

        predictions = model.predict(df)
        predictions = np.asarray(predictions).tolist()

        return {"Results": predictions}

    except Exception as e:
        return {"error": str(e)}
