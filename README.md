# Bank Marketing — Azure ML SDK v2

Udacity **Machine Learning Engineer for Microsoft Azure — Project 2: Operationalizing Machine Learning Pipeline**

This project is a modernized implementation of the Udacity Bank Marketing project using the **Azure Machine Learning Python SDK v2** (`azure-ai-ml`).

The original course materials use Azure ML SDK v1. This project updates the workflow to SDK v2 using AutoML, MLflow, Managed Online Endpoints, and REST-based inference.

---

## Project Architecture

```text
Bank Marketing Data
        │
        ▼
prepare_data.py
        │
        ▼
Cleaned MLTable Data Asset
        │
        ▼
Azure ML AutoML
        │
        ▼
Best AutoML Run
        │
        ▼
MLflow Model
        │
        ▼
Managed Online Endpoint
        │
        ▼
Blue Deployment
        │
        ├── SDK v2 inference
        │
        └── REST inference
```

The deployment uses Azure ML's **MLflow no-code deployment** path, so the primary deployment does not require a custom `score.py` or manually maintained inference environment.

---

## Repository Structure

```text
azure-ml-bankmarketing-sdk-v2/
│
├── notebooks/
│   └── bankmarketing_sdk_v2.ipynb
│       # Main SDK v2 project notebook
│
├── src/
│   └── prepare_data.py
│       # Data preparation
│
├── pipeline/
│   └── bankmarketing_pipeline.py
│       # SDK v2 prepare-data → AutoML pipeline
│
├── deployment/
│   ├── deploy.py
│   │   # Primary MLflow deployment script
│   ├── consume.py
│   │   # REST endpoint consumer
│   ├── sample-request.json
│   │   # Endpoint test payload
│   ├── deploy_custom_environment.py
│   ├── score.py
│   └── conda_inference.yaml
│       # Fallback/custom deployment files
│
├── environment/
│   └── conda.yaml
│       # Training/pipeline environment
│
├── README.md
└── .gitignore
```

---

## Azure ML Resources

| Resource         | Value                                 |
| ---------------- | ------------------------------------- |
| Workspace        | `UdacityWorkspace`                    |
| Resource Group   | `udacity-practice`                    |
| Region           | `eastus2`                             |
| Compute          | `udacity-project2-cluster`            |
| Compute SKU      | `Standard_DS2_v2`                     |
| Data Asset       | `bankmarketing-clean-sdkv2:1`         |
| AutoML Job       | `ivory_worm_z0wbhqwv9s`               |
| Best Run         | `ivory_worm_z0wbhqwv9s_8`             |
| Registered Model | `bankmarketing-automl-mlflow-model:4` |
| Endpoint         | `bankmarketing-sdkv2-endpoint`        |
| Deployment       | `blue`                                |

---

## Notebook

The main notebook is:

```text
notebooks/bankmarketing_sdk_v2.ipynb
```

It supports two execution modes.

### Existing Resources

```python
EXECUTION_MODE = "EXISTING"
```

Use this mode to continue from the already completed AutoML job, model, endpoint, and deployment.

This is the recommended mode for reproducing the completed project without retraining or redeploying unnecessarily.

### New Training / Deployment

```python
EXECUTION_MODE = "NEW"
```

This mode:

1. Creates a new AutoML classification job.
2. Waits for completion.
3. Identifies the winning child run.
4. Calls `deployment/deploy.py`.
5. Verifies the managed online endpoint.
6. Tests SDK and REST inference.

---

## AutoML

The project uses Azure ML AutoML classification with:

```text
Target column:       y
Primary metric:      accuracy
Cross-validation:    5 folds
Compute:             udacity-project2-cluster
Max trials:          10
Concurrent trials:   1
```

The completed AutoML experiment produced:

```text
Parent job:
ivory_worm_z0wbhqwv9s

Best child run:
ivory_worm_z0wbhqwv9s_8
```

The winning model was registered as:

```text
bankmarketing-automl-mlflow-model:4
```

---

## Deployment

The primary deployment mechanism is:

```text
deployment/deploy.py
```

Example:

```bash
cd deployment

python deploy.py --best_run_id ivory_worm_z0wbhqwv9s_8
```

The script registers the winning MLflow model and deploys it to the managed online endpoint.

Azure ML supports deployment of MLflow models to managed online endpoints without requiring a custom scoring script for the standard MLflow deployment path.

The files:

```text
deploy_custom_environment.py
score.py
conda_inference.yaml
```

are retained as fallback/reference implementations and are not the primary deployment path.

---

## Managed Online Endpoint

The deployed endpoint is:

```text
bankmarketing-sdkv2-endpoint
```

with:

```text
Deployment:      blue
Instance type:   Standard_DS2_v2
Instance count:  1
Traffic:         100%
```

Scoring URI:

```text
https://bankmarketing-sdkv2-endpoint.eastus2.inference.ml.azure.com/score
```

---

## Testing the Endpoint

### SDK v2

The notebook invokes the endpoint using:

```python
response = ml_client.online_endpoints.invoke(
    endpoint_name="bankmarketing-sdkv2-endpoint",
    deployment_name="blue",
    request_file="deployment/sample-request.json"
)

print(response)
```

Result:

```text
[false]
```

### REST

The project also includes:

```text
deployment/consume.py
```

This script retrieves the endpoint credential through the Azure ML SDK and sends an HTTP POST request to the scoring URI.

Validated result:

```text
Status code: 200
Prediction: [false]
```

---

## Final Validation

The completed deployment was successfully verified:

```text
AutoML job:
    Completed

Best child run:
    ivory_worm_z0wbhqwv9s_8

Registered model:
    bankmarketing-automl-mlflow-model:4

Endpoint:
    bankmarketing-sdkv2-endpoint
    Succeeded

Deployment:
    blue
    Succeeded

Traffic:
    {'blue': 100}

SDK v2 inference:
    Successful

REST inference:
    HTTP 200 / Successful
```

---

## Data Preparation

Raw Bank Marketing data is prepared using:

```text
src/prepare_data.py
```

The cleaned data is registered as:

```text
bankmarketing-clean-sdkv2:1
```

The project also includes an SDK v2 pipeline:

```text
pipeline/bankmarketing_pipeline.py
```

which represents the:

```text
Data Preparation → AutoML
```

workflow.

---

## Environment

Training and pipeline dependencies are defined in:

```text
environment/conda.yaml
```

The production deployment uses the MLflow model deployment path rather than manually recreating the training environment for inference.

---

## Authentication

Azure resources are accessed using:

```python
DefaultAzureCredential()
```

The REST consumer retrieves the endpoint key programmatically rather than storing the key in source code.

**Do not commit Azure credentials, endpoint keys, or other secrets to GitHub.**

---

## Technology Stack

* Python
* Azure Machine Learning SDK v2
* Azure AutoML
* MLflow
* scikit-learn
* MLTable
* Managed Online Endpoints
* REST API

---

## References

* [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/)
* [Managed Online Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints)
* [Deploy MLflow Models](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models)
* [Azure ML SDK v2 Deployment Tutorial](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-model)

---

## Project Status

**Completed and validated.**

The Bank Marketing model has been trained with Azure ML AutoML, registered as an MLflow model, deployed to a Managed Online Endpoint, and successfully tested through both the Azure ML SDK v2 and REST API.
