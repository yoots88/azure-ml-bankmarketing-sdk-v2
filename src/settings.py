"""Central, non-secret settings for Udacity Project 2.

This module defines names and limits only. Importing it must not connect to
Azure, submit jobs, create resources, invoke endpoints, or perform cleanup.
"""

# ============================================================
# WORKSPACE
# ============================================================
RESOURCE_GROUP = "rg-udacity-azureml"
WORKSPACE_NAME = "udacity-project"
COMPUTE_INSTANCE_NAME = "ci-udacity-project"
COMPUTE_NAME = "cpu-cluster"

# ============================================================
# DATA
# ============================================================
RAW_DATA_NAME = "bankmarketing"
RAW_DATA_VERSION = "1"
DATA_NAME = "bankmarketing-mltable"
DATA_VERSION = "1"
TARGET_COLUMN = "y"

# ============================================================
# PART 1: AUTOML
# ============================================================
EXPERIMENT_NAME = "bankmarketing-automl-sdk-v2"
PRIMARY_METRIC = "AUC_weighted"
AUTOML_TIMEOUT_MINUTES = 60
TRIAL_TIMEOUT_MINUTES = 20
MAX_TRIALS = 5
MAX_CONCURRENT_TRIALS = 1

# ============================================================
# PART 1: MODEL REGISTRATION
# ============================================================
MODEL_NAME = "bankmarketing-automl-model"

# ============================================================
# PART 1: MANAGED ONLINE ENDPOINT
# ============================================================
ONLINE_ENDPOINT_NAME = "bankmarketing-endpoint"
DEPLOYMENT_NAME = "blue"

# ONLINE_INSTANCE_TYPE = "Standard_DS3_v2"
ONLINE_INSTANCE_TYPE = "Standard_DS1_v2"
ONLINE_INSTANCE_COUNT = 1

EXPECTED_MIN_NODES = 0
EXPECTED_MAX_NODES = 1
EXPECTED_IDLE_SECONDS = 120

# ============================================================
# PART 2: PIPELINE OPERATIONALIZATION SETTINGS
# ============================================================
PIPELINE_EXPERIMENT_NAME = (
	"bankmarketing-pipeline-sdk-v2"
)
PIPELINE_COMPONENT_NAME = (
	"bankmarketing_training_pipeline"
)
PIPELINE_COMPONENT_VERSION = "1"
BATCH_ENDPOINT_NAME = (
	"bankmarketing-pipeline-endpoint"
)
BATCH_DEPLOYMENT_NAME = (
	"bankmarketing-pipeline-v1"
)
SCHEDULE_NAME = (
	"bankmarketing-pipeline-evidence-schedule"
)


def validate_part0_settings() -> dict[str, str]:
    """Validate required Part 0 values and return a safe summary."""
    required = {
        "RESOURCE_GROUP": RESOURCE_GROUP,
        "WORKSPACE_NAME": WORKSPACE_NAME,
        "COMPUTE_INSTANCE_NAME": COMPUTE_INSTANCE_NAME,
        "COMPUTE_NAME": COMPUTE_NAME,
        "RAW_DATA_NAME": RAW_DATA_NAME,
        "RAW_DATA_VERSION": RAW_DATA_VERSION,
        "DATA_NAME": DATA_NAME,
        "DATA_VERSION": DATA_VERSION,
        "TARGET_COLUMN": TARGET_COLUMN,
    }
    empty = [name for name, value in required.items() if not str(value).strip()]
    if empty:
        raise ValueError(f"Empty required settings: {empty}")
    if DATA_NAME == RAW_DATA_NAME:
        raise ValueError(
            "The raw uri_file and MLTable must use separate asset families."
        )
    return required


if __name__ == "__main__":
    print(validate_part0_settings())