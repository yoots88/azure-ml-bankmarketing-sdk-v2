import json
from pathlib import Path

import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


# ============================================================
# CONFIGURATION
# ============================================================

SUBSCRIPTION_ID = "6214e780-2ea7-42fc-965e-dae0d86631fc"
RESOURCE_GROUP = "udacity-practice"
WORKSPACE_NAME = "UdacityWorkspace"

ENDPOINT_NAME = "bankmarketing-sdkv2-endpoint"

ENDPOINT_URL = (
    "https://bankmarketing-sdkv2-endpoint.eastus2.inference.ml.azure.com/score"
)


# ============================================================
# CONNECT TO AZURE ML
# ============================================================

credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)


# ============================================================
# GET ENDPOINT KEY
# ============================================================
#
# The endpoint uses key authentication.
#
# Azure ML retrieves the key for us through the authenticated
# control-plane connection.
#
# The key is NOT hard-coded and is NOT printed.
#
# Microsoft documents get_keys() for retrieving the data-plane
# credential when using a custom HTTP client.
# ============================================================

api_key = ml_client.online_endpoints.get_keys(
    name=ENDPOINT_NAME
).primary_key


# ============================================================
# LOAD REQUEST
# ============================================================

request_file = (
    Path(__file__).resolve().parent / "sample-request.json"
)

with request_file.open("r", encoding="utf-8") as f:
    data = json.load(f)


# ============================================================
# SEND REQUEST
# ============================================================

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

response = requests.post(
    ENDPOINT_URL,
    headers=headers,
    json=data,
    timeout=60,
)


# ============================================================
# RESULT
# ============================================================

print("Status code:", response.status_code)
print("Prediction:", response.text)

response.raise_for_status()
