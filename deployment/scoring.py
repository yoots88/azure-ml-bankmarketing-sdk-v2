"""Invoke and validate a Managed Online Endpoint response."""
import json
from pathlib import Path


def invoke_and_validate(ml_client, endpoint_name, request_file,
                        deployment_name=None):
    request_file = Path(request_file)
    if not request_file.exists():
        raise FileNotFoundError(request_file)
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        request_file=str(request_file),
    )
    if response is None or not str(response).strip():
        raise RuntimeError("Endpoint returned an empty response.")
    try:
        return json.loads(response) if isinstance(response, str) else response
    except json.JSONDecodeError:
        return response


if __name__ == "__main__":
    print("Import this module from Project_2_Master_Notebook_SDK_v2_Working.ipynb.")