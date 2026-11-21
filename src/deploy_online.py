
"""Rerun-safe Managed Online Endpoint and deployment helpers."""

from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)

from src import settings


def get_or_create_endpoint(ml_client, endpoint_name, allow_creation=False):
    try:
        return ml_client.online_endpoints.get(endpoint_name)

    except ResourceNotFoundError:
        if not allow_creation:
            raise RuntimeError(
                f"Endpoint {endpoint_name!r} does not exist. "
                "Set ALLOW_ENDPOINT_CREATION=True to create it."
            )

    candidate = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Udacity Project 2 Bank Marketing online endpoint",
        auth_mode="key",
        tags={"project": "udacity-project-2", "sdk": "v2"},
    )

    return ml_client.online_endpoints.begin_create_or_update(
        candidate
    ).result()


def get_or_create_deployment(
    ml_client,
    endpoint_name,
    deployment_name,
    model,
    allow_creation=False,
):
    try:
        deployment = ml_client.online_deployments.get(
            name=deployment_name,
            endpoint_name=endpoint_name,
        )

        deployed_model = str(getattr(deployment, "model", ""))

        if (
            model.name not in deployed_model
            or str(model.version) not in deployed_model
        ):
            raise RuntimeError(
                "Existing deployment does not reference the selected "
                "model version. Do not replace it silently."
            )

        return deployment

    except ResourceNotFoundError:
        if not allow_creation:
            raise RuntimeError(
                f"Deployment {deployment_name!r} does not exist. "
                "Set ALLOW_DEPLOYMENT_CREATION=True to create it."
            )

    candidate = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type=settings.ONLINE_INSTANCE_TYPE,
        instance_count=settings.ONLINE_INSTANCE_COUNT,
        app_insights_enabled=True,
    )

    return ml_client.online_deployments.begin_create_or_update(
        candidate
    ).result()


def ensure_traffic(
    ml_client,
    endpoint,
    deployment_name,
    allow_update=False,
):
    current = dict(getattr(endpoint, "traffic", {}) or {})

    if current.get(deployment_name) == 100:
        return endpoint

    if not allow_update:
        raise RuntimeError(
            f"Traffic is currently {current}. "
            "Set ALLOW_TRAFFIC_UPDATE=True to route 100% traffic."
        )

    endpoint.traffic = {deployment_name: 100}

    return ml_client.online_endpoints.begin_create_or_update(
        endpoint
    ).result()
