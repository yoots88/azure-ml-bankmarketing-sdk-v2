"""Retrieve Managed Online Deployment logs through MLClient."""


def get_deployment_logs(ml_client, endpoint_name, deployment_name, lines=100):
    return ml_client.online_deployments.get_logs(
        name=deployment_name,
        endpoint_name=endpoint_name,
        lines=lines,
    )


if __name__ == "__main__":
    print("Import this module from Project_2_Master_Notebook_SDK_v2_Working.ipynb.")