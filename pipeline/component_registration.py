
"""
Register or retrieve a reusable Azure ML SDK v2 pipeline component.
"""

from azure.ai.ml.entities import PipelineComponent


def get_or_register_pipeline_component(
    ml_client,
    pipeline_job_definition,
    component_name,
    component_version,
    allow_registration=True,
):
    """
    Retrieve an existing pipeline component or register it
    when absent and registration is enabled.
    """

    # Validate Azure ML component naming requirements.
    if not component_name:
        raise ValueError("Component name cannot be empty.")

    if not (
        component_name[0].islower()
        and all(
            char.islower()
            or char.isdigit()
            or char == "_"
            for char in component_name
        )
    ):
        raise ValueError(
            "Invalid component name. Use lowercase letters, "
            "numbers, and underscores, beginning with a "
            "lowercase letter."
        )

    # --------------------------------------------------------
    # STEP 1: Retrieve existing component.
    # --------------------------------------------------------

    try:
        existing_component = ml_client.components.get(
            name=component_name,
            version=str(component_version),
        )

        print("Reusing existing pipeline component:")
        print("Name:", existing_component.name)
        print("Version:", existing_component.version)

        return existing_component, False

    except Exception as exc:
        # Only continue to registration when the component
        # is genuinely absent. Do not hide authentication,
        # permission, or service errors.
        error_text = str(exc).lower()

        if not any(
            term in error_text
            for term in [
                "notfound",
                "not found",
                "resource not found",
                "404",
            ]
        ):
            raise

    # --------------------------------------------------------
    # STEP 2: Register component only when explicitly allowed.
    # --------------------------------------------------------

    if not allow_registration:
        raise RuntimeError(
            f"Pipeline component {component_name}:"
            f"{component_version} does not exist, and "
            "component registration is disabled."
        )

   
    # --------------------------------------------------------
    # STEP 3: Reuse the DSL-generated PipelineComponent.
    # --------------------------------------------------------

    component = pipeline_job_definition.component

    if component is None:
        raise ValueError(
            "PipelineJob does not contain a generated PipelineComponent."
        )

    component.name = component_name
    component.version = str(component_version)

    component.tags = {
        "project": "udacity-project-2",
        "sdk": "v2",
    }

    registered = ml_client.components.create_or_update(component)

    print("Pipeline component registered successfully.")

    return registered, True
