# Azure ML Bank Marketing – SDK v2 Architecture

## 1. Project Overview

This project implements an end-to-end machine learning workflow
using Azure Machine Learning SDK v2.

The solution covers data validation, model training, pipeline
orchestration, reusable pipeline components, Batch Endpoint
invocation, and scheduled pipeline execution.

## 2. Architecture Overview

Bank Marketing Dataset
    |
    v
Azure ML MLTable Data Asset
    |
    v
Azure ML SDK v2 Pipeline
    |
    +--> Data Validation
    |        |
    |        v
    |    Validated Data
    |        |
    |        v
    |    Model Training
    |        |
    |        v
    |    Trained Model and Metrics
    |
    +--> Registered Pipeline Component
             |
             v
         Batch Endpoint
             |
             v
         Pipeline Deployment
             |
             v
         Endpoint-Triggered Pipeline Job

JobSchedule
    |
    v
Scheduled Pipeline Job

## 3. Core Azure ML Resources

| Resource | Purpose |
|---|---|
| MLClient | Authenticates and manages Azure ML resources |
| MLTable data asset | Provides versioned training data |
| Pipeline components | Encapsulate reusable processing and training steps |
| Pipeline job | Orchestrates the end-to-end workflow |
| Batch Endpoint | Exposes the reusable pipeline for invocation |
| Batch Deployment | Connects the endpoint to the pipeline component |
| JobSchedule | Triggers recurring pipeline executions |
| CPU compute cluster | Executes pipeline workloads |

## 4. Pipeline Workflow

### Data Validation

- Reads the registered MLTable data asset.
- Validates the input dataset and target column.
- Produces validated data for model training.

### Model Training

- Consumes the validated dataset.
- Trains the classification model.
- Produces model and evaluation metric outputs.

### Pipeline Orchestration

Azure ML SDK v2 pipeline jobs coordinate component execution,
input/output dependencies, compute selection, and job monitoring.

## 5. Operationalization

### Reusable Pipeline Component

The pipeline is registered as a versioned Azure ML component.
This supports reuse without rebuilding the pipeline definition
for every execution.

### Batch Endpoint

A Batch Endpoint exposes the registered pipeline component
through a deployment.

The deployment uses the configured CPU compute cluster.
Invoking the endpoint creates a pipeline job in Azure ML.

### Scheduled Execution

A JobSchedule is configured to trigger the pipeline automatically.

The project includes evidence of a successful scheduled execution.
The temporary evidence schedule was subsequently disabled to
prevent additional recurring executions.

## 6. SDK v1 to SDK v2 Migration

| SDK v1 concept | SDK v2 equivalent |
|---|---|
| Workspace | MLClient |
| Dataset.Tabular | MLTable data asset |
| Experiment and Run | SDK v2 jobs |
| PublishedPipeline | Versioned pipeline component |
| PipelineEndpoint | Batch Endpoint with pipeline deployment |
| PipelineSchedule | JobSchedule |
| RunDetails | Studio job details, graph, logs, and metadata |

## 7. Monitoring and Evidence

Execution evidence is captured through Azure ML Studio and
the project repository.

Evidence includes:

- Completed pipeline graph and component execution.
- Registered pipeline component and version.
- Batch Endpoint and deployment configuration.
- Successful Batch Endpoint invocation.
- Endpoint-created pipeline job.
- Scheduled pipeline execution.
- Disabled final schedule state.

## 8. Repository Organization

The repository separates orchestration code, pipeline components,
source scripts, notebooks, documentation, and evidence.

The notebook coordinates the workflow and documents execution.
Python modules encapsulate reusable pipeline and operationalization
logic.

## 9. Security and Reproducibility

- Credentials and secrets must not be committed.
- Local configuration and generated outputs are excluded from Git.
- Data is referenced through a versioned MLTable asset.
- Azure resource operations are explicitly invoked rather than
  executed as side effects of importing Python modules.
- Screenshots and documentation should be reviewed and redacted
  before publication.
