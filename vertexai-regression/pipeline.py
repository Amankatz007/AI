import kfp
from kfp import dsl
from kfp.dsl import component, Output, Model, Metrics
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp

# 1. Training Component (Reads from BigQuery)
@component(
    packages_to_install=[
        "scikit-learn", "pandas", "joblib", 
        "google-cloud-storage", "google-cloud-bigquery", "db-dtypes"
    ],
    base_image="python:3.9"
)
def train_regression_model(
    bq_project_id: str,
    bq_query: str,
    target_column: str,
    model_output: Output[Model],
    metrics: Output[Metrics]
) -> float:
    # All imports must be inside the component function for KFP
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import joblib
    import os

    # 1. Load Data from BigQuery
    print(f"Initializing BigQuery Client for project: {bq_project_id}")
    bq_client = bigquery.Client(project=bq_project_id)
    
    print(f"Executing query: {bq_query}")
    df = bq_client.query(bq_query).to_dataframe()
    print(f"Loaded {len(df)} rows from BigQuery.")

    # 2. Preprocess Data
    df = df.dropna() # Basic cleaning (Add your own logic here)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # 3. Train Model
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate Model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    metrics.log_metric("MSE", mse)
    print(f"Model trained with MSE: {mse}")

    # 5. Save Model (Vertex AI expects exactly 'model.joblib')
    os.makedirs(model_output.path, exist_ok=True)
    model_path = os.path.join(model_output.path, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Return MSE so the pipeline can use it in a deployment condition
    return float(mse)


# 2. Pipeline Definition
@dsl.pipeline(
    name="bq-automated-regression-pipeline",
    description="E2E Regression Pipeline reading from BigQuery with Deployment Condition"
)
def regression_pipeline(
    project_id: str,
    bq_query: str,
    target_column: str,
    mse_threshold: float,
    region: str = "us-central1"
):
    # Step A: Train the model using BigQuery Data
    # Upgraded machine specs to handle BigQuery to Pandas memory overhead
    train_task = train_regression_model(
        bq_project_id=project_id,
        bq_query=bq_query,
        target_column=target_column
    ).set_cpu_limit('8').set_memory_limit('32G')

    # Step B: Deployment Condition (Only deploy if MSE < Threshold)
    with dsl.Condition(train_task.output < mse_threshold, name="check-model-quality"):
        
        # 1. Upload to Vertex AI Model Registry
        upload_task = ModelUploadOp(
            project=project_id,
            location=region,
            display_name="bq-regression-model",
            unmanaged_container_model=train_task.outputs["model_output"],
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        )

        # 2. Create Endpoint (or use existing)
        endpoint_task = EndpointCreateOp(
            project=project_id,
            location=region,
            display_name="bq-regression-endpoint"
        )

        # 3. Deploy Model to Endpoint
        deploy_task = ModelDeployOp(
            model=upload_task.outputs["model"],
            endpoint=endpoint_task.outputs["endpoint"],
            dedicated_resources_machine_type="n1-standard-2",
            dedicated_resources_min_replica_count=1,
            dedicated_resources_max_replica_count=1
        )