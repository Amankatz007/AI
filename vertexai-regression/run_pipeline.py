import os
from kfp import compiler
from google.cloud import aiplatform
import pipeline

# Load Environment Variables from Cloud Build
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

# Define your BQ Query, Target, and Quality Threshold
# Example: Select specific columns to avoid loading massive unneeded features into pandas
BQ_QUERY = f"""
    SELECT WIND, IND, RAIN, IND1, T_MAX, IND_2, T_MIN, T_MIN_G, T_MAX_CLEANSED, T_MAX_CLEANSED_FROM_RAW, T_MAX_FLOAT, T_MIN_CLEANSED 
    FROM `gen-learning-492714.ml_data.training_data2` 
    
"""
TARGET_COLUMN = "WIND"
MSE_THRESHOLD = 150.0  # Only deploy if Mean Squared Error is less than this!

# 1. Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=pipeline.regression_pipeline,
    package_path="regression_pipeline.json"
)

# 2. Initialize Vertex AI client
aiplatform.init(project=PROJECT_ID, location=REGION)

# 3. Submit the Pipeline Job
job = aiplatform.PipelineJob(
    display_name="bq-automated-regression-job",
    template_path="regression_pipeline.json",
    pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root",
    parameter_values={
        "project_id": PROJECT_ID,
        "bq_query": BQ_QUERY,
        "target_column": TARGET_COLUMN,
        "mse_threshold": MSE_THRESHOLD,
        "region": REGION
    },
    enable_caching=False # Set to True in production to save money if data hasn't changed
)

job.submit()
print("Pipeline with BigQuery extraction submitted successfully!")