#Intall the pre-requisite

#API's enable
gcloud services enable aiplatform.googleapis.com cloudbuild.googleapis.com storage.googleapis.com bigquery.googleapis.com

#Make gcs bucket
gsutil mb -l us-central1 gs://<YOUR_PROJECT_ID>-vertex-pipelines
