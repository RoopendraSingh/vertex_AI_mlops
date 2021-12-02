"""
This config file contains the different GCS buckets used and fLags
"""

AUTOML_PARAM = 0
VIZIER_PARAM = 0

PROJECT_ID = "advanced-analytics-engineering"
PROJECT_NUMBER = "844226895177"

BUCKET_NAME_PIPELINE_ARTIFACTS = "mlops-pipelines-artifacts"
BUCKET_NAME_RAW_DATA = "mlops-data-versioning"
BUCKET_NAME_SERIALIZED_PIPELINE = "mlops-pipelines-artifacts"
BUCKET_NAME_TRASFORMED_DATA = "mlops-transformed-data"
BUCKET_NAME_MODEL = "mlops-trained-model"
BUCKET_NAME_THRESHOLD = "podium_mlops_model_metrics"
MODEL_BUCKET_NAME = "mlops_model"
AUTOML_BUCKET_NAME = "mlops-automl-model"

REGION = "us-central1"
PIPELINE_NAME = "mlops-customer-churn-pilot"
DISPLAY_NAME = PIPELINE_NAME
PIPELINE_ROOT = f"gs://{BUCKET_NAME_PIPELINE_ARTIFACTS}/pipeline_root/"
