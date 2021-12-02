# Installing Dependencies
# ! pip3 install google-cloud-aiplatform==1.0.0 --upgrade
# ! pip3 install kfp google-cloud-pipeline-components==0.1.1 --upgrade

#importing dependencies
import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from kfp.v2.google.client import AIPlatformClient

from google.cloud import aiplatform
from google_cloud_pipeline_components import aiplatform as gcc_aip

from typing import NamedTuple

PIPELINE_NAME = "mlops-podium-internal-demo-prediction-serving"
PIPELINE_ROOT = "gs://prediction_serving"

PROJECT_ID = 'advanced-analytics-engineering'
REGION = "us-central1"

BUCKET_NAME_PIPELINE_ARTIFACTS = "gs://prediction_serving"  # path to save Pipeline artifacts
BUCKET_NAME_PRED_DATA = "gs://batch-pred-data/batch_pred_data.csv"  # Path to Input Prediction data
BUCKET_NAME_TRAIN_DATA = 'gs://mlops-transformed-data/train.csv' # Path to Train Data after transformation
MODEL_BUCKET_NAME = "mlops_validated_model" # Path to FInal Validated .bst model file
BUCKET_NAME_SERIALIZED_PIPELINE = "prediction_serving"

# Data Validation Componenet
@component(base_image="python:3.9", output_component_file="data_validation.yaml",
           packages_to_install=["numpy", "pandas","dvc","gcsfs", "google-cloud","google-cloud-storage"])
def data_validation(pred_data_path: str, train_data_path: str):
    from google.cloud import storage
    import pandas as pd
    import numpy as np

    print("read the prediction dataset")
    prediction_data = pd.read_csv(pred_data_path)
    print("read the training dataset")
    train_data = pd.read_csv(train_data_path)
    train_features = train_data.drop(['class'], axis=1)
    
    # Get a Dictionary containing the pairs of column names & data type objects.
    dataTypeDict_1 = dict(prediction_data.dtypes)
    dataTypeDict_2 = dict(train_features.dtypes)
    if (dataTypeDict_1 == dataTypeDict_2):
        print(" --- PASS : Prediction Data Schema matches with Training Data --- ")
    else: 
        print(" --- ALERT : Input data is not proper --- ")
    # obtaining the number of columns
    no_col = prediction_data.shape[1]
    
    if (no_col) != 97:
        print(" --- ALERT : Input data is not proper --- ")
    else: 
        print("PASS | all columns are available")
        print("number of columns : ", no_col)
        

# Data Transformation Component
@component(base_image="python:3.9", output_component_file="data_transformed.yaml",
           packages_to_install=["numpy", "pandas","dvc","gcsfs", "google-cloud","google-cloud-storage"])
def data_transformation(pred_data_path: str, bucket_tranformed_data_path: str)-> NamedTuple("Outputs", [("transformed_data_path", str)]):
    import pandas as pd
    import numpy as np
    from google.cloud import storage

    # Read the data
    print("load the prediction dataset")
    prediction_data = pd.read_csv(pred_data_path)
    
    #dropping the NAs
    print("Total Na rows present in dataset are - ",prediction_data.isnull().values.ravel().sum())
    prediction_data.dropna(inplace=True)
    prediction_data.reset_index(inplace=True, drop=True)
    print("Dropped all the NAs")
    
    tranformed_data_path = f"gs://{bucket_tranformed_data_path}/transformed_pred_data.csv"
    prediction_data.to_csv(tranformed_data_path, index = False)    
    return (tranformed_data_path, )

# Batch Prediction COmponent
@component(base_image="python:3.9", output_component_file="batch_prediction.yaml",
           packages_to_install=["numpy", "sklearn", "pandas","dvc","gcsfs", "google-cloud","google-cloud-storage","xgboost","google-cloud-bigquery","pyarrow"])
def batch_prediction(tranformed_data_path: str, MODEL_BUCKET_NAME: str):
    
    from xgboost import XGBClassifier
    import pandas as pd
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import classification_report
    import time
    import numpy as np
    import pickle
    from google.cloud import storage
    import gcsfs
    
    from google.cloud import bigquery
    from datetime import date
    from datetime import datetime
    import xgboost

    
    print("load the trained model")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET_NAME)
    bucket.blob('model.bst').download_to_filename('model.bst')
    
    xgb_model_latest = xgboost.XGBClassifier() # or which ever sklearn booster you're are using
    xgb_model_latest.load_model("model.bst")
    

    prediction_data = pd.read_csv(tranformed_data_path)


    #Take predictions on test dataset
    print("Prediction started on prediction dataset")
    y_test_pred=xgb_model_latest.predict(prediction_data).tolist()
    y_test_pred_prob=xgb_model_latest.predict_proba(prediction_data).tolist()
    print("Prediction done on prediction dataset successfully...")
    
    
    df_1 = pd.DataFrame(y_test_pred_prob, columns=['prob_1', 'prob_2'])
    df_2 = pd.DataFrame(y_test_pred, columns=['Churn_0_5'])
    df = pd.concat([prediction_data, df_1, df_2], axis=1)

    
    Dataset_name = "prediction_result"
    bigquery_client = bigquery.Client(project="advanced-analytics-engineering")
    dataset = bigquery_client.dataset(Dataset_name)

    today = date.today()
    now = datetime.now()
    # Month abbreviation, day and year	
    d4 = today.strftime("%b_%d_%Y")

    dt_string = now.strftime("_%H_%M_%S")
    print("date and time =", dt_string)

    table_name = str("batch_results_")+d4+dt_string
    table_ref = dataset.table(table_name)
    dataTypeDict = dict(prediction_data.dtypes)
    
    SCHEMA_data = []
    for i in dataTypeDict :
        SCHEMA_data.append(bigquery.SchemaField(i, str(dataTypeDict[i]).upper(), mode="REQUIRED"))

    
    SCHEMA_result = [
        bigquery.SchemaField("prob_1", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("prob_2", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("Churn_0_5", "INTEGER", mode="REQUIRED")
    ]
    
    SCHEMA = SCHEMA_data + SCHEMA_result

    table = bigquery.Table(table_ref, schema=SCHEMA)
    table = bigquery_client.create_table(table)
    print("write prediction results to Bigquery")
    table = Dataset_name+str('.')+table_name
    print("results are written to", table)
    # Load data to BQ
    job = bigquery_client.load_table_from_dataframe(df, table)

# Define a pipeline and create a task from a component:
@dsl.pipeline(name=PIPELINE_NAME,pipeline_root=PIPELINE_ROOT)
def pipeline():
    validate_data = data_validation(BUCKET_NAME_PRED_DATA, BUCKET_NAME_TRAIN_DATA)
    data_transform = data_transformation(BUCKET_NAME_PRED_DATA, 'prediction_serving').after(validate_data)
    prediction_result = (batch_prediction(data_transform.outputs["transformed_data_path"], MODEL_BUCKET_NAME).after(data_transform).set_cpu_limit('12').
    set_memory_limit('5G'))
    
compiler.Compiler().compile(
    pipeline_func=pipeline, package_path="mlops_internal_pipeline.json"
)

from google.cloud import storage

storage_client = storage.Client()
pipelinJsonFile = 'mlops_internal_pipeline.json'
bucket = storage_client.bucket(BUCKET_NAME_SERIALIZED_PIPELINE)
bucket.blob(pipelinJsonFile).upload_from_filename(pipelinJsonFile)
print("Pipeline compilation uploaded successfully")

api_client = AIPlatformClient(project_id=PROJECT_ID,region=REGION)

response = api_client.create_run_from_job_spec(
    job_spec_path="mlops_internal_pipeline.json",
    enable_caching = False,
    pipeline_root=PIPELINE_ROOT
)