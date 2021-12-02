'''This component is for model evaluation, model validation and model registration'''
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple
'''This component is for model evaluation, model validation and model registration'''
@component(base_image="python:3.9", output_component_file="./yaml/model_E_V_R.yaml",
           packages_to_install=["sklearn", "pandas","google-api-python-client","gcsfs", "google-cloud",
                                "google-cloud-storage","argparse","google-cloud-aiplatform==1.1.1",
                                "xgboost","numpyencoder"])
def model_E_V_R(model:Input[Model],MODEL_BUCKET_NAME: str, BUCKET_NAME_THRESHOLD: str, BUCKET_NAME_TESTING: str,
                metrics:Output[Metrics])-> NamedTuple("Outputs", [("check", bool),("new_model_path",str)]):
    """
       This function is for getting predictions and F1-score from new model and then validating the model by comparing
       it to previously validated model and then saving model metadata to Vertex AI Metadata service.
       Inputs: model                  - recently trained model
               MODEL_BUCKET_NAME      - bucket where recently trained model is saved as model.bst
               BUCKET_NAME_THRESHOLD  - bucket containing F1 score of deployed model
               BUCKET_NAME_TESTING    - path to latest test data which is in a gcs bucket
       Outputs: flag            -  deployment decision for the model
                new_model_path  -  path to current model saved in GCS bucket
    """
    import argparse
    import numpy as np
    import pandas as pd
    import os
    from sklearn.metrics import f1_score
    from google.cloud import aiplatform
    import json
    from google.cloud import storage
    import gcsfs
    from numpyencoder import NumpyEncoder
    from xgboost import XGBClassifier
    #Getting model path
    model_path =model.uri
    #Getting Test Data
    new_test_data = pd.read_csv(BUCKET_NAME_TESTING)
    x_test_data=new_test_data.drop('class',axis=1)
    y_test_data=new_test_data['class']
    #Getting trained model
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(model_path.split("/")[2])
    blob_name = model_path.split('/')[3:]
    blob_name = "/".join(blob_name)+'.bst'
    blob = bucket.blob(blob_name)
    model_name = model_path.split("/")[-1]
    model_name = model_name+'.bst'
    blob.download_to_filename(model_name)
    model3 = XGBClassifier()
    model3.load_model(model_name)
    #Getting the predictions
    y_pred = model3.predict(x_test_data)
    model_metrics=f1_score(y_test_data,y_pred,average=None)
    #Saving model metrics in a Dictionary
    model_metrics_avg=(model_metrics[0]+model_metrics[1])/2
    model_metric_dict = {}
    model_metric_dict['F1-score'] = model_metrics_avg
    #Getting object names from BUCKET_NAME_THRESHOLD
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME_THRESHOLD)
    blobs = storage_client.list_blobs(BUCKET_NAME_THRESHOLD)
    bucket_object_names=''
    #initiallizing flag variable which will we used to identify which model to deploy
    flag = False
    for blob in blobs:
        bucket_object_names=bucket_object_names+'/'+blob.name

    #Checking if BUCKET_NAME_THRESHOLD is empty or not
    #If it's empty then pipeline is running for first time
    if bucket_object_names=='':
        #Checking if F1 score is greater than 0.6
        if model_metrics_avg>0.6:
            flag = True
            json_file_name = "model.json"
            #Saving metric Dictionary in json file
            with open(json_file_name, 'w') as file:
                json.dump(model_metric_dict, file, cls=NumpyEncoder)
            #Saving json file to GCS bucket
            storage_client = storage.Client()
            bucket = storage_client.bucket(BUCKET_NAME_THRESHOLD)
            bucket.blob(json_file_name).upload_from_filename(json_file_name)
            #Saving model to GCS bucket 
            storage_client = storage.Client()
            bucket = storage_client.bucket(MODEL_BUCKET_NAME)
            bucket.blob('model.bst').upload_from_filename(model_name)
            new_model_path = 'gs://'+MODEL_BUCKET_NAME+'/model.bst'
            #Saving metadata in Vertex AI metadata service
            metrics.log_metric("new-model-F1-Score-avg",model_metrics_avg)
            metrics.log_metric("deployed-model-F1-Score-avg","no model deployed yet")
            #metrics.log_metric("newly-uploaded-model",model_name)
            metrics.log_metric("deployment-status","true")
            metrics.log_metric("framework", "XGB Classifier")
            metrics.log_metric("dataset_size", len(new_test_data))
        else:
            flag = False
            #Saving model to GCS bucket 
            storage_client = storage.Client()
            bucket = storage_client.bucket(MODEL_BUCKET_NAME)
            bucket.blob('model.bst').upload_from_filename(model_name)
            new_model_path = 'gs://'+MODEL_BUCKET_NAME+'/model.bst'
            #Saving metadata in Vertex AI metadata service
            metrics.log_metric("new-model-F1-Score-avg",model_metrics_avg)
            metrics.log_metric("deployed-model-F1-Score-avg","no model deployed yet")
            #metrics.log_metric("newly-uploaded-model",model_name)
            metrics.log_metric("deployment-status","false")
            metrics.log_metric("framework", "XGB Classifier")
            metrics.log_metric("dataset_size", len(new_test_data))
            
    else:
        #Getting model.json file from 'BUCKET_NAME_THRESHOLD' bucket
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(BUCKET_NAME_THRESHOLD)
        blob = bucket.blob('model.json')
        # Download the contents of the blob as a string and then parse it using json.loads() method
        data = json.loads(blob.download_as_string(client=None))
        print(data)
        print(type(data))
        print(list(data.values())[0])
        deployed_model_threshold = list(data.values())[0]
        
        #Checking if F1-score of new model is greater or smaller than Deployed model's F1-score  
        if  model_metrics_avg < deployed_model_threshold:
            #F1-score is smaller so new_model will not be deployed
            flag = False
            #Saving Validated model to GCS bucket 
            storage_client = storage.Client()
            bucket = storage_client.bucket(MODEL_BUCKET_NAME)
            bucket.blob('model.bst').upload_from_filename(model_name)
            new_model_path = 'gs://'+MODEL_BUCKET_NAME+'/model.bst'
            #Saving metadata in Vertex AI metadata service
             #new_model_path = 'gs://'+MODEL_BUCKET_NAME+'/model.bst'
            metrics.log_metric("new-model-F1-Score-avg",model_metrics_avg)
            metrics.log_metric("deployed-model-F1-Score-avg",deployed_model_threshold)
            #metrics.log_metric("newly-uploaded-model",model_name)
            metrics.log_metric("deployment-status","false")
            metrics.log_metric("framework", "XGB Classifier")
            metrics.log_metric("dataset_size", len(new_test_data))

        else:
            #F1-score is greater so new_model will be deployed
            flag = True
            json_Filename = "model.json"
            #Updating metric json file with new F1-score
            with open(json_Filename, 'w') as file:  
                json.dump(model_metric_dict, file, cls=NumpyEncoder)
            bucket.blob(json_Filename).upload_from_filename(json_Filename)
            #Saving Validated model to GCS bucket 
            storage_client = storage.Client()
            bucket = storage_client.bucket(MODEL_BUCKET_NAME)
            bucket.blob('model.bst').upload_from_filename(model_name)
            new_model_path = 'gs://'+MODEL_BUCKET_NAME+'/model.bst'
            #Saving metadata in Vertex AI metadata service
            metrics.log_metric("new-model-F1-Score-avg",model_metrics_avg)
            metrics.log_metric("deployed-model-F1-Score-avg",deployed_model_threshold)
            #metrics.log_metric("newly-uploaded-model",model_name)
            metrics.log_metric("deployment-status","true")
            metrics.log_metric("framework", "XGB Classifier")
            metrics.log_metric("dataset_size", len(new_test_data))
        
    return (flag,new_model_path,)
