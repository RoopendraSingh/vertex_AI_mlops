from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple
'''This component is for evaluating vizier model'''
@component(base_image="python:3.9", output_component_file="./yaml/model-evaluation-vizier.yaml",packages_to_install=["sklearn", "pandas","google-api-python-client","gcsfs", "google-cloud","google-cloud-storage","argparse","google-cloud-aiplatform==1.1.1","xgboost","numpyencoder"])
def model_evaluation_vizier(model:Input[Model],MODEL_BUCKET_NAME: str, BUCKET_NAME_THRESHOLD: str, BUCKET_NAME_TESTING: str,metrics: Output[Metrics])-> NamedTuple("Outputs", [("check", bool),("new_model_path",str)]):
    '''
          This function make predictions from vizier model and calculate F1 score
          INPUTS : model path, latest test data
          OUTPUTS : none
    '''
    import argparse
    import numpy as np
    import pandas as pd
    import os
    from sklearn.metrics import f1_score
    from google.cloud import aiplatform
    import json
    from google.cloud import storage
    import gcsfs
    import pickle
    from numpyencoder import NumpyEncoder
    from xgboost import XGBClassifier
    
    model_path = model.uri
    new_test_data = pd.read_csv(BUCKET_NAME_TESTING)
    x_test_data=new_test_data.drop('class',axis=1)
    y_test_data=new_test_data['class']
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(model_path.split("/")[2])
    blob_name = model_path.split('/')[3:]
    blob_name = "/".join(blob_name)+'.bst'
    blob = bucket.blob(blob_name)
    blob.download_to_filename("model_sklearn_viz.bst")
    model3 = XGBClassifier()
    model3.load_model('model_sklearn_viz.bst')
    model_name = 'model_sklearn_viz.bst'
    y_pred = model3.predict(x_test_data)
    model_metrics=f1_score(y_test_data,y_pred,average=None)
    metrics.log_metric("F1-Score_0",model_metrics[0])
    metrics.log_metric("F1-Score_1",model_metrics[1])
    metrics.log_metric("framework", "XGB Classifier")
    metrics.log_metric("dataset_size", len(new_test_data))

    return None

