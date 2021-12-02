"""
This component is for AutoML

"""
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple
@component(base_image="python:3.9", output_component_file="./yaml/automl.yaml",
           packages_to_install=["numpy","pandas","google-cloud",
                                "google-cloud-storage","google-api-python-client",
                                "google-cloud-aiplatform","fsspec", "gcsfs"])
def auto_ml_exp(trainfile: str,testfile: str, auto_ml_bucket:str, 
                project: str, location: str, smetrics: Output[Metrics]):   
    import pandas as pd
    import numpy as np
    from google.cloud import storage
    from google.cloud import aiplatform
    import fsspec
    """
    It runs AutoML tables
    Inputs: train file path,
            test file path,
            region,project_id
    outputs: shows f1-score as metrics
    """
    #take train and test and merge it into one df with additional column for train,test split.
    # UNASSIGNED: means automl take it as train or validation data
    def train_test_merge (train_data, test_data,gcs_bucket):
        train= pd.read_csv(train_data)
        test= pd.read_csv(test_data)
        train['split_cat']= 'UNASSIGNED'
        test['split_cat']= 'TEST'
        data= train.append(test)
        data_path = f"gs://{gcs_bucket}/data_auto_ml.csv"
        data.to_csv(data_path, index = False)
        test_data_len= len(test)
        return data_path, test_data_len    
    #load the whole dataset to vertex AI and fetch the dataset ID. 
    def create_and_import_dataset_tabular_gcs(
        display_name: str, 
        project: str, 
        location: str, 
        gcs_source,):
        
        aiplatform.init(project=project, location=location)
        dataset = aiplatform.TabularDataset.create(
            display_name=display_name, gcs_source=gcs_source,)
        dataset.wait()
        res_name= dataset.resource_name
        print(f'\tname: "{res_name}"')
        #e.g = projects/844226895177/locations/us-central1/datasets/5221985340587245568
        dataset_id = res_name.split('/')[-1]
        return (dataset_id)
    #create the autoML training job
    def create_training_pipeline_tabular_classification(
        project: str,
        display_name: str,
        dataset_id: int,
        location: str,
        target_col: str,
        predefined_split_column_name: str= 'split_cat',
        model_display_name: str = None,
        #training_fraction_split: float = 0.75,(no need in case of explicitly mentioned in data)
        #validation_fraction_split: float = 0.10,
        #test_fraction_split: float = 0.15,
        budget_milli_node_hours: int = 100,#(how many hours are allowed for training)
        disable_early_stopping: bool = False,
        sync:bool = True):
        
        aiplatform.init(project=project, location=location)
        
        tabular_classification_job = aiplatform.AutoMLTabularTrainingJob(
            display_name=display_name,
            optimization_prediction_type= 'classification')
        my_tabular_dataset = aiplatform.TabularDataset(dataset_id)
        print(my_tabular_dataset)
        model = tabular_classification_job.run(
            dataset=my_tabular_dataset,
            predefined_split_column_name=predefined_split_column_name,
            budget_milli_node_hours=budget_milli_node_hours,
            model_display_name=model_display_name,
            disable_early_stopping=disable_early_stopping,
            sync=sync,
            target_column=target_col,)
        model.wait()
        return model.resource_name
    #this function is for getting the f1-score as model metrics
    def getting_f1_score(
        model_resource_name,
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",):
        
        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.ModelServiceClient(client_options=client_options)
        path= client.list_model_evaluations(parent = model_resource_name)
        for x in path:
            metric= x.metrics['confidenceMetrics']
            for y in metric[51].items(): #for 0.5 threshold
                if y[0]== 'f1Score':
                    f1_score= y[1]
                    break
        return (f1_score)
    data_path, test_data_len= train_test_merge(trainfile, testfile, auto_ml_bucket)
    data_id= create_and_import_dataset_tabular_gcs('automl_data',project,location,data_path)
    model_res_name= create_training_pipeline_tabular_classification(project,
                                                           'automl_model',
                                                           data_id,
                                                           location,
                                                           'class',)
    f1_score_= getting_f1_score(model_res_name)
    smetrics.log_metric("F1-Score_1",f1_score_)
    smetrics.log_metric("framework", "XGB Classifier")
    smetrics.log_metric("dataset_size", test_data_len)
    return None
