"""Installing the dependencies"""
#!pip3 install {USER_FLAG} google-cloud-aiplatform --upgrade
#!pip3 install {USER_FLAG} kfp==1.8.6 google-cloud-pipeline-components==0.1.7 --upgrade
import warnings 
warnings.filterwarnings('ignore')
import sys

from kfp.v2 import compiler
from kfp import dsl
from kfp.v2.google.client import AIPlatformClient
from google.cloud import storage

#importing the components
from data_ingestion import data_ingestion
from data_validation import data_validation
from data_transformation import data_transformation
from model_tuning import model_tuning
from model_tuning_vizier import model_tuning_vizier
from model_training import model_training
from model_training_vizier import model_training_vizier
from model_evaluation import model_E_V_R
from model_evaluation_vizier import model_evaluation_vizier
from model_deployment import model_deployment
from automl import auto_ml_exp
import config

"""This file compiles the the pipline and triggers the vertexAI pipelines"""
    
# Checking the arguments
if len(sys.argv) == 2:
    commit_id = sys.argv[1]
    NOTEBOOK_PATH = f"https://gitlab.qdatalabs.com/applied-ai/applied-ai_west/podium_mlopsworkflow/-/commit/{commit_id}"

elif len(sys.argv) == 3:
    config.AUTOML_PARAM = int(sys.argv[1])
    config.VIZIER_PARAM = int(sys.argv[2])
    NOTEBOOK_PATH = 'NA'
else:
    NOTEBOOK_PATH = 'NA'

#Below condition runs default pipeline 
if config.AUTOML_PARAM == 0 and config.VIZIER_PARAM == 0:
    @dsl.pipeline(name=config.PIPELINE_NAME,pipeline_root=config.PIPELINE_ROOT)
    def pipeline():
        download_data = data_ingestion(config.BUCKET_NAME_RAW_DATA,NOTEBOOK_PATH)
        validation_data = data_validation(download_data.outputs["gcsFile"]).after(download_data)
        with dsl.Condition(validation_data.outputs["data_validation_sucess_output_flag"] == "true",
                           name="data_validation_success_check",):
            data_processing = data_transformation(download_data.outputs["gcsFile"], 
                                         config.BUCKET_NAME_TRASFORMED_DATA).after(validation_data)
            hpt = model_tuning(data_processing.outputs["train_data"],config.BUCKET_NAME_MODEL).after(data_processing)
            training_hyper = model_training(hpt.outputs["tuned_hp"],data_processing.outputs["train_data"]).after(hpt)
            testing_hyper = model_E_V_R(training_hyper.outputs["model"],config.MODEL_BUCKET_NAME, config.BUCKET_NAME_THRESHOLD, 
                                data_processing.outputs["test_data"]).after(training_hyper)
            deploy = model_deployment(testing_hyper.outputs["new_model_path"], testing_hyper.outputs["check"],config.PROJECT_ID, config.PROJECT_NUMBER, config.REGION, 
                                          config.PIPELINE_NAME).after(testing_hyper)
    #Below condition runs default pipeline with AutoML component        
elif config.AUTOML_PARAM == 1 and config.VIZIER_PARAM == 0:
    @dsl.pipeline(name=config.PIPELINE_NAME,pipeline_root=config.PIPELINE_ROOT)
    def pipeline():
        download_data = data_ingestion(config.BUCKET_NAME_RAW_DATA,NOTEBOOK_PATH)
        validation_data = data_validation(download_data.outputs["gcsFile"]).after(download_data)
        with dsl.Condition(validation_data.outputs["data_validation_sucess_output_flag"] == "true",
                           name="data_validation_success_check",):
            data_processing = data_transformation(download_data.outputs["gcsFile"], 
                                         config.BUCKET_NAME_TRASFORMED_DATA).after(validation_data)
            automl = auto_ml_exp(data_processing.outputs["train_data"],data_processing.outputs["test_data"],config.AUTOML_BUCKET_NAME, 
                                 config.PROJECT_ID, config.REGION).after(data_processing)
            hpt = model_tuning(data_processing.outputs["train_data"],config.BUCKET_NAME_MODEL).after(data_processing)
            training_hyper = model_training(hpt.outputs["tuned_hp"],data_processing.outputs["train_data"]).after(hpt)
            testing_hyper = model_E_V_R(training_hyper.outputs["model"],config.MODEL_BUCKET_NAME, config.BUCKET_NAME_THRESHOLD, 
                                data_processing.outputs["test_data"]).after(training_hyper)
            deploy = model_deployment(testing_hyper.outputs["new_model_path"], testing_hyper.outputs["check"],config.PROJECT_ID, config.PROJECT_NUMBER, config.REGION, 
                                          config.PIPELINE_NAME).after(testing_hyper)
#Below condition runs default pipeline with Vertex Vizier component           
elif config.AUTOML_PARAM == 0 and config.VIZIER_PARAM == 1:
    # Define a pipeline and create a task from a component:
    @dsl.pipeline(name=config.PIPELINE_NAME,pipeline_root=config.PIPELINE_ROOT)
    def pipeline():
        download_data = data_ingestion(config.BUCKET_NAME_RAW_DATA,NOTEBOOK_PATH)
        validation_data = data_validation(download_data.outputs["gcsFile"]).after(download_data)
        with dsl.Condition(validation_data.outputs["data_validation_sucess_output_flag"] == "true",
                           name="data_validation_success_check",):
            data_processing = data_transformation(download_data.outputs["gcsFile"], 
                                         config.BUCKET_NAME_TRASFORMED_DATA).after(validation_data)
            hpt = model_tuning(data_processing.outputs["train_data"],config.BUCKET_NAME_MODEL).after(data_processing)
            hpt_vizier = model_tuning_vizier(data_processing.outputs["train_data"],
                                         config.BUCKET_NAME_MODEL,config.REGION,config.PROJECT_ID).after(data_processing)
            training_hyper = model_training(hpt.outputs["tuned_hp"],data_processing.outputs["train_data"]).after(hpt)
            training_vizier = model_training_vizier(hpt_vizier.outputs["tuned_hp"],
                                                    data_processing.outputs["train_data"],).after(hpt_vizier)
            testing_hyper = model_E_V_R(training_hyper.outputs["model"],config.MODEL_BUCKET_NAME, config.BUCKET_NAME_THRESHOLD, 
                                data_processing.outputs["test_data"]).after(training_hyper)
            testing_vizier = model_evaluation_vizier(training_vizier.outputs["model"],config.MODEL_BUCKET_NAME, config.BUCKET_NAME_THRESHOLD, 
                                data_processing.outputs["test_data"]).after(training_vizier)
            deploy = model_deployment(testing_hyper.outputs["new_model_path"], testing_hyper.outputs["check"],config.PROJECT_ID, config.PROJECT_NUMBER, config.REGION, 
                                          config.PIPELINE_NAME).after(testing_hyper)

#Below condition runs default pipeline with AutoML and Vertex Vizier component                   
elif config.AUTOML_PARAM == 1 and config.VIZIER_PARAM == 1:
    @dsl.pipeline(name=config.PIPELINE_NAME,pipeline_root=config.PIPELINE_ROOT)
    def pipeline():
        download_data = data_ingestion(config.BUCKET_NAME_RAW_DATA,NOTEBOOK_PATH)
        validation_data = data_validation(download_data.outputs["gcsFile"]).after(download_data)
        with dsl.Condition(validation_data.outputs["data_validation_sucess_output_flag"] == "true",
                           name="data_validation_success_check",):
            data_processing = data_transformation(download_data.outputs["gcsFile"], 
                                         config.BUCKET_NAME_TRASFORMED_DATA).after(validation_data)
            automl = auto_ml_exp(data_processing.outputs["train_data"],data_processing.outputs["test_data"],config.AUTOML_BUCKET_NAME, 
                                 config.PROJECT_ID, config.REGION).after(data_processing)
            hpt = model_tuning(data_processing.outputs["train_data"],config.BUCKET_NAME_MODEL).after(data_processing)
            hpt_vizier = model_tuning_vizier(data_processing.outputs["train_data"],
                                         config.BUCKET_NAME_MODEL,config.REGION,config.PROJECT_ID).after(data_processing)
            training_hyper = model_training(hpt.outputs["tuned_hp"],data_processing.outputs["train_data"]).after(hpt)
            training_vizier = model_training_vizier(hpt_vizier.outputs["tuned_hp"],
                                                    data_processing.outputs["train_data"],).after(hpt_vizier)
            testing_hyper = model_E_V_R(training_hyper.outputs["model"],config.MODEL_BUCKET_NAME, config.BUCKET_NAME_THRESHOLD, 
                                data_processing.outputs["test_data"]).after(training_hyper)
            testing_vizier = model_evaluation_vizier(training_vizier.outputs["model"],config.MODEL_BUCKET_NAME, config.BUCKET_NAME_THRESHOLD, 
                                data_processing.outputs["test_data"]).after(training_vizier)
            deploy = model_deployment(testing_hyper.outputs["new_model_path"], testing_hyper.outputs["check"],config.PROJECT_ID, config.PROJECT_NUMBER, config.REGION, 
                                          config.PIPELINE_NAME).after(testing_hyper)
    #storing the latest compiled Json to the GCS bucket
def pipeline_to_storage():
    storage_client = storage.Client()
    pipeline_json_file = 'mlops_internal_pipeline.json'
    bucket = storage_client.bucket(config.BUCKET_NAME_SERIALIZED_PIPELINE )
    bucket.blob(pipeline_json_file).upload_from_filename(pipeline_json_file)
    print("Pipeline compilation uploaded successfully")

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline, package_path="mlops_internal_pipeline.json"
    )
    pipeline_to_storage()
    api_client = AIPlatformClient(project_id=config.PROJECT_ID,region = config.REGION)
    
    response = api_client.create_run_from_job_spec(
        job_spec_path="mlops_internal_pipeline.json",
        enable_caching = False,
        pipeline_root=config.PIPELINE_ROOT
    )
