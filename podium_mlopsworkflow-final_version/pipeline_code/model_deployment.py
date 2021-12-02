#This component is for saving model to Vertex AI, creating an endpoint and deploying model to endpoint
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple
'''This component is for saving model to Vertex AI, creating an endpoint and deploying model to endpoint'''
@component(base_image="python:3.9", output_component_file="./yaml/model_deployment.yaml",packages_to_install=["google-api-python-client" ,"gcsfs", "google-cloud","google-cloud-storage","argparse","google-cloud-aiplatform==1.1.1","typing"])
def model_deployment(model_path: str, flag: bool, PROJECT_ID: str, PROJECT_NUMBER: str, REGION: str, PIPELINE_NAME: str,
                     metrics:Output[Metrics])-> NamedTuple("Outputs", [("endpoint_name", str)]):
    '''
       This function will upload the model to Vertex AI then deploy the model to endpoint, if endpoint doesn't exist
       then it will create the endpoint and then deploy the model.
       INPUTS: model_path - model path from model_E_V_R function
               flag       - deployment decision
       OUTPUTS: endpoint_name  - name of the endpoint
    '''
    from google.cloud.aiplatform import gapic as aip
    from google.cloud import aiplatform
    from typing import Dict, Optional
    #from google.cloud import aiplatform_v1
    project_id = PROJECT_ID
    project_region = REGION
    #gcs_bucket = model_path.split("/model.bst")[0]
    display_name = PIPELINE_NAME + model_path.split("/")[-2]
    saved_model_path = model_path.split("/model.bst")[0]
    serving_container_image_uri = 'us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest'
    PROJECT_NUMBER = PROJECT_NUMBER
    flag=flag
    #Function to upload model to Vertex AI
    def upload_model(
        project: str,
        location: str,
        display_name: str,
        serving_container_image_uri: str,
        artifact_uri: str,
        sync: bool = True,):
        aiplatform.init(project=project, location=location)
        model = aiplatform.Model.upload(
          display_name=display_name,
          artifact_uri=artifact_uri,
          serving_container_image_uri=serving_container_image_uri,
          sync=sync,)
        model.wait()
        print(model.display_name)
        print(model.resource_name)
        return model
    #Function to create endpoint
    def create_endpoint(project_id: str, display_name: str, project_region: str, sync: bool = True,):
        aiplatform.init(project=project_id, location=project_region)
        endpoint = aiplatform.Endpoint.create(
          display_name=display_name, project=project_id, location=project_region,)
        print(endpoint.display_name)
        print(endpoint.resource_name)
        return endpoint
    #Function to deploy a model
    def deploy_model_with_dedicated_resources(
        project: str,
        location: str,
        model_name: str,
        machine_type: str,
        PROJECT_NUMBER: str,
        deployed_model_display_name: Optional[str] = None,
        endpoint: Optional[aiplatform.Endpoint] = None,
        traffic_split: Optional[Dict[str, int]] = None,
        min_replica_count:int = 1,
        max_replica_count:int = 1,
        sync: bool = True,):
        aiplatform.init(project=project, location=location)
        model = aiplatform.Model(model_name=model_name)
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=deployed_model_display_name,
            traffic_split={"0":100},
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            sync=True,)
        model.wait()
        print(model.display_name)
        print(model.resource_name)
        return model
    #Uploading validated model to Vertex AI
    model=upload_model(
        project_id,project_region,display_name,serving_container_image_uri,saved_model_path)
    model_name=model.resource_name
    aiplatform.init(project=project_id, location=project_region)
    api_endpoint = f"{project_region}-aiplatform.googleapis.com"  # @param {type:"string"}
    client_options = {"api_endpoint": api_endpoint}
    client = aip.EndpointServiceClient(client_options=client_options)
    endpoints_all = client.list_endpoints(parent=f'projects/{PROJECT_NUMBER}/locations/{project_region}')
    #endpoint_id = ''
    endpoint_exists = False
    #deployed_model = ''
    #endpoint_name = display_name
    for oneEnd in endpoints_all:
        if oneEnd.display_name == display_name:
            print(oneEnd.display_name)
            print(oneEnd.name)
            endpoint_exists = True
            endpoint_id = oneEnd.name
            for oneModel in oneEnd.deployed_models:
                print(oneModel.model)
                deployed_model_id = oneModel.model.split('/')[-1]
                deployed_model = oneModel.model
    if flag == True and endpoint_exists == True:
        aiplatform.init(project=project_id, location=project_region)
        endpoint1=aiplatform.Endpoint(endpoint_name=endpoint_id, project= project_id, location= project_region,)
        endpoint1.undeploy_all()
        endpoint_name = endpoint_id
        deploy_model_with_dedicated_resources(
            project=project_id,
            location=project_region,
            model_name=model_name,
            machine_type='n1-standard-4',
            PROJECT_NUMBER=PROJECT_NUMBER,
            deployed_model_display_name=display_name,
            endpoint = endpoint1,
            traffic_split={"0":100},
            min_replica_count = 1,
            max_replica_count = 1,
            sync= True,
        )
        metrics.log_metric("previously-deployed-model",deployed_model)
        metrics.log_metric("newly-deployed-model",model_name)
        metrics.log_metric("current-model",model_name)
        metrics.log_metric("endpoint", endpoint_name)
    elif flag == True and endpoint_exists == False:
        #Creating endpoint
        endpoint=create_endpoint(project_id, display_name, project_region)
        endpoint_name=endpoint.resource_name 
        #Deploying Validated model to endpoint
        deploy_model_with_dedicated_resources(
            project=project_id,
            location=project_region,
            model_name=model_name,
            machine_type='n1-standard-4',
            PROJECT_NUMBER=PROJECT_NUMBER,
            deployed_model_display_name=display_name,
            endpoint = endpoint,
            traffic_split={"0":100},
            min_replica_count = 1,
            max_replica_count = 1,
            sync= True,
        )
        metrics.log_metric("previously-deployed-model","no model deployed yet")
        metrics.log_metric("newly-deployed-model",model_name)
        metrics.log_metric("current-model",model_name)
        metrics.log_metric("endpoint", endpoint_name)
    elif flag == False and endpoint_exists == True:
        endpoint_name = endpoint_id
        metrics.log_metric("previously-deployed-model",deployed_model)
        metrics.log_metric("newly-deployed-model","not-deployed")
        metrics.log_metric("current-model",model_name)
        metrics.log_metric("endpoint", endpoint_name)
    elif flag == False and endpoint_exists == False:
        endpoint_name = 'none'
        metrics.log_metric("previously-deployed-model","no model deployed yet")
        metrics.log_metric("newly-deployed-model","not-deployed")
        metrics.log_metric("current-model",model_name)
        metrics.log_metric("endpoint", endpoint_name)
    return (endpoint_name, )
