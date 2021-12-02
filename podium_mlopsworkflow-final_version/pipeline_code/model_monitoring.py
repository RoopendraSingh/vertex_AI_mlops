# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import json
import pickle
from google.cloud import storage

## Specify constants for the monitoring job

# Region definition
REGION = "us-central1"
# Define api and endpoint
SUFFIX = "aiplatform.googleapis.com"
API_ENDPOINT = f"{REGION}-{SUFFIX}"
# Define user email to receive job updates
USER_EMAIL = "abc"  
# Training Data folder/path
preprocessedMonFolder = "gs://aae-dropzone/data/quantiphi_podium_churn.csv"
# Deployed model endpoint name
ENDPOINT_NAME = 'mlops-model-monitoring-gau'
# Define Project ID
PROJECT_ID = 'advanced-analytics-engineering'
# Define Project number
PROJECT_NUMBER = '844226895177'

# Set the trigger default threshold value
DEFAULT_THRESHOLD_VALUE = 0.001 
# Sampling rate (optional, default=.8)
LOG_SAMPLE_RATE = 0.8  
# Monitoring Interval in seconds (optional, default=3600).
MONITOR_INTERVAL = 3600  
# Specify target feature
TARGET = "event"
# Skew and drift thresholds # @param {type:"string"}
SKEW_DEFAULT_THRESHOLDS = "prev_month_int_invite_days_avg" 
SKEW_CUSTOM_THRESHOLDS = "prev_month_int_invite_days_avg:.001"  
DRIFT_DEFAULT_THRESHOLDS = "prev_month_int_invite_days_avg"
DRIFT_CUSTOM_THRESHOLDS = "prev_month_int_invite_days_avg:.001" 

### Use the following code to get the Project ID and Project number
# #  Find Project ID
# project_id_core = !(gcloud config list --format 'value(core.project)' 2>/dev/null)
# PROJECT_ID = project_id_core[0]
# print("Your project id is {}.".format(PROJECT_ID))

# # Find Project number
# project_number_core = !(gcloud config list --format 'value(core.account)' 2>/dev/null)
# PROJECT_NUMBER = project_number_core[0].split('-')[0]
# print("Your project number is {}.".format(PROJECT_NUMBER))

## Function to create model monitoring job
def create_mon_job(PROJECT_ID: str, PROJECT_NUMBER: str, REGION: str, ENDPOINT_NAME: str, preprocessedMonFolder: str, SUFFIX: str, API_ENDPOINT: str, USER_EMAIL: str, DEFAULT_THRESHOLD_VALUE: float, LOG_SAMPLE_RATE: float, MONITOR_INTERVAL: int, TARGET: str, SKEW_DEFAULT_THRESHOLDS: str,
SKEW_CUSTOM_THRESHOLDS: str, DRIFT_DEFAULT_THRESHOLDS: str, DRIFT_CUSTOM_THRESHOLDS: str):
    """
    Creates a model monitoring job on a specified endpoint. Enables functionalities to pause/delete the 
    created monitoring jobs. 
    
    Inputs:
    1. PROJECT_ID                - Name of the project being worked in
    2. PROJECT_NUMBER            - Number of the project being worked in
    3. REGION                    - Region of the model/endpoint
    4. ENDPOINT_NAME             - Name of the endpoint on which monitoring job needs to
                                   be created
    5. preprocessedMonFolder     - Versioned training data folder
    6. SUFFIX                    - extension suffix name
    7. API_ENDPOINT              - API name 
    8. USER_EMAIL                - User email ID to receive monitoring job updates
    9. DEFAULT_THRESHOLD_VALUE   - Default threshold value for alert detection
    10. LOG_SAMPLE_RATE          - Rate for sampling functionality
    11. MONITOR_INTERVAL         - Monitoring interval for every monitoring job run
    12. TARGET                   - Target feature/ Label
    13. SKEW_DEFAULT_THRESHOLDS  - Default thresholds for training-serving skew detection
    14. SKEW_CUSTOM_THRESHOLDS   - Custom thresholds for training-serving skew detection
    15. DRIFT_DEFAULT_THRESHOLDS - Default thresholds for prediction drift detection
    16. DRIFT_CUSTOM_THRESHOLDS  - Custom thresholds for prediction drift detection

    Outputs:
    Creation of a model monitoring job
    
    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1beta1.services.endpoint_service import EndpointServiceClient
    from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient
    from google.cloud.aiplatform_v1beta1.types.io import BigQuerySource, GcsSource
    from google.cloud.aiplatform_v1beta1.types.model_deployment_monitoring_job import (
        ModelDeploymentMonitoringJob, ModelDeploymentMonitoringObjectiveConfig,
        ModelDeploymentMonitoringScheduleConfig)
    from google.cloud.aiplatform_v1beta1.types.model_monitoring import (
        ModelMonitoringAlertConfig, ModelMonitoringObjectiveConfig,
        SamplingStrategy, ThresholdConfig)
    from google.protobuf import json_format
    from google.protobuf.duration_pb2 import Duration
    from google.protobuf.struct_pb2 import Value
    import copy
    
    # Initialize variables
    GcsSource_URI = []
    model_ids = []
    current_monitoring_job = ''
        
    # Helper functions for model monitoring job
    def create_monitoring_job(objective_configs):
        """
        Launches the model monitoring job. 
        The input objective_configs contains information about which features to monitor and the configuration of this monitoring job (threshold, etc)
        """

        # Create sampling configuration.
        random_sampling = SamplingStrategy.RandomSampleConfig(sample_rate=LOG_SAMPLE_RATE)
        sampling_config = SamplingStrategy(random_sample_config=random_sampling)

        # Create schedule configuration.
        duration = Duration(seconds=MONITOR_INTERVAL)
        schedule_config = ModelDeploymentMonitoringScheduleConfig(monitor_interval=duration)

        # Create alerting configuration.
        emails = ["vaidehi.joshi@quantiphi.com"]
        email_config = ModelMonitoringAlertConfig.EmailAlertConfig(user_emails=emails)
        alerting_config = ModelMonitoringAlertConfig(email_alert_config=email_config)

        # Create the monitoring job.
        endpoint = endpoint_id
        predict_schema = ""
        analysis_schema = ""
        job = ModelDeploymentMonitoringJob(
            display_name=JOB_NAME,
            endpoint=endpoint,
            model_deployment_monitoring_objective_configs=objective_configs,
            logging_sampling_strategy=sampling_config,
            model_deployment_monitoring_schedule_config=schedule_config,
            model_monitoring_alert_config=alerting_config,
            predict_instance_schema_uri=predict_schema,
            analysis_instance_schema_uri=analysis_schema,
        )
        options = dict(api_endpoint=API_ENDPOINT)
        client = JobServiceClient(client_options=options)
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        response = client.create_model_deployment_monitoring_job(
            parent=parent, model_deployment_monitoring_job=job
        )
        print("Created monitoring job:")
        print(response)
        return response
    
    def pause_monitoring_job(job):
        """Pause existing monitoring job"""
        
        client_options = dict(api_endpoint=API_ENDPOINT)
        client = JobServiceClient(client_options=client_options)
        response = client.pause_model_deployment_monitoring_job(name=job)
        print(response)

    def delete_monitoring_job(job):
        """Delete existing monitoring job"""
        
        client_options = dict(api_endpoint=API_ENDPOINT)
        client = JobServiceClient(client_options=client_options)
        response = client.delete_model_deployment_monitoring_job(name=job)
        print(response)

    def get_thresholds(default_thresholds, custom_thresholds):
        """
        Sets the threshold values at which to trigger an alert.
        Args:
            default_thresholds (list): List of features which are triggered by the default threshold value
            custom_thresholds (pair): List of features which are triggered by a custom threshold value, together with the actual custom value
        """

        thresholds = {}
        default_threshold = ThresholdConfig(value=DEFAULT_THRESHOLD_VALUE)
        for feature in default_thresholds.split(","):
            feature = feature.strip()
            thresholds[feature] = default_threshold
        for custom_threshold in custom_thresholds.split(","):
            pair = custom_threshold.split(":")
            if len(pair) != 2:
                print(f"Invalid custom skew threshold: {custom_threshold}")
                return
            feature, value = pair
            thresholds[feature] = ThresholdConfig(value=float(value))
        return thresholds

    def set_objectives(model_ids, objective_template):
        """
        Sets objectives on all models deployed on the created endpoint
        """
        objective_configs = []
        for model_id in model_ids:
            objective_config = copy.deepcopy(objective_template)
            objective_config.deployed_model_id = model_id
            objective_configs.append(objective_config)
        return objective_configs
    
    # Get GCS bucket name
    GcsSource_URI.append(preprocessedMonFolder)
    print(GcsSource_URI)
    
    # Get endpoint name
    aiplatform.init(project=PROJECT_ID,location=REGION)
    endpoint = aiplatform.Endpoint.list(filter=f'display_name={ENDPOINT_NAME}', order_by='update_time')[-1]
    endpoint_uri = endpoint.gca_resource.name
    print(endpoint_uri)
    
    # Get endpoint ID
    endpoint_id0 = endpoint_uri.split("/")[-1]
    
    # Specify monitoring job name
    JOB_NAME = f"monitoring-" + endpoint_id0
    
    # Set thresholds specifying alerting criteria for training/serving skew and create config object.
    skew_thresholds = get_thresholds(SKEW_DEFAULT_THRESHOLDS, SKEW_CUSTOM_THRESHOLDS)
    skew_config = ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
        skew_thresholds=skew_thresholds
    )

    # Set thresholds specifying alerting criteria for serving drift and create config object.
    drift_thresholds = get_thresholds(DRIFT_DEFAULT_THRESHOLDS, DRIFT_CUSTOM_THRESHOLDS)
    drift_config = ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
        drift_thresholds=drift_thresholds
    )
    
    # Specify training dataset source location (used for schema generation).
    training_dataset = ModelMonitoringObjectiveConfig.TrainingDataset(target_field=TARGET, data_format = 'csv')
    training_dataset.gcs_source = GcsSource(uris = GcsSource_URI)

    # Aggregate the above settings into a ModelMonitoringObjectiveConfig object and use
    # that object to adjust the ModelDeploymentMonitoringObjectiveConfig object.
    objective_config = ModelMonitoringObjectiveConfig(
        training_dataset=training_dataset,
        training_prediction_skew_detection_config=skew_config,
    )
    objective_template = ModelDeploymentMonitoringObjectiveConfig(
        objective_config=objective_config
    )
    client_options = dict(api_endpoint=API_ENDPOINT)
    client = EndpointServiceClient(client_options=client_options)
    endpoint_id = endpoint_uri
    response = client.get_endpoint(name=endpoint_id)
    for model in response.deployed_models:
        model_ids.append(model.id)
        
    # Find all deployed model ids on the created endpoint and set objectives for each.
    objective_configs = set_objectives(model_ids, objective_template)
    
    # Check if model monitoring job already exists for given endpoint
    client_options = dict(api_endpoint=API_ENDPOINT)
    parent = f"projects/{PROJECT_ID}/locations/us-central1"
    client = JobServiceClient(client_options=client_options)
    response = client.list_model_deployment_monitoring_jobs(parent=parent)
    endpoint = endpoint_id
 
    for one in response:
        if one.endpoint == endpoint:
            current_monitoring_job = one.name
            
    # Create the monitoring job for all deployed models on this endpoint.
    if len(current_monitoring_job)>0: # If monitoring job exists, pause and delete it before creating a new one
        # pause_monitoring_job(current_monitoring_job)
        # delete_monitoring_job(current_monitoring_job)
        monitoring_job = create_monitoring_job(objective_configs)
    elif len(current_monitoring_job)==0: # No monitoring job exists
        monitoring_job = create_monitoring_job(objective_configs)
        
    # Get created monitoring job IDs
    mon_job_str = str(monitoring_job)
    job_name = ""
    for i in mon_job_str.split("\n"):
        if "modelDeploymentMonitoringJobs" in i:
            job_name = i.strip()
    mon_job_id = job_name.replace('"', '').split("/")[-1]
    print("Monitoring job id: ", mon_job_id)

## Create model monitoring job
# Create monitoring job for the specified endpoint for training-serving skew detection

create_mon_job(PROJECT_ID, PROJECT_NUMBER, REGION, ENDPOINT_NAME, preprocessedMonFolder , SUFFIX, API_ENDPOINT, USER_EMAIL, DEFAULT_THRESHOLD_VALUE, LOG_SAMPLE_RATE, MONITOR_INTERVAL, TARGET, SKEW_DEFAULT_THRESHOLDS, SKEW_CUSTOM_THRESHOLDS, DRIFT_DEFAULT_THRESHOLDS, DRIFT_CUSTOM_THRESHOLDS)

## GCloud Commands for created monitoring jobs
# gcloud command to view the information/status of the specific monitoring job
# !gcloud ai model-monitoring-jobs describe <MON_JOB_ID> --project=<PROJECT_ID> --region=<REGION>

# gcloud command to pause the specific monitoring job
# !gcloud ai model-monitoring-jobs pause <MON_JOB_ID> --project=<PROJECT_ID> --region=<REGION>

# gcloud command to delete the specific monitoring job
# !gcloud ai model-monitoring-jobs delete <MON_JOB_ID> --project=<PROJECT_ID> --region=<REGION>