# Podium Churn MLOps pilot

Building the MLOps pipeline for the podium using VertexAI in GCP as a pilot 

## Instructions to run the trainig pipeline
#### Automatically with existing json on GCS bucket
* Upload a input.csv in mlops-data-drop bucket it will trigger the pipeline automatically
* Note: compiled json for CTD pipeline is stored in mlops-pipelines-artifacts, it runs the pipeline.
        if any changes in the code compile the components again to create new json

  #### How to run the pipeline manually:
* Clone the repo in your vertex notebooks
* Open the terminal and make the working directory as pipeline_code
* Run the "python pipeline.py" command it will start running the pipelines in vertexAI
* Two arguments can be passed to the code either 0 or 1 (not mandatory)
* If no arguments given default pipeline will run
* If first argument is 0 and second argument is 1 then default pipeline with vertex vizier pipeline will run
* If first argument is 1 and second argument is 0 then default pipeline with autoML pipeline will run
* If first argument is 1 and second argument is 1 then default pipeline with vertex vizier and autoML pipeline will run

## Instructions to run the batch prediction pipeline:

#### Automatically with existing json on GCS bucket
* Upload a prediction filed named exactly as batch_pred_data.csv in batch-pred-data gcs bucket, it will trigger the batch prediction pipeline automatically
* Note: compiled json for pipeline is stored in prediction_serving bucket, it is mandatory to run the pipeline if any chages are made to code in Batch_prediction/batch_prediction.py

  #### How to run the pipeline manually:
* Clone the repo in your vertex notebooks
* Open the terminal and make the working directory as Batch_prediction
* Run the "python batch_prediction.py" command it will start running the pipelines in vertexAI


## High level understanding of the technologies used in the solution

### ML Techniques
* XGB Classifier - For predicting churn

### Programming languages

* Python 3.7.10 - For Building VertexAI pipeliens

### Infrastructure (training and Deployment)

* VetexAI
* Cloud Pub/Sub
* Cloud functions

## Setup
Step by step guide to setup the applicatio
clone the below repo in vertexai workbench
 
```bash
git clone https://gitlab.qdatalabs.com/applied-ai/applied-ai_west/podium_mlopsworkflow
```

Make pipeline_code as working directory to run the pipeline.py (To Run CTD pipeline)
give the arguments based upon the requirements as mentioned in Instructions to run the trainig pipeline section

```bash
python pipeline.py
```

Make pipeline_code as working directory to run the model_monitoring.py (To Run model monitoring job)

```bash
python model_monitoring.py
```

Make Batch_prediction as working directory to run the pipeline.py (To run batch predictions)

```bash
python batch_prediction.py
```

