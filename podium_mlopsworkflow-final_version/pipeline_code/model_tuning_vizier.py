from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple

"""This component is for Hyperparametrs tuning with vizier"""
@component(base_image="python:3.9", output_component_file="./yaml/model-tuning-vizier.yaml",
           packages_to_install=["numpy", "sklearn", "pandas","dvc","gcsfs", "google-cloud",
                                "google-cloud-storage","google-api-python-client","xgboost",
                                "google-cloud-aiplatform", "typing","numpyencoder"])
def model_tuning_vizier(trainfile: str, bucket_name_model: str, region: str, project_id: str, smetrics: Output[Metrics])-> NamedTuple("Outputs", [("tuned_hp", str)]):
    """ This function search for best hyperparameters for XGBClassifier
        using vertex vizier.
        Inputs:Train file path, Bucket to save hyperprameters Json
        Outputs: Json path(Hyper parametrs)
    """
    from google.cloud.aiplatform_v1beta1 import VizierServiceClient
    from typing import List, Dict
    import pandas as pd
    import datetime
    import json
    import proto
    import time
    import numpy as np
    from xgboost import XGBClassifier
    from google.cloud import storage
    from numpyencoder import NumpyEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss
    
    region = region
    project_id = project_id
    start= time.time()
    def create_study(parameters: List[Dict],
                     metrics: List[Dict],
                     vizier_client,
                     project_id: str,
                     location: str):
        parent = f"projects/{project_id}/locations/{location}"
        display_name = "{}_study_{}".format(project_id.replace("-", ""),
                                            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        """ALGORITHM_UNSPECIFIED' means Bayesian optimization
           can also be 'GRID_SEARCH' or 'RANDOM_SEARCH
        """
        study = {'display_name': display_name,
                 'study_spec': {'algorithm': 'ALGORITHM_UNSPECIFIED',
                                'parameters': parameters,
                                'metrics': metrics}}
        study = vizier_client.create_study(parent=parent, study=study)
        return study.name
    
    def params_to_dict(parameters):
        return {p.parameter_id: p.value for p in parameters}
    
    def run_study(metric_fn,
                  requests: int,
                  suggestions_per_request: int,
                  client_id: str,
                  study_id: str,
                  vizier_client):
        for k in range(requests):
            suggest_response = vizier_client.suggest_trials({"parent": study_id,
                                                             "suggestion_count": suggestions_per_request,
                                                             "client_id": client_id})
            print(f"Request {k}")
            for suggested_trial in suggest_response.result().trials:
                suggested_params = params_to_dict(suggested_trial.parameters)
                metric = metric_fn(suggested_params)
                print("Trial Results", metric)
                vizier_client.add_trial_measurement({'trial_name':suggested_trial.name,
                                                     'measurement': {'metrics': [metric]}})
                response = vizier_client.complete_trial({"name": suggested_trial.name,
                                                         "trial_infeasible": False})
    
    def get_optimal_trials(study_id, vizier_client):
        optimal_trials = vizier_client.list_optimal_trials({'parent': study_id})
        optimal_params = []
        for trial in proto.Message.to_dict(optimal_trials)['optimal_trials']:
            optimal_params.append({p['parameter_id']: p['value'] for p in trial['parameters']})
        return optimal_params
    
    #discrete_value_spec, integer_value_spec, double_value_spec
    def get_params():
        parameters = [{'parameter_id':'max_depth',
                       'integer_value_spec':{'min_value':2,'max_value':8}},
                      {'parameter_id':'min_child_weight',
                       'discrete_value_spec':{'values':np.arange(0.01,0.2,0.01)}},
                      {'parameter_id':'learning_rate',
                       'double_value_spec':{'min_value':0.005,'max_value':0.3}},
                      {'parameter_id':'subsample',
                       'discrete_value_spec':{'values':np.arange(0.1,1,0.1)}},
                      {'parameter_id':'colsample_bylevel',
                       'discrete_value_spec':{'values':np.arange(0.1,1,0.1)}},
                      {'parameter_id':'colsample_bytree',
                       'discrete_value_spec':{'values':np.arange(0.1,1,0.1)}},
                      {'parameter_id':'n_estimators',
                       'integer_value_spec':{'min_value':25,'max_value':100}},
                      {'parameter_id':'gamma',
                       'discrete_value_spec':{'values':np.arange(0.5,1,0.1)}},
                      {'parameter_id':'eta',
                       'double_value_spec':{'min_value':0.025,'max_value':0.5}}]
        return parameters

    parameters = get_params()
    end_point = region + "-aiplatform.googleapis.com"
    vizier_client = VizierServiceClient(client_options=dict(api_endpoint=end_point))
    
    def metric_fn(params):
        max_depth = int(params['max_depth'])
        min_child_weight = float(params['min_child_weight'])
        learning_rate = float(params['learning_rate'])
        subsample = float(params['subsample'])
        colsample_bylevel = float(params['colsample_bylevel'])
        colsample_bytree = float(params['colsample_bytree'])
        n_estimators = int(params['n_estimators'])
        gamma = float(params['gamma'])
        eta = float(params['eta'])
        df_train = pd.read_csv(trainfile)
        adt_df_new = df_train.copy()
        train_no_taget = adt_df_new.drop(['class'], axis=1)
        train_target = adt_df_new['class']
        x_train, x_test, y_train, y_test = train_test_split(train_no_taget,train_target,
                                                            train_size = 0.75,
                                                            test_size=0.25, random_state=1)
           # """Train the model using XGBRegressor."""
        model = XGBClassifier(max_depth=max_depth,
                              min_child_weight=min_child_weight,
                              learning_rate=learning_rate,
                              subsample=subsample,
                              colsample_bylevel=colsample_bylevel,
                              colsample_bytree=colsample_bytree,
                              n_estimators=n_estimators,
                              gamma=gamma,
                              eta=eta,                       
                              eval_metric= 'logloss',
                              n_jobs=-1,
                              booster= 'gbtree',
                              tree_method= 'hist',
                              verbosity=0,
                              use_label_encoder=False,
                              random_state=124)
        model.fit(x_train,y_train)
        #predictions= model.predict(X_train)
        loss= float(log_loss(y_test, model.predict_proba(x_test)))
        #print('log loss value is:'+ str(r2_scor))
        return {'value': loss}
    
    metrics = [{'metric_id': 'log_loss',  # the name of the quantity we want to minimize
                'goal': 'MINIMIZE',  # choose MINIMIZE or MAXIMIZE
               }]
    # Call a helper function to create the study
    study_id = create_study(location = region,
                            parameters=parameters,
                            metrics=metrics,
                            vizier_client=vizier_client,
                            project_id=project_id)
    run_study(metric_fn,
              requests=10,
              # set >1 to get suggestions "in parallel", good for distributed training
              suggestions_per_request=5,
              # keep the name the same to resume a trial
              client_id="client_1",
              study_id=study_id,
              vizier_client=vizier_client,)
    params = get_optimal_trials(study_id, vizier_client)[0]
    print(params)
    max_depth = int(params['max_depth'])
    learning_rate = round(float(params['learning_rate']),2)
    n_estimators = int(params['n_estimators'])
    smetrics.log_metric("max_depth", max_depth)
    smetrics.log_metric("learning_rate", learning_rate)
    smetrics.log_metric("n_estimator", n_estimators)
    json_filename = "hptuned_vizier.json"
    with open(json_filename, 'w') as file:
        json.dump(params, file, cls=NumpyEncoder)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name_model)
    bucket.blob('vizier_model_hp/'+json_filename).upload_from_filename(json_filename)
    file_path= f"gs://{bucket_name_model}/vizier_model_hp/hptuned_vizier.json"
    end= time.time()
    smetrics.log_metric("time taken for 50 trials", (end-start))
    return (file_path,)
