from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple

"""This component is for hyperparameters tuning using Hyperopt"""
@component(base_image="python:3.9", output_component_file="./yaml/model-tuning.yaml",
           packages_to_install=["numpy", "sklearn", "pandas","gcsfs", "google-cloud",
                                "google-cloud-storage","xgboost","hyperopt","numpyencoder"])
def model_tuning(train_file: str, bucket_name_model: str,smetrics: Output[Metrics])-> NamedTuple("Outputs", [("tuned_hp", str)]):  
    """ This function search for best hyperparameters for XGBClassifier
        using hyperopt.
        Inputs:Train file path, Bucket to save hyperprameters Json
        Outputs: Json path(Hyper parametrs)
    """
    from xgboost import XGBClassifier
    import pandas as pd
    from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    import time
    import numpy as np
    from hyperopt import space_eval
    from hyperopt import Trials
    from google.cloud import storage
    import json
    from numpyencoder import NumpyEncoder
    # Set up the XGBoost version
    # Declare xgboost search space for Hyperopt
    start= time.time()
    train = pd.read_csv(train_file)
    train_features = train.drop(['class'], axis=1)
    train_target = train['class']
    #best_score =1
    xgboost_space={'max_depth': hp.choice('x_max_depth',[2,3,4,5,6,7,8]),
                   'min_child_weight':hp.choice('x_min_child_weight',
                                                np.round(np.arange(0.0,0.2,0.01),5)),
                   'learning_rate':hp.choice('x_learning_rate',
                                             np.round(np.arange(0.005,0.3,0.01),5)),
                   'subsample':hp.choice('x_subsample',
                                         np.round(np.arange(0.1,1.0,0.05),5)),
                   'colsample_bylevel':hp.choice('x_colsample_bylevel',
                                                 np.round(np.arange(0.1,1.0,0.05),5)),
                   'colsample_bytree':hp.choice('x_colsample_bytree',
                                                np.round(np.arange(0.1,1.0,0.05),5)),
                   'n_estimators':hp.choice('x_n_estimators',np.arange(25,100,5)),
                   'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                   'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
                   'booster': 'gbtree',
                   'tree_method': 'hist',
                   'eval_metric': 'logloss'}
    def objective(space):
        #global best_score
        best_score = 1
        model = XGBClassifier(**space, n_jobs=-1, use_label_encoder=False)
        kfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1114)
        score = -cross_val_score(model,train_features,train_target,
                                 cv=kfold,
                                 scoring='neg_log_loss',
                                 verbose=False).mean()
        if (score < best_score):
            best_score = score
        return score
    start = time.time()
    trials = Trials()
    best = fmin(objective,space = xgboost_space, algo = tpe.suggest,max_evals = 50,trials = trials)
    print("Hyperopt search took %.2f seconds for 100 candidates" % ((time.time() - start)))
    tuned_hp_dict= space_eval(xgboost_space, best)
    print(tuned_hp_dict)
    json_file_name = "hptuned.json"   
    with open(json_file_name, 'w') as file:  
        json.dump(tuned_hp_dict, file, cls=NumpyEncoder)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name_model)
    bucket.blob('tuned_hp/'+json_file_name).upload_from_filename(json_file_name)
    file_path= f"gs://{bucket_name_model}/tuned_hp/hptuned.json"
    end= time.time()
    smetrics.log_metric("max_depth",float(tuned_hp_dict['max_depth']))
    smetrics.log_metric("n_estimators",float(tuned_hp_dict['n_estimators']))
    smetrics.log_metric("learning_rate",float(tuned_hp_dict['learning_rate']))
    smetrics.log_metric("time taken for 50 trials", (end-start))
    return (file_path,)
