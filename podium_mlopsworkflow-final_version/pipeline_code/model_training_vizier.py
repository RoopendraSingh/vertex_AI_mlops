from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple

"""This component is for training XGB classifier using 
   hyperparametrs obtained from hyperopt
"""
@component(base_image="python:3.9", output_component_file="./yaml/model-training-vizier.yaml",
           packages_to_install=["numpy", "sklearn", "pandas","gcsfs", "google-cloud",
                                "google-cloud-storage","xgboost","hyperopt"])
def model_training_vizier(tuned_hp: str, trainfile: str,model:Output[Model]):
    #-> NamedTuple("Outputs", [("model_path", str)]):
    """This function uses the hyper-parameters using vizier
       and trains a XGBClassifier for churn prediction
       Inputs: Json contains hyperparametrs, train file path
       Output: file path of saved model
    """
    from xgboost import XGBClassifier
    import pandas as pd
    from hyperopt import hp, fmin, tpe, rand, STATUS_OK, Trials
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import classification_report
    import time
    import numpy as np
    import pickle
    from hyperopt import space_eval
    from hyperopt import Trials
    from google.cloud import storage
    import gcsfs
    import json
    # Set up the XGBoost version
    # Declare xgboost search space for Hyperopt
    # Get best model in a dict format.
    train = pd.read_csv(trainfile)
    train_features = train.drop(['class'], axis=1)
    train_target = train['class']
    file_system = gcsfs.GCSFileSystem()
    best_model = json.load(file_system.open(tuned_hp, 'rb'))
    # Create the pipeline and set hyperparameters.
    exported_pipeline = XGBClassifier(eta=best_model.get('eta'), 
                                      gamma=best_model.get('gamma'),
                                      colsample_bylevel=best_model.get('colsample_bylevel'),
                                      colsample_bytree=best_model.get('colsample_bytree'),
                                      eval_metric=best_model.get('eval_metric'),
                                      learning_rate=best_model.get('learning_rate'), 
                                      max_depth=int(best_model.get('max_depth')), 
                                      min_child_weight=best_model.get('min_child_rate'), 
                                      n_estimators=int(best_model.get('n_estimators')), 
                                      n_jobs=-1, 
                                      booster=best_model.get('booster'),
                                      subsample=best_model.get('subsample'), 
                                      tree_method=best_model.get('tree_method'), 
                                      verbosity=0,
                                      use_label_encoder = False,
                                      early_stopping_rounds=10)
    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 1114)
    # Fit model
    exported_pipeline.fit(train_features, train_target)
    print("Training Sucessfully Completed")
    #dumping the model to gcs bucket.
    exported_pipeline.save_model("model_sklearn_viz.bst")
    exported_pipeline.save_model(model.path+".bst")

