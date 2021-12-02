from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple
""" This components is for data preprocessing and spliting the data"""
@component(base_image="python:3.9", output_component_file="./yaml/data_transformation.yaml",
           packages_to_install=["numpy", "sklearn", "pandas","gcsfs", "google-cloud",
                                "google-cloud-storage"])
def data_transformation(gcs_file: str, bucket_tranformed_data: str)->NamedTuple("Outputs", [("train_data", str),("test_data", str)]):
    """This the Preprocessing functions checks for NA ,
       drops them  and splits the dataset in to test train splits
       Inputs: File path
       Outputs: train and test datavser file paths
       """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    churn = pd.read_csv(gcs_file)
    churn.dropna(inplace=True)
    churn.reset_index(inplace=True, drop=True)
    print("Dropped all the NAs")
    # Rename depvar to class as required by TPOT
    churn.rename(columns={'event': 'class'}, inplace=True)
    # Create depvar in separate data set.
    churn_class = churn['class'].values
    training_indices, testing_indices = train_test_split(churn.index,stratify = churn_class,
                                                         train_size=0.75,test_size=0.25,
                                                         random_state = 1114)
    lifetime_enriched_train_rows_qp = churn.iloc[training_indices, :]
    lifetime_enriched_test_rows_qp = churn.iloc[testing_indices, :]
    train_data = f"gs://{bucket_tranformed_data}/train.csv"
    test_data = f"gs://{bucket_tranformed_data}/test.csv"
    lifetime_enriched_train_rows_qp.to_csv(train_data, index = False)
    lifetime_enriched_test_rows_qp.to_csv(test_data, index = False)
    return (train_data,test_data,)
