"""This component is for getting data from gcs bucket"""
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple
@component(base_image="python:3.9", output_component_file="./yaml/data-validation.yaml",
           packages_to_install=["pandas", "scipy", "numpy",
                                "google-api-python-client" ,
                                "gcsfs", "google-cloud","google-cloud-storage"])
def data_validation(gcs_file: str)-> NamedTuple("Outputs", [("data_validation_sucess_output_flag", bool)]):
    
    import pandas as pd
    import numpy as np
    import pickle
    from scipy import stats
    from google.cloud import storage
    
    '''
    Inputs:
    GCSFile : Entire path and name of the csv file of training data
    generationNumber: Object generation number
    timeStamp: Object generation timestamp
    
    Function details:
    This function is used to perform data validation to identify:
    1. Outlier detection -
    Use of Interquartile range rule to detect outliers and raise alerts.
    2. Data schema matching -
    Checking the data types of ingested data features with the respective 
    target data types.
    3. Missing data check -
    Check for missing data in the dataframe and raise alerts if missing
    values are detected in the dataframe.
    '''
    
    # Read input ingested data
    raw_data_in = pd.read_csv(gcs_file)
    def outlier_detection(df):
        df_numerical_cols = df.select_dtypes(include=['float64', 'int64'])
        # Emulation of box plots
        Q1 = df_numerical_cols.quantile(0.25)
        Q3 = df_numerical_cols.quantile(0.75)
        # Interquartile range  
        IQR = Q3 - Q1
        # print(IQR)
        # Use Interquartile range rule to detect presence of outliers in the data
        outlier_bool = (df_numerical_cols < (Q1 - 1.5 * IQR)) |(df_numerical_cols > (Q3 + 1.5 * IQR))
        # print(outlier_bool)
        # Curate the datapoint indices which are outliers
        outlier_idx = list(np.where(np.array(outlier_bool.all(axis=1)) == True)[0])
        # print(len(outlier_idx))
        print("-"*20)
        print("Check 1: Outlier detection")
        # Raise alert if outliers are detected
        if len(outlier_idx) != 0:
            print(" --- ALERT : Outliers detected in the dataset --- ")
            print("Outliers are at the following index in the Dataframe: ")
            print(outlier_idx)
            outlier_detection_flag = False
        else: 
            print("--- PASS : No outliers detected ---")
            outlier_detection_flag = True
        print("-"*20)
        return outlier_detection_flag
    
    def schema_check(df):
        # Get the data type schema in the ingested data
        trainData_schema = dict(df.dtypes)
        # Load the target schema file
        storage_client = storage.Client()
        bucket = storage_client.bucket('mlops-data-validation')
        blob = bucket.blob('data_schema.pickle')
        pickle_in = blob.download_as_string()
        target_schema = pickle.loads(pickle_in)
        # Check if the ingested data feature data types match the target 
        # and raise alerts if they don't
        print("Check 2: Data schema matching")
        if trainData_schema == target_schema:
            print("--- PASS : Data types match the expected data types ---")
            schema_check_flag = True
        else: 
            print("--- ALERT : Data types do not match the expected data types ---")
            schema_check_flag = False
        print("-"*20)
        return schema_check_flag
    
    def missing_val_check(df):    
        # Check for NaN values/missing values
        missing_val_list = np.asarray(df.isnull().any(axis=1))
        # Find the indices of the datapoints where missing values are detected
        missing_val_idx = list(np.where(missing_val_list == True)[0])
        # Raise alerts if presence of missing values are detected
        print("Check 3: Missing data check")
        if len(missing_val_idx) == 0: 
            print("--- PASS : No missing values detected ---")
            missing_val_flag = True
        else: 
            print("--- ALERT : Missing values detected in the dataset ---")
            print("Missing values detected in the datapoints at the following indices:",missing_val_idx)
            
            # Calculate the percentage of datapoints/rows containing missing values
            total_rows_df = df.shape[0]
            rows_missing_vals_df = len(missing_val_idx)
            missing_data_percent = round((rows_missing_vals_df/total_rows_df)*100, 2)
            print("{} % of datapoints in the given data contain missing values.".format(missing_data_percent))
            missing_val_flag = False
        print("-"*20)
        return missing_val_flag
    # Check 1: Outlier detection
    Oulier_flag = outlier_detection(raw_data_in)
    # Check 2: Data schema matching
    schema_flag = schema_check(raw_data_in)
    # Check 3: Missing data check
    missing_flag = missing_val_check(raw_data_in)
    
    if Oulier_flag==False or schema_flag == False or missing_flag == False:
        data_validation_suceess_flag = False
    else:
        data_validation_suceess_flag = True
    return (data_validation_suceess_flag,)