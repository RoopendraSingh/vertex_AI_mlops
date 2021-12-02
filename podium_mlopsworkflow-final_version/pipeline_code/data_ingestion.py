from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)
from typing import NamedTuple
"""This component is for getting data from gcs bucket"""
@component(base_image="python:3.9", output_component_file="./yaml/data_ingestion.yaml",
           packages_to_install=["google-api-python-client", "gcsfs",
                                "google-cloud","google-cloud-storage"])
def data_ingestion(bucket_name: str,notebook_path:str,smetrics: Output[Metrics])-> NamedTuple("Outputs", [("gcsFile", str)]):
    from google.cloud import storage
    
    """This module returns the filepath"""
    def list_blobs_with_prefix(bucket_name:str, delimiter=None):
        """Gives the last updated file path
           Inputs: GCS bucket name
           Outputs: train file path
        """
        storage_client = storage.Client()
        blobs = storage_client.list_blobs(bucket_name, delimiter=delimiter)
        dict_blob = {}
        for blob in blobs:
            dict_blob[blob.name] = blob.updated
        blob_update_list = list(dict_blob.values())
        sorted_dict_blob = {k: v for k, v in sorted(dict_blob.items(),
                                                    key=lambda item: item[1], reverse = True)}
        get_latest_file_name = list(sorted_dict_blob.keys())[0]
        return get_latest_file_name
    file_name = list_blobs_with_prefix(bucket_name,delimiter=None)
    get_gcs_File = f'gs://{bucket_name}/'+file_name
    print("Data DownLoaded Sucessfully")
    print(get_gcs_File)
    if notebook_path == 'NA':
        return(get_gcs_File,)
    else:
        smetrics.log_metric("path to code:",notebook_path)
        return(get_gcs_File,)

