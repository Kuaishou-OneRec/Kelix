import os
import pandas as pd
from typing import Optional, Dict
from recovlm.utils.blobstore_client import BlobStoreClient

class DataSource(object):

    def __call__(self, row: Dict[str, any]) -> Optional[bytes]:
        raise NotImplementedError


class BlobStoreDataSource(DataSource):

    def __init__(self, bucket, key_name):
        self.client = BlobStoreClient(caller="recovlm_downloader")
        self.bucket = bucket
        self.key_name = key_name
    
    def __call__(self, row: Dict[str, any]) -> Optional[bytes]:
        key = row[self.key_name]
        data = self.client.get_object(self.bucket, key)
        return data


def create_datasource(class_name, kwargs) -> DataSource:
    return eval(class_name)(**kwargs)