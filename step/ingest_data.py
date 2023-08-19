import logging
import pandas as pd
from zenml import step
from dataclasses import dataclass



class IngestData:
    """
    Ingestion data from data path.
    """
    def __init__(self,data_path:str):
        self.data_path=data_path

    def get_data(self):
        logging.info(f"Ingestion data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    Ingestion the data from data path.

    Input: Data path where csv file are saved.

    Output: Panda dataframe created by the CSV info

    """
    try:
        ingest_data=IngestData(data_path)
        df=ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in ingestion data: {e}")
        raise e