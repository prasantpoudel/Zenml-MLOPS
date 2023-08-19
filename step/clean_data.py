import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataPreprocessStrategy,DataSplitStrategy, DataCleaning
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"]]:
    """
    cleaning the data.

    Input: Dataframe from ingestion data.

    Return: X_train, X_test, y_train, y_test

    """
    try:
        logging.info("Cleaning data started")
        preprocess_strategy=DataPreprocessStrategy()
        data_clean=DataCleaning(df,preprocess_strategy)
        df=data_clean.handle_data()

        split_strategy=DataSplitStrategy()
        split_data=DataCleaning(df,split_strategy)
        X_train,X_test,y_train,y_test=split_data.handle_data()

        # we can add more strategy here
        logging.info("Cleaning data completed")
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.error(f"Error in cleaning data: {e}")
        raise e
