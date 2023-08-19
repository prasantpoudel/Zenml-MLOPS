import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class for data cleaning strategy.
    """
    @abstractmethod
    def handle_data(self, data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        """
        Abstract method for cleaning data.
        """
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Strategy for data preprocessing.
    """

    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        try:
            data=data.drop_duplicates()
            data=data.drop([
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
                ],axis=1)

            data["product_weight_g"].fillna(data["product_weight_g"].median(),inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(),inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(),inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(),inplace=True)
            data["review_comment_message"].fillna("No Review",inplace=True) 

            data=data.select_dtypes(include=[np.number])
            data.drop(['order_item_id','customer_zip_code_prefix'],axis=1,inplace=True)

            return data
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e
        
class DataSplitStrategy(DataStrategy):
    """
    Strategy for data splitting.
    """
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame,pd.Series]: # pd.DataFrame | pd.Series
        try:
            X=data.drop(["review_score"],axis=1)
            y=data["review_score"]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            raise e
        
class DataCleaning:
    """
    Context class for data cleaning.
    """
    def __init__(self, data:pd.DataFrame,strategy:DataStrategy):
        self.data=data
        self.strategy=strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in data cleaning: {e}")
            raise e

