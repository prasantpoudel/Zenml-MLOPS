import logging
import pandas as pd
from zenml import step
from src.model_dev import Model, LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import pickle

@step
def train_model(
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series,
    config:ModelNameConfig) ->RegressorMixin:
    """
    Train the model using the dataframe.

    Input: X_train, X_test, y_train, y_test

    Return: model
    """
    try:
        model=None
        if config.model_name=="LinearRegressionModel":
            model=LinearRegressionModel()
            trained_model=model.train(X_train,y_train)
        
            filename = 'model.pkl'
            pickle.dump(trained_model, open(filename, 'wb'))
            logging.info("Model training completed")
            return trained_model
        else:
            raise ValueError(f"Model name {config.model_name} is not supported")
        
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
