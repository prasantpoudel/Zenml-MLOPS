import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for model.
    """
    @abstractmethod
    def train(self,X_train,y_train):
        """
        Train the model.

        Input: X_train, y_train

        Return: None
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression model.
    
    """
    def train(self, X_train, y_train):
        try:
            reg=LinearRegression()
            reg.fit(X_train,y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e