import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):
    """
    Abstract class for evaluation.
    """
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Evaluate the model.

        Input: trained_model, X_test, y_test

        Return: None
        """
        pass

class MSE(Evaluation):
    """
    Mean Square Error.
    
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            mse=mean_squared_error(y_true,y_pred)
            logging.info(f"Mean Square Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e

class R2(Evaluation):
    """
    R2 score.
    
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            r2=r2_score(y_true,y_pred)
            logging.info(f"R2 score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 score: {e}")
            raise e
        
class RMSE(Evaluation):
    """
    Root Mean Square Error.
    
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            rmse=np.sqrt(mean_squared_error(y_true,y_pred))
            logging.info(f"Root Mean Square Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e
