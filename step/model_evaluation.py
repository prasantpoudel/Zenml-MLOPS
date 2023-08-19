import logging
import pandas as pd
from zenml import step
from src.evaluation import R2,MSE,RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated


@step
def evaluate_model(model:RegressorMixin,
                   x_test:pd.DataFrame,
                   y_test:pd.Series) -> Tuple[
                       Annotated[float,"mse"],
                       Annotated[float,"r2"],
                       Annotated[float,"rmse"]]:
    """
    Evaluate the model using the data.
    """
    try:
        logging.info("Model evaluation started")
        prediction=model.predict(x_test)
        mse_class=MSE()
        mse=mse_class.calculate_scores(y_test,prediction)

        r2_class=R2()
        r2=r2_class.calculate_scores(y_test,prediction)

        rmse_class=RMSE()
        rmse=rmse_class.calculate_scores(y_test,prediction)

        logging.info("Model evaluation completed")
        logging.info(f"MSE: {mse}, R2: {r2}, RMSE: {rmse}")

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e

    return mse,r2,rmse