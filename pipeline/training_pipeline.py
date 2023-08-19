from zenml import pipeline
from step.ingest_data import ingest_data
from step.clean_data import clean_data
from step.model_train import train_model
from step.model_evaluation import evaluate_model
import logging


@pipeline(enable_cache=True)
def training_pipeline(data_path:str):
    """
    Training pipeline with all the steps.
    """
    try:
        df=ingest_data(data_path)
        X_train,X_test,y_train,y_test=clean_data(df)
        model=train_model(X_train,X_test,y_train,y_test)
        mse,r2,rmse=evaluate_model(model,X_test,y_test)
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise e


