import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class Evaluation(ABC):
    """Abstract base class for evaluating a model's performance."""
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        Calculate the score of the model
        """
        pass
    
class MSE(Evaluation):
    """Mean Squared Error evaluation metric."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE ")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}.")
            return mse
        except Exception as e:
            logging.error(f"Failed to calculate MSE due to error: {e}")
            raise e
        
class R2(Evaluation):
    """R-squared (coefficient of determination) evaluation metric."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Failed to calculate R2 Score due to error: {e}")
            raise e

class RMSE(Evaluation):
    """Root Mean Squared Error evaluation metric."""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE ")
            rmse = mean_squared_error(y_true, y_pred, squared= False)
            logging.info(f"RMSE: {rmse}.")
            return rmse
        except Exception as e:
            logging.error(f"Failed to calculate RMSE due to error: {e}")
            raise e