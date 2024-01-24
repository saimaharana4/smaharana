import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit()
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model : {e}")
            raise e