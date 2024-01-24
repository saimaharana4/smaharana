from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    mode_name: str = "LinearRegression"
    