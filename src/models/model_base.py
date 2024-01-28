import logging
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd


class ModelBase:
    """
    Base class for machine learning models.

    ...

    Attributes
    ----------
    model_name : str
        The name of the machine learning model.
    model : object
        The instance of the machine learning model.

    Methods
    -------
    save_model(directory)
        Saves the model to a pickle file in the specified directory.
    train_model(X_train, y_train)
        Trains the model using the provided training data and labels.
    predict(X)
        Makes predictions using the model.
    set_params(**params)
        Sets the parameters of the model.
    """

    def __init__(self, model_name: str, model_instance: BaseEstimator):
        """
        Constructs all the necessary attributes for the ModelBase object.

        Parameters
        ----------
            model_name : str
                The name of the machine learning model.
            model_instance : BaseEstimator from Sklearn
                The instance of the machine learning model.
        """
        self.model_name = model_name
        self.model = model_instance

    def save_model(
        self,
        directory: Path,
    ) -> None:
        """
        Saves the model to a pickle file in the specified directory.

        Parameters
        ----------
            directory : Path
                The directory where the model will be saved.
        """
        filepath = directory / f"{self.model_name}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logging.info(f"Model saved to {filepath}")

    def train_model(self, X_train: pd.DataFrame , y_train: pd.Series) -> None:
        """
        Trains the model using the provided training data and labels.

        Parameters
        ----------
            X_train : pd.DataFrame
                The training data.
            y_train : pd.Series
                The labels for the training data.
        """
        self.model.fit(X_train, y_train)
        logging.info(f"{self.model_name} trained successfully.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the model.

        Parameters
        ----------
            X : pd.DataFrame
                The data to make predictions on.

        Returns
        -------
            np.ndarray
                The predictions made by the model.
        """
        return self.model.predict(X)

    def set_params(self, **params: dict) -> None:
        """
        Sets the parameters of the model.

        Parameters
        ----------
            **params : dict
                The parameters to set for the model.
        """
        self.model.set_params(**params)
