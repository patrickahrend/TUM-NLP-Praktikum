import json
import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)

# Custom imports
from src.models.model_classes import (
    LogisticRegressionModel,
    RandomForestModel,
    GaussianNBModel,
    DecisionTreeModel,
    PerceptronModel,
    SVCModel,
    SGDClassifierModel,
    BernoulliNBModel,
    GradientBoostingModel,
)
from src.models.tune_hyperparameters import param_grids, tune_hyperparameters


class ModelManager:
    """
    A class used to manage different machine learning models.

    ...

    Attributes
    ----------
    model_constructor : list
        a list of classes of different machine learning models
    embeddings : dict
        a dictionary of embeddings used for training the models
    labels : tuple
        a tuple containing the labels for the training and test sets
    models : list
        a list to store the trained models

    Methods
    -------
    train_and_save_models(save_directory, tune, hyperparamater_path)
        Trains the models and saves them in the specified directory.
    evaluate_models(save_directory)
        Evaluates the performance of the models and returns the results.
    """

    def __init__(self, embeddings: dict, labels: tuple):
        """
        Constructs all the necessary attributes for the ModelManager object.

        Parameters
        ----------
            embeddings : dict
                a dictionary of embeddings used for training the models
            labels : tuple
                a tuple containing the labels for the training and test sets
        """
        self.model_constructor = [
            LogisticRegressionModel,
            RandomForestModel,
            GaussianNBModel,
            DecisionTreeModel,
            PerceptronModel,
            SVCModel,
            SGDClassifierModel,
            BernoulliNBModel,
            GradientBoostingModel,
        ]

        self.embeddings = embeddings
        self.labels = labels
        self.models = []

    def train_and_save_models(
        self,
        save_directory: Path,
        tune: bool,
        hyperparamater_path: Path,
    ) -> None:
        """
        Trains the models and saves them in the specified directory.

        Parameters
        ----------
            save_directory : Path
                the directory where the trained models will be saved
            tune : bool
                a flag indicating whether to tune the hyperparameters of the models
            hyperparamater_path : Path
                the path to the file containing the hyperparameters
        """
        save_directory.mkdir(parents=True, exist_ok=True)
        hyperparameters_file = hyperparamater_path / "hyperparameters.json"

        tuning_results = []
        for embedding_name, (X_train, X_test) in self.embeddings.items():
            embedding_save_dir = save_directory / embedding_name
            embedding_save_dir.mkdir(exist_ok=True)

            y_train, y_test = self.labels

            if tune:
                # Case actually doing hyperparameter tuning
                if hyperparamater_path.exists() and not (
                    hyperparameters_file.is_file()
                ):
                    for constructor in self.model_constructor:
                        model = constructor()
                        model_name = model.model_name
                        logging.info(
                            f"Training {model_name} with {embedding_name} embeddings"
                        )

                        # Hyperparameter tuning
                        best_model, best_params, best_score = tune_hyperparameters(
                            model.model, param_grids[model_name], X_train, y_train, cv=5
                        )

                        model.model = best_model

                        tuning_results.append(
                            {
                                "model": model_name,
                                "data": embedding_name,
                                "best_params": best_params,
                                "best_score": best_score,
                            }
                        )

                # Case for loading pre-tuned hyperparameters
                elif hyperparameters_file.is_file():
                    with open(hyperparameters_file, "r") as f:
                        hyperparameters = json.load(f)

                    for constructor in self.model_constructor:
                        model = constructor()
                        model_name = model.model_name
                        logging.info(
                            f"Training {model_name} with {embedding_name} embeddings using pre-tuned hyperparameters"
                        )

                        # Load pre-tuned hyperparameters
                        if model_name in hyperparameters.get(embedding_name, {}):
                            model_params = hyperparameters[embedding_name][model_name]
                            # Convert class_weight keys to integers as they are saved as strings in json
                            if "class_weight" in model_params:
                                model_params["class_weight"] = {
                                    int(k): v
                                    for k, v in model_params["class_weight"].items()
                                }

                            model.model.set_params(**model_params)
                            logging.info(
                                f"Pre-tuned parameters for {model_name}: {model_params}"
                            )

                        model.model.fit(X_train, y_train)
                        model.save_model(embedding_save_dir)

            # Case for no hyperparameter tuning
            else:
                for constructor in self.model_constructor:
                    model = constructor()
                    model_name = model.model_name
                    logging.info(
                        f"Training {model_name} with {embedding_name} embeddings with default parameters"
                    )
                    model.model.fit(X_train, y_train)
                    model.save_model(embedding_save_dir)

            # Save tuning results if there was hyperparameter tuning
            if tune and tuning_results:
                tuning_results_df = pd.DataFrame(tuning_results)
                tuning_results_df.to_csv(
                    save_directory / f"hyperparameter_tuning_results.csv",
                    index=False,
                )

    def evaluate_models(self, save_directory: Path) -> pd.DataFrame:
        """
        Evaluates the performance of the models and returns the results.

        Parameters
        ----------
            save_directory : Path
                the directory where the trained models are saved

        Returns
        -------
            results : DataFrame
                a DataFrame containing the evaluation results of the models
        """
        columns = [
            "model",
            "data",
            "accuracy_is",
            "accuracy_oos",
            "precision_oos",
            "recall_oos",
            "f1_oos",
        ]
        results_list = []
        for embedding_name, (X_train, X_test) in self.embeddings.items():
            y_train, y_test = self.labels
            model_dir = save_directory / embedding_name

            # Iterate through each saved model in the directory
            for model_file in model_dir.iterdir():
                if model_file.suffix == ".pkl":
                    with open(model_file, "rb") as file:
                        model = pickle.load(file)

                    # Extract model name from the file name
                    model_name = model_file.stem

                    logging.info(
                        f"Evaluating {model_name} with {embedding_name} embeddings"
                    )
                    logging.info(
                        f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
                    )

                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                    accuracy_train = accuracy_score(y_train, y_pred_train)
                    accuracy_test = accuracy_score(y_test, y_pred_test)
                    precision_oos = precision_score(
                        y_test, y_pred_test
                    )
                    recall_oos = recall_score(y_test, y_pred_test)
                    f1_oos = f1_score(y_test, y_pred_test)

                    results_list.append(
                        {
                            "model": model_name,
                            "data": embedding_name,
                            "accuracy_is": accuracy_train,
                            "accuracy_oos": accuracy_test,
                            "precision_oos": precision_oos,
                            "recall_oos": recall_oos,
                            "f1_oos": f1_oos,
                        }
                    )
                    logging.info(
                        f"Model: {model_name}\n"
                        f"Data: {embedding_name}\n"
                        f"In-sample accuracy: {accuracy_train:.3f}\n"
                        f"Out-of-sample accuracy: {accuracy_test:.3f}\n"
                    )
        results = pd.DataFrame(results_list, columns=columns)
        return results
