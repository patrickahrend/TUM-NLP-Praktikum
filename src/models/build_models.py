import json
import pickle


import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)

# Custom imports
from model_classes import (
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
    def __init__(self, embeddings, labels):
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
        save_directory,
        tune,
        hyperparamater_path,
    ):
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
                        print(f"Training {model_name} with {embedding_name} embeddings")

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
                        print(
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
                            print(
                                f"Pre-tuned parameters for {model_name}: {model_params}"
                            )

                        model.model.fit(X_train, y_train)
                        model.save_model(embedding_save_dir)

            # Case for no hyperparameter tuning
            else:
                for constructor in self.model_constructor:
                    model = constructor()
                    model_name = model.model_name
                    print(
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

    def evaluate_models(self, save_directory):
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

                    print(f"Evaluating {model_name} with {embedding_name} embeddings")
                    print(
                        f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
                    )

                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                    accuracy_train = accuracy_score(y_train, y_pred_train)
                    accuracy_test = accuracy_score(y_test, y_pred_test)
                    precision_oos = precision_score(
                        y_test, y_pred_test, average="weighted"
                    )
                    recall_oos = recall_score(y_test, y_pred_test, average="weighted")
                    f1_oos = f1_score(y_test, y_pred_test, average="weighted")

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
                    print(
                        f"Model: {model_name}\n"
                        f"Data: {embedding_name}\n"
                        f"In-sample accuracy: {accuracy_train:.3f}\n"
                        f"Out-of-sample accuracy: {accuracy_test:.3f}\n"
                    )
        results = pd.DataFrame(results_list, columns=columns)
        return results
