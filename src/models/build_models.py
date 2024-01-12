import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

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
from tune_hyperparameters import param_grids, tune_hyperparameters


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
    ):
        save_directory.mkdir(parents=True, exist_ok=True)

        tuning_results = []
        for embedding_name, (X_train, X_test) in self.embeddings.items():
            embedding_save_dir = save_directory / embedding_name
            embedding_save_dir.mkdir(exist_ok=True)

            y_train, y_test = self.labels

            for constructor in self.model_constructor:
                model = constructor()
                model_name = model.model_name
                print(f"Training {model_name} with {embedding_name} embeddings")

                # # Hyperparameter tuning
                if tune:
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
                else:
                    model.model.fit(X_train, y_train)

                model.save_model(embedding_save_dir)

        if tune:
            tuning_results_df = pd.DataFrame(tuning_results)
            tuning_results_df.to_csv(
                save_directory / f"tuning_results.csv",
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


def load_embeddings(embeddings_path):
    embeddings = {}
    for embedding_name in ["tfidf", "w2v", "bert", "gpt", "glove", "ft"]:
        train_pickle = embeddings_path / f"{embedding_name}_train.pkl"
        test_pickle = embeddings_path / f"{embedding_name}_test.pkl"

        with open(train_pickle, "rb") as f:
            X_train = pickle.load(f)
        with open(test_pickle, "rb") as f:
            X_test = pickle.load(f)

        embeddings[embedding_name] = (X_train, X_test)

    return embeddings


def load_labels(labels_path):
    train_labels_df = pd.read_csv(
        labels_path / "processed/training_data_preprocessed.csv"
    )
    test_labels_df = pd.read_csv(
        labels_path / "evaluation/gold_standard_preprocessed.csv"
    )

    y_train = train_labels_df["Label"]
    y_test = test_labels_df["Label"]

    return y_train, y_test


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def main():
    project_dir = Path(__file__).resolve().parents[2]

    is_tuned = False
    dataset_variant = "separate"  # or "combined"
    use_pca_variant = False  # or True
    dataset_dir = "pca" if use_pca_variant else "normal"
    tuned_dir = "tuned" if is_tuned else "no_tuning"
    experiment_name = f"experiment_2_{dataset_variant}_{use_pca_variant}_{is_tuned}"

    labels_path = project_dir / "data/"

    # Load labels
    y_train, y_test = load_labels(labels_path)

    # Initialize a dictionary to store the datasets
    embeddings = {}

    embedding_path = (
        project_dir / f"data/processed/datasets/{dataset_dir}/{dataset_variant}"
    )
    # Process each type of embedding for Dataset 2
    for emb_type in ["gpt", "ft", "w2v", "glove", "bert", "tfidf"]:
        # Load the dataset for separate process and legal text embeddings
        X_train = load_pickle(embedding_path / f"{emb_type}_train.pkl")
        X_test = load_pickle(embedding_path / f"{emb_type}_test.pkl")

        # Add the dataset to the dictionary
        embeddings[emb_type] = (X_train, X_test)

    model_manager = ModelManager(embeddings, (y_train, y_test))

    models_path = (
        project_dir
        / "models"
        / experiment_name
        / dataset_dir
        / dataset_variant
        / tuned_dir
    )

    model_manager.train_and_save_models(
        models_path,
        is_tuned,
    )

    results_df = model_manager.evaluate_models(models_path)
    timestamp = datetime.now().strftime("%m%d-%H%M")

    if len(sys.argv) > 1:
        details = "_".join(sys.argv[1:])
    else:
        details = input(
            "Add details of this experiement e.g which dataset, which features: "
        )

    results_filename = f"model_evaluation_results_{details}_{timestamp}.csv"
    os.makedirs(project_dir / "references/model results", exist_ok=True)
    results_path = project_dir / "references/model results" / results_filename
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
