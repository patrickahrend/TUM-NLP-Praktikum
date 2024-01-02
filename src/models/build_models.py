import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)

from model_classes import (
    LogisticRegressionModel,
    RandomForestModel,
    GaussianNBModel,
    DecisionTreeModel,
    PerceptronModel,
    SVCModel,
    SGDClassifierModel,
    BernoulliNBModel,
)


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
        ]

        self.embeddings = embeddings
        self.labels = labels
        self.models = []

    def train_and_save_models(self, save_directory):
        for embedding_name, (X_train, X_test) in self.embeddings.items():
            y_train, y_test = self.labels

            for constructor in self.model_constructor:
                model = constructor()
                model_name = model.model_name
                print(f"Training {model_name} with {embedding_name} embeddings")
                model.train_model(X_train, y_train)
                model_specific_dir = save_directory / embedding_name
                model_specific_dir.mkdir(exist_ok=True)
                model.save_model(model_specific_dir)

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


def main():
    project_dir = Path(__file__).resolve().parents[2]
    embeddings_path = project_dir / "data/processed/embeddings"
    labels_path = project_dir / "data/"

    embeddings = load_embeddings(embeddings_path)
    print(embeddings.keys())

    y_train, y_test = load_labels(labels_path)

    model_manager = ModelManager(embeddings, (y_train, y_test))

    models_directory = project_dir / "models"

    model_manager.train_and_save_models(models_directory)
    model_path = project_dir / "models"
    results_df = model_manager.evaluate_models(model_path)
    timestamp = datetime.now().strftime("%m%d-%H%MM")

    results_filename = f"model_evaluation_results_{timestamp}.csv"
    results_path = project_dir / "references" / results_filename
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
