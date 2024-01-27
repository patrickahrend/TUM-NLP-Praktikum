import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

# Custom imports
from src.models.build_models import ModelManager


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)
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

def main():
    logger = logging.getLogger(__name__)
    logger.info("Building models and saving them to models/")
    project_dir = Path(__file__).resolve().parents[2]
    is_tuned = False
    dataset_variant = "combined"  # combined or separate
    tuned_dir = "tuned" if is_tuned else "no_tuning"

    # Initialize a dictionary to store the datasets
    embeddings = {}

    embedding_files = os.listdir(project_dir / "data/processed/embeddings")

    # Process each type of embedding for Dataset 2
    for emb_type in ["gpt", "fasttext", "word2vec", "glove", "bert", "tfidf"]:
        print(emb_type)
        variant_files = [
            f
            for f in embedding_files
            if dataset_variant in f and f.startswith(emb_type)
        ]
        print("Variant Files here", variant_files)

        training_data = None
        test_data = None

        for file_name in variant_files:
            if "train" in file_name:
                training_data = load_pickle(
                    project_dir / f"data/processed/embeddings/{file_name}"
                )
            elif "test" in file_name:
                test_data = load_pickle(
                    project_dir / f"data/processed/embeddings/{file_name}"
                )
        if training_data is not None and test_data is not None:
            y_train = training_data["Label"]
            y_test = test_data["Label"]
            columns_to_drop = [
                "Text",
                "Label",
                "Process",
                "Process_description",
                "Combined_Text",
            ]
            X_train = training_data.drop(columns=columns_to_drop)
            X_test = test_data.drop(columns=columns_to_drop)

            # Add the dataset to the dictionary
            embeddings[emb_type] = (X_train, X_test)
        else:
            print(f"Training or test data not found for {emb_type} embeddings.")

    model_manager = ModelManager(embeddings, (y_train, y_test))

    models_path = (
            project_dir / "models" / "trained_models" / dataset_variant / tuned_dir
    )

    hparams_path = project_dir / "data/processed/hyperparameters"

    model_manager.train_and_save_models(models_path, is_tuned, hparams_path)

    results_df = model_manager.evaluate_models(models_path)
    timestamp = datetime.now().strftime("%m%d-%H%M")

    if len(sys.argv) > 1:
        details = "_".join(sys.argv[1:])
    else:
        details = input(
            "Add details of this experiment e.g which dataset, which features: "
        )

    results_filename = f"model_evaluation_results_{details}_{timestamp}.csv"
    os.makedirs(project_dir / "references/model results", exist_ok=True)
    results_path = project_dir / "references/model results" / results_filename
    results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
