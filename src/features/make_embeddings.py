import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_word_embeddings import EmbeddingProcessor


def save_df_with_embeddings(
    original_df: pd.DataFrame,
    embeddings: pd.DataFrame,
    embedding_type: str,
    filename: Path,
) -> None:
    """
    Saves a DataFrame with embeddings to a pickle file.

    Parameters:
    original_df (DataFrame): The original DataFrame.
    embeddings (pd.DataFrame): The computed embeddings.
    embedding_type (str): The type of the embeddings.
    filename (Path): The path to the output file.
    """
    # If the embeddings are 3-dimensional and the second dimension is 1, squeeze it
    if len(embeddings.shape) == 3 and embeddings.shape[1] == 1:
        embeddings = np.squeeze(embeddings, axis=1)

    num_dimensions = embeddings.shape[1]
    # needed as for the seperate approach the embeddings are already a dataframe, which otherwise introduces null values in the final dataframe
    if not isinstance(embeddings, pd.DataFrame):
        embedding_df = pd.DataFrame(
            embeddings,
            columns=[f"{embedding_type}_{i}" for i in range(num_dimensions)],
        )
    else:
        embedding_df = embeddings

    final_df = pd.concat([original_df.reset_index(drop=True), embedding_df], axis=1)

    filename_with_dimensions = str(filename) + f"_{num_dimensions}.pkl"

    with open(filename_with_dimensions, "wb") as file:
        pickle.dump(final_df, file)


def concat_embeddings_with_df(
    embedding1: pd.DataFrame, embedding2: pd, embedding1_name: str, embedding2_name: str
) -> pd.DataFrame:
    """
    Concatenates two sets of embeddings into a DataFrame.

    Parameters:
    embedding1 (pd.DataFrame): The first set of embeddings.
    embedding2 (pd.DataFrame): The second set of embeddings.
    embedding1_name (str): The name of the first set of embeddings.
    embedding2_name (str): The name of the second set of embeddings.

    Returns:
    DataFrame: The DataFrame containing the concatenated embeddings.
    """
    if len(embedding1.shape) == 3 and embedding1.shape[1] == 1:
        embedding1 = np.squeeze(embedding1, axis=1)
    if len(embedding2.shape) == 3 and embedding2.shape[1] == 1:
        embedding2 = np.squeeze(embedding2, axis=1)

    embedding1_df = pd.DataFrame(
        embedding1,
        columns=[f"{embedding1_name}_{i}" for i in range(embedding1.shape[1])],
    )
    embedding2_df = pd.DataFrame(
        embedding2,
        columns=[f"{embedding2_name}_{i}" for i in range(embedding2.shape[1])],
    )

    combined_embeddings_df = pd.concat([embedding1_df, embedding2_df], axis=1)

    return combined_embeddings_df


def process_and_save_embeddings(
    embedding_processor: EmbeddingProcessor,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    embedding_type: str,
    project_dir: Path,
) -> None:
    """
    Processes and saves embeddings for the training and test data.

    Parameters:
    embedding_processor (EmbeddingProcessor): The EmbeddingProcessor object.
    df_train (DataFrame): The training data.
    df_test (DataFrame): The test data.
    embedding_type (str): The type of the embeddings.
    project_dir (str): The path to the project directory.
    """
    logging.info(f"Computing {embedding_type} embeddings ...")

    # dynamically call the right method
    compute_embedding_method = getattr(
        embedding_processor, f"compute_{embedding_type}_embedding"
    )

    # Combined embeddings
    train_combined = compute_embedding_method(df_train["Combined_Text"])
    test_combined = compute_embedding_method(df_test["Combined_Text"])

    # Separate embeddings
    train_proc_desc = compute_embedding_method(df_train["Process_description"])
    train_legal_text = compute_embedding_method(df_train["Text"])
    test_proc_desc = compute_embedding_method(df_test["Process_description"])
    test_legal_text = compute_embedding_method(df_test["Text"])

    # Saving combined embeddings
    save_df_with_embeddings(
        df_train,
        train_combined,
        embedding_type,
        project_dir / f"data/processed/embeddings/{embedding_type}_train_combined",
    )
    save_df_with_embeddings(
        df_test,
        test_combined,
        embedding_type,
        project_dir / f"data/processed/embeddings/{embedding_type}_test_combined",
    )

    # Concatenate separate embeddings and save
    train_separate_df = concat_embeddings_with_df(
        train_proc_desc,
        train_legal_text,
        f"{embedding_type}_proc_desc",
        f"{embedding_type}_legal_text",
    )
    test_separate_df = concat_embeddings_with_df(
        test_proc_desc,
        test_legal_text,
        f"{embedding_type}_proc_desc",
        f"{embedding_type}_legal_text",
    )

    save_df_with_embeddings(
        df_train,
        train_separate_df,
        embedding_type,
        project_dir / f"data/processed/embeddings/{embedding_type}_train_separate",
    )
    save_df_with_embeddings(
        df_test,
        test_separate_df,
        embedding_type,
        project_dir / f"data/processed/embeddings/{embedding_type}_test_separate",
    )

    # Save just legal text embeddings for further analysis
    save_df_with_embeddings(
        df_train,
        train_legal_text,
        embedding_type,
        project_dir / f"data/processed/embeddings/{embedding_type}_train_legal_text",
    )
    save_df_with_embeddings(
        df_test,
        test_legal_text,
        embedding_type,
        project_dir / f"data/processed/embeddings/{embedding_type}_test_legal_text",
    )


def main():
    """
    The main function that orchestrates the creation of embeddings from preprocessed data.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Embeddings out of the preprocessed data")
    project_dir = Path(__file__).resolve().parents[2]
    embedding_processor = EmbeddingProcessor()

    # load data and combine text and process description
    df_train = pd.read_csv(
        project_dir / "data/processed/training_data_preprocessed.csv"
    )
    df_test = pd.read_csv(
        project_dir / "data/evaluation/gold_standard_preprocessed.csv"
    )

    df_train["Combined_Text"] = df_train["Process_description"] + " " + df_train["Text"]
    df_test["Combined_Text"] = df_test["Process_description"] + " " + df_test["Text"]

    logger.info("Training Embeddings models ...")
    # training embeddings models
    embedding_processor.train_model("tfidf")
    embedding_processor.train_model("word2vec")
    embedding_processor.train_model("glove")
    embedding_processor.train_model("fasttext")

    # Loop through each embedding type and create/save the embeddings
    for embedding_type in ["word2vec", "glove", "fasttext", "tfidf", "bert", "gpt"]:
        process_and_save_embeddings(
            embedding_processor, df_train, df_test, embedding_type, project_dir
        )
    logger.info("Embeddings have been saved in /data/processed/embeddings folder")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
