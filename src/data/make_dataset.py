# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from data_processor import DataProcessor

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed andin ../evaluation ) under the names training_data_preposssed.csv,
    final_labels_with_descriptions.xlsx and gold_standard_preprocessed.csv .

    Parameters:
    input_filepath (str): The path to the input file.
    output_filepath (str): The path to the output file.
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    # Define the project directory
    project_dir = Path(__file__).resolve().parents[2]

    # Define the path to the data file
    data_file_path = project_dir / "data/raw/Use_Case_Data_With_All_Proccesses.xlsx"

    # Define the mapping of process names to file paths
    process_to_file = {
        # ...
    }

    # Define the mapping of process names to sheet names
    process_sheet_mapping = {
        # ...
    }

    # Initialize the DataProcessor
    processor = DataProcessor(data_file_path, process_to_file)

    # Initialize an empty DataFrame to store all processed data
    all_processed_data = pd.DataFrame()

    # Process the data for each process
    for process_name, sheet_name in process_sheet_mapping.items():
        processed_data = processor.process_matching_data(process_name, sheet_name)
        all_processed_data = pd.concat([all_processed_data, processed_data])

    # Save the processed data to an Excel file and a CSV file
    all_processed_data.to_excel(
        project_dir / "data/processed/final_labels_with_description.xlsx", index=False
    )
    all_processed_data.to_csv(
        project_dir / "data/processed/final_labels_with_description.csv", index=False
    )

    # Define the sample sizes for the gold standard subset
    gold_standard_samples = {
        # ...
    }

    # Create the gold standard subset
    gold_standard_subset, gold_standard_indices = processor.create_gold_standard_subset(
        all_processed_data, gold_standard_samples
    )

    # Save the gold standard subset to a CSV file
    gold_standard_subset.to_csv(
        project_dir / "data/evaluation/gold_standard.csv", index=False
    )

    # Reset the indices of all_processed_data
    all_processed_data.reset_index(drop=True, inplace=True)

    # Exclude the gold standard samples from all_processed_data to create the training data
    train_data = all_processed_data.drop(gold_standard_indices).reset_index(drop=True)

    # Preprocess the data for word embedding
    train_data["Process_description"] = processor.preprocess_lemma(
        train_data["Process_description"]
    )
    train_data["Text"] = processor.preprocess_lemma(train_data["Text"])
    train_data["Process_description"] = processor.preprocess_statements_nltk(
        train_data["Process_description"]
    )
    train_data["Text"] = processor.preprocess_statements_nltk(train_data["Text"])

    # Preprocess the gold standard subset for word embedding
    gold_standard_subset["Process_description"] = processor.preprocess_lemma(
        gold_standard_subset["Process_description"]
    )
    gold_standard_subset["Text"] = processor.preprocess_lemma(
        gold_standard_subset["Text"]
    )
    gold_standard_subset["Process_description"] = processor.preprocess_statements_nltk(
        gold_standard_subset["Process_description"]
    )
    gold_standard_subset["Text"] = processor.preprocess_statements_nltk(
        gold_standard_subset["Text"]
    )

    # Save the preprocessed training data and gold standard subset to CSV files
    train_data.to_csv(
        project_dir / "data/processed/training_data_preprocessed.csv",
        index=False,
    )
    gold_standard_subset.to_csv(
        project_dir / "data/evaluation/gold_standard_preprocessed.csv", index=False
    )

    # Log the lengths of the datasets
    logger.info(f"Length of all_processed_data: {len(all_processed_data)}")
    logger.info(f"Length of gold_standard_subset: {len(gold_standard_subset)}")
    logger.info(f"Length of train_data: {len(train_data)}")


if __name__ == "__main__":
    # Set the logging format
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Run the main function
    main()