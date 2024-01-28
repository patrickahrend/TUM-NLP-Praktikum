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

    process_to_file = {
        "Know Your Customer": project_dir
                              / "data/raw/processes/textual_description/know_your_customer.txt",
        "Hiring Employee": project_dir
                           / "data/raw/processes/textual_description/hiring_employee.txt",
        "Travel Insurance Claim": project_dir
                                  / "data/raw/processes/textual_description/travel_insurance_claim.txt",
        "GDPR_1": project_dir / "data/raw/processes/textual_description/GDPR_1.txt",
        "GDPR_2": project_dir / "data/raw/processes/textual_description/GDPR_2.txt",
        "GDPR_3": project_dir / "data/raw/processes/textual_description/GDPR_3.txt",
        "GDPR_4": project_dir / "data/raw/processes/textual_description/GDPR_4.txt",
        "GDPR_5": project_dir / "data/raw/processes/textual_description/GDPR_5.txt",
        "GDPR_6": project_dir / "data/raw/processes/textual_description/GDPR_6.txt",
        "GDPR_7": project_dir / "data/raw/processes/textual_description/GDPR_7.txt",
        "SM2_1": project_dir / "data/raw/processes/textual_description/SM2_1.txt",
        "SM2_2": project_dir / "data/raw/processes/textual_description/SM2_2.txt",
        "SM2_3": project_dir / "data/raw/processes/textual_description/SM2_3.txt",
        "SM2_5": project_dir / "data/raw/processes/textual_description/SM2_5.txt",
        "SM6_1": project_dir / "data/raw/processes/textual_description/SM6_1.txt",
        "SM6_3": project_dir / "data/raw/processes/textual_description/SM6_3.txt",
    }
    process_sheet_mapping = {
        "Travel Insurance Claim": "1_matching_reordered",
        "Know Your Customer": "2_matching_reordered",
        "Hiring Employee": "3_training_matching",
        "GDPR_1": "4_GDPR_1_matching",
        "GDPR_2": "5_GDPR_2_matching",
        "GDPR_3": "6_GDPR_3_matching",
        "GDPR_4": "7_GDPR_4_matching",
        "GDPR_5": "8_GDPR_5_matching",
        "GDPR_6": "9_GDPR_6_matching",
        "GDPR_7": "10_GDPR_7_matching",
        "SM2_1": "11_SM_2.1_matching",
        "SM2_2": "12_SM_2.2_matching",
        "SM2_3": "13_SM_2.3_matching",
        "SM2_5": "14_SM_2.5_matching",
        "SM6_1": "15_SM_6.1_matching",
        "SM6_3": "16_SM_6.3_matching",
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
        "Travel Insurance Claim": (10, 88),
        "Know Your Customer": (7, 55),
        "Hiring Employee": (2, 12),
        "GDPR_2": (3, 29),
        "GDPR_3": (3, 19),
        "SM2_1": (2, 14),
        "SM2_2": (3, 19),
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
