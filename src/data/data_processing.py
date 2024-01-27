 import json

import pandas as pd
from sklearn.model_selection import train_test_split


## For later refactor this file to use the DataProcessor class
def create_gpt_finetuning_data():
    """
    Creates the data for fine-tuning the GPT-3.5 model.

    This function reads an Excel file containing the labels and associated processes
    for a dataset, appends the appropriate process description from corresponding .txt files,
    and formats it into a JSONL file with prompts and completions suitable for fine-tuning.

    The prompts are structured with the process description, followed by the text,
    and ending with 'Relevant:' to indicate where the model should give its classification.
    The completion is the label '0' or '1'. The dataset is split into training and validation sets,
    which are then saved to separate JSONL files.
    I followed this best pracitise guide from OpenAI: https://platform.openai.com/docs/guides/legacy-fine-tuning/data-formatting
    """

    all_labels = pd.read_excel("../data/processed/all_labels.xlsx")
    all_labels = all_labels[["Process", "Text", "label"]]

    # Dictionary to map process names to file paths
    process_to_file = {
        "Hiring Employee": "../data/raw/processes/textual_description/hiring_employee.txt",
        "Know Your Customer": "../data/raw/processes/textual_description/know_your_customer.txt",
        "Travel Insurance Claim": "../data/raw/processes/textual_description/travel_insurance_claim.txt",
    }

    # Function to read the process description from a file
    def read_process_description(process_name):
        with open(process_to_file[process_name], "r") as file:
            return file.read().strip()

    # Function to create the prompt with process description and the label indicator 'Relevant:'
    def create_prompt(row):
        process_description = read_process_description(row["Process"])
        return f"Process: {process_description}\n\nText: {row['Text']}\n\nRelevant:"

    # Apply the function to each row to create the prompts
    all_labels["prompt"] = all_labels.apply(create_prompt, axis=1)

    # Make sure the completion starts with a whitespace and ends with the designated stop sequence
    stop_sequence = "###"
    all_labels["completion"] = all_labels["label"].apply(
        lambda x: " " + str(x) + stop_sequence
    )

    # Drop the original columns as they are no longer needed
    all_labels = all_labels[["prompt", "completion"]]

    train, val = train_test_split(all_labels, test_size=0.20, random_state=42)

    train.to_json(
        "../data/processed/train_davinci_classification.jsonl",
        orient="records",
        lines=True,
    )
    val.to_json(
        "../data/processed/val_davinci_classification.jsonl",
        orient="records",
        lines=True,
    )


def create_gpt3_5_fine_tuning_data():
    all_labels = pd.read_excel("../data/processed/all_labels.xlsx")
    all_labels = all_labels[["Process", "Text", "label"]]

    # Dictionary to map process names to file paths
    process_to_file = {
        "Hiring Employee": "../data/external/processes/textual_description/hiring_employee.txt",
        "Know Your Customer": "../data/external/processes/textual_description/know_your_customer.txt",
        "Travel Insurance Claim": "../data/external/processes/textual_description/travel_insurance_claim.txt",
    }

    # Function to read the process description from a file
    def read_process_description(process_name):
        with open(process_to_file[process_name], "r") as file:
            return file.read().strip()

    def create_gpt_prompt(row):
        process_description = read_process_description(row["Process"])
        user_message = f"Process Description: {process_description}.\nText to Classify: {row['Text']}\n0=Not Relevant 1=Relevant"
        assistant_message = f"{row['label']}"
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Determine if the text is relevant to the process description.",
                },
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        }

    all_labels["gpt_prompt"] = all_labels.apply(create_gpt_prompt, axis=1)

    train, val = train_test_split(all_labels, test_size=0.20, random_state=42)

    # Extracting only the GPT-3.5 formatted prompts
    train_gpt = train["gpt_prompt"].tolist()
    val_gpt = val["gpt_prompt"].tolist()

    # Save the formatted data
    with open("../data/processed/train_gpt_finetuning.jsonl", "w") as file:
        for item in train_gpt:
            file.write(f"{json.dumps(item)}\n")

    with open("../data/processed/val_gpt_finetuning.jsonl", "w") as file:
        for item in val_gpt:
            file.write(f"{json.dumps(item)}\n")


def main():
    australinen_excel = pd.ExcelFile("../data/raw/Use_Case_Data(2).xlsx")
    # australinen_excel_2 = pd.ExcelFile("../data/external/Addition_Australia_data_small_use_case_3_hire_employee.xlsx")

    # process_matching_data("GPT_5", australinen_excel, "8_GDPR_5_matching", "../data/interim/GPT_5.csv")
    # process_matching_data("GPT_6", australinen_excel, "9_GDPR_6_matching", "../data/interim/GPT_6.csv")
    # process_matching_data("SM2_1", australinen_excel, "11_SM_2.1_matching", "../data/interim/SM2_1.csv")
    # process_matching_data("SM2_2", australinen_excel, "12_SM_2.2_matching", "../data/interim/SM2_2.csv")

    # create_gpt_fine_tuning_data()
    # create_gpt3_5_fine_tuning_data()


if __name__ == "__main__":
    main()
