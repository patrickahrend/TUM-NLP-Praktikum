import json

import pandas as pd
from sklearn.model_selection import train_test_split


def process_matching_data(
    process_name: str, excel_file: pd.ExcelFile, sheet_name: str, output_file: str
):
    """
    Processes the matching data from an Excel file and exports it to a CSV file.

    This function reads a specified sheet from an Excel file, manipulates the data,
    and then exports it to a specified CSV file. It adds a new column 'Process' at
    the beginning of the DataFrame, which contains the name of the process. It also
    renames certain columns for clarity and adjusts the values in the 'label' column.

    Parameters:
    process_name (str): Name of the process to be added as a column.
    excel_file (pd.ExcelFile): The Excel file object to read data from.
    sheet_name (str): The name of the sheet in the Excel file to process.
    output_file (str): The file path where the resulting CSV will be saved.

    The function prints the original DataFrame columns, the count of labels after processing,
    and a confirmation message upon completion.
    """
    matching_data = excel_file.parse(sheet_name)

    print(matching_data.columns)

    df_labels = matching_data[
        ["document_title", "requirement_text", "0 = not relevant; 1 = relevant"]
    ].copy()
    df_labels = df_labels.rename(
        columns={"0 = not relevant; 1 = relevant": "label", "requirement_text": "Text"}
    )
    df_labels["label"] = df_labels["label"].replace(2, 1)
    df_labels.insert(0, "Process", process_name)
    df_labels.to_csv(output_file, index=False)

    print(f"Processed {sheet_name}:")
    print(df_labels["label"].value_counts())


def merge_labels():
    """
    Merges the labels from the three processes into one CSV file.

    This function reads the CSV files containing the labels for the three processes,
    merges them into one DataFrame, and exports it to a CSV file.
    """
    travel_insurance_claim_labels = pd.read_csv(
        "../data/interim/travel_insurance_claim_labels.csv"
    )
    know_your_customer_labels = pd.read_csv(
        "../data/interim/know_your_customer_labels.csv"
    )
    hiring_employee_labels = pd.read_csv("../data/interim/hiring_employee_labels.csv")
    gpt_5 = pd.read_csv("../data/interim/GPT_5.csv")
    gpt_6 = pd.read_csv("../data/interim/GPT_6.csv")
    sm2_1 = pd.read_csv("../data/interim/SM2_1.csv")
    sm2_2 = pd.read_csv("../data/interim/SM2_2.csv")

    all_labels = pd.concat(
        [
            travel_insurance_claim_labels,
            know_your_customer_labels,
            hiring_employee_labels,
            gpt_5,
            gpt_6,
            sm2_1,
            sm2_2,
        ]
    )
    all_labels.to_excel("../data/processed/all_labels.xlsx", index=False)


def create_gold_standard_data():
    all_labels = pd.read_excel("../data/processed/all_labels.xlsx")
    # Dictionary to map process names to file paths
    process_to_file = {
        "Hiring Employee": "../data/raw/processes/textual_description/hiring_employee.txt",
        "Know Your Customer": "../data/raw/processes/textual_description/know_your_customer.txt",
        "Travel Insurance Claim": "../data/raw/processes/textual_description/travel_insurance_claim_process.txt",
        "GPT_5": "../data/raw/processes/textual_description/gpt_5.txt",
        "GPT_6": "../data/raw/processes/textual_description/gpt_6.txt",
        "SM2_1": "../data/raw/processes/textual_description/SM2_1.txt",
        "SM2_2": "../data/raw/processes/textual_description/SM2_2.txt",
    }

    # Function to read the process description from a file
    def read_process_description(process_name):
        with open(process_to_file[process_name], "r") as file:
            return file.read().strip()

    # List to hold the new DataFrame rows
    new_rows = []

    # Iterate through each row in the all_labels DataFrame
    for index, row in all_labels.iterrows():
        process_description = read_process_description(row["Process"])
        new_row = {
            "process": row["Process"],
            "process_description": process_description,
            "text": row["Text"],
            "label": row["label"],
        }
        new_rows.append(new_row)

    # Create the final DataFrame
    final_df = pd.DataFrame(new_rows)

    final_df.to_csv("../data/evaluation/gold_standard.csv", index=False)

    travel_insurance_df = final_df[final_df["process"] == "Travel Insurance Claim"]
    travel_insurance_1 = travel_insurance_df[travel_insurance_df["label"] == 1].sample(
        n=10
    )
    travel_insurance_0 = travel_insurance_df[travel_insurance_df["label"] == 0].sample(
        n=88
    )

    kyc_insurance_df = final_df[final_df["process"] == "Know Your Customer"]
    kyc_insurance_1 = kyc_insurance_df[kyc_insurance_df["label"] == 1].sample(n=7)
    kyc_insurance_0 = kyc_insurance_df[kyc_insurance_df["label"] == 0].sample(n=55)

    hiring_insurance_df = final_df[final_df["process"] == "Hiring Employee"]
    hiring_insurance_1 = hiring_insurance_df[hiring_insurance_df["label"] == 1].sample(
        n=2
    )
    hiring_insurance_0 = hiring_insurance_df[hiring_insurance_df["label"] == 0].sample(
        n=10
    )

    gp5_insurance_df = final_df[final_df["process"] == "GPT_5"]
    gp5_insurance_1 = gp5_insurance_df[gp5_insurance_df["label"] == 1].sample(n=3)
    gp5_insurance_0 = gp5_insurance_df[gp5_insurance_df["label"] == 0].sample(n=29)

    gpt_6_insurance_df = final_df[final_df["process"] == "GPT_6"]
    gpt_6_insurance_1 = gpt_6_insurance_df[gpt_6_insurance_df["label"] == 1].sample(n=2)
    gpt_6_insurance_0 = gpt_6_insurance_df[gpt_6_insurance_df["label"] == 0].sample(
        n=18
    )

    sm21_insurance_df = final_df[final_df["process"] == "SM2_1"]
    sm21_insurance_1 = sm21_insurance_df[sm21_insurance_df["label"] == 1].sample(n=2)
    sm21_insurance_0 = sm21_insurance_df[sm21_insurance_df["label"] == 0].sample(n=14)

    sm22_insurance_df = final_df[final_df["process"] == "SM2_2"]
    sm22_insurance_1 = sm22_insurance_df[sm22_insurance_df["label"] == 1].sample(n=3)
    sm22_insurance_0 = sm22_insurance_df[sm22_insurance_df["label"] == 0].sample(n=19)

    subset_df = pd.concat(
        [
            travel_insurance_1,
            travel_insurance_0,
            kyc_insurance_0,
            kyc_insurance_1,
            hiring_insurance_0,
            hiring_insurance_1,
            gp5_insurance_1,
            gp5_insurance_0,
            gpt_6_insurance_0,
            gp5_insurance_1,
            gpt_6_insurance_1,
            sm21_insurance_0,
            sm21_insurance_1,
            sm22_insurance_0,
            sm22_insurance_1,
        ],
        ignore_index=True,
    )

    subset_df.to_csv("../data/evaluation/gold_standard_subset.csv", index=False)


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
        "Travel Insurance Claim": "../data/raw/processes/textual_description/travel_insurance_claim_process.txt",
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


def create_gpt3_5_finetuning_data():
    all_labels = pd.read_excel("../data/processed/all_labels.xlsx")
    all_labels = all_labels[["Process", "Text", "label"]]

    # Dictionary to map process names to file paths
    process_to_file = {
        "Hiring Employee": "../data/external/processes/textual_description/hiring_employee.txt",
        "Know Your Customer": "../data/external/processes/textual_description/know_your_customer.txt",
        "Travel Insurance Claim": "../data/external/processes/textual_description/travel_insurance_claim_process.txt",
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

    # create_gpt_finetuning_data()
    # create_gpt3_5_finetuning_data()

    create_gold_standard_data()


if __name__ == "__main__":
    main()
