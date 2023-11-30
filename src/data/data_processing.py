import json
import pandas as pd
from sklearn.model_selection import train_test_split


def process_matching_data(process_name:str,excel_file:pd.ExcelFile, sheet_name:str, output_file:str):
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

    df_labels = matching_data[["document_title",'requirement_text', '0 = not relevant; 1 = business compliance relevance; 2 = (customer) informative relevance']].copy()
    df_labels = df_labels.rename(columns={
        '0 = not relevant; 1 = business compliance relevance; 2 = (customer) informative relevance': 'label',
        'requirement_text': 'Text'
    })
    

    df_labels['label'] = df_labels['label'].replace(2, 1)
    df_labels.insert(0, 'Process', process_name)
    

    df_labels.to_csv(output_file, index=False)
    
    print(f'Processed {sheet_name}:')
    print(df_labels['label'].value_counts())

def merge_lables():
    """
    Merges the labels from the three processes into one CSV file.

    This function reads the CSV files containing the labels for the three processes,
    merges them into one DataFrame, and exports it to a CSV file.
    """
    travel_insurance_claim_labels = pd.read_csv('../data/interim/travel_insurance_claim_labels.csv')
    know_your_customer_labels = pd.read_csv('../data/interim/know_your_customer_labels.csv')
    hiring_employee_labels = pd.read_csv('../data/interim/hiring_employee_labels.csv')

    all_labels = pd.concat([travel_insurance_claim_labels, know_your_customer_labels, hiring_employee_labels])
    all_labels.to_excel('../data/processed/all_labels.xlsx', index=False)
 
def read_excel_file(filepath, columns):
    """
    Reads an Excel file and returns a DataFrame with specified columns.
    """
    return pd.read_excel(filepath)[columns]

def read_text_file(filepath):
    """
    Reads a text file and returns its content as a string.
    """
    with open(filepath, 'r') as file:
        return file.read().strip()

def create_finetuning_data(all_labels, process_to_file, prompt_creator):
    """
    General function to create fine-tuning data.
    """
    all_labels['prompt'] = all_labels.apply(prompt_creator, axis=1, process_to_file=process_to_file)
    all_labels['completion'] = all_labels['label'].apply(lambda x: ' ' + str(x) + "###")
    all_labels = all_labels[['prompt', 'completion']]
    return train_test_split(all_labels, test_size=0.20, random_state=42)

def create_legacy_prompt(row, process_to_file):
    """
    Creates a prompt for the legacy fine-tuning format.
    """
    process_description = read_text_file(process_to_file[row['Process']])
    return f"Process: {process_description}\n\nText: {row['Text']}\n\nRelevant:"

def create_gpt3_5_prompt(row, process_to_file):
    """
    Creates a prompt for the GPT-3.5 fine-tuning format.
    """
    process_description = read_text_file(process_to_file[row['Process']])
    user_message = f"Process Description: {process_description}.\nText to Classify: {row['Text']}\n0=Not Relevant 1=Relevant"
    assistant_message = f"{row['label']}"
    return {"messages": [{"role": "system", "content": "Determine if the text is relevant to the process description."},
                         {"role": "user", "content": user_message},
                         {"role": "assistant", "content": assistant_message}]}

def save_to_jsonl(data, filepath):
    """
    Saves the data to a JSONL file.
    """
    with open(filepath, "w") as file:
        for item in data:
            file.write(f"{json.dumps(item)}\n")

def main():
    # australinen_excel = pd.ExcelFile("../data/external/Australia_Use_Cases.xlsx")
    # australinen_excel_2 = pd.ExcelFile("../data/external/Addition_Australia_data_small_use_case_3_hire_employee.xlsx")

    # process_matching_data("Travel Insurance Claim", australinen_excel, "1_matching_reordered", "../data/interim/travel_insurance_claim_labels.csv")
    # process_matching_data("Know Your Customer", australinen_excel, "2_matching_reordered", "../data/interim/know_your_customer_labels.csv")
    # process_matching_data("Hiring Employee", australinen_excel_2, "training_matching", "../data/interim/hiring_employee_labels.csv")
    # merge_lables()

    process_to_file = {
        'Hiring Employee': '../data/external/processes/textual_description/hiring_employee.txt',
        'Know Your Customer': '../data/external/processes/textual_description/know_your_customer.txt',
        'Travel Insurance Claim': '../data/external/processes/textual_description/travel_insurance_claim_process.txt'
    }
    all_labels = read_excel_file('../data/processed/all_labels.xlsx', ['Process', 'Text', 'label'])
    train, val = create_finetuning_data(all_labels, process_to_file, create_legacy_prompt)
    save_to_jsonl(train, "../data/processed/train_davinci_classification.jsonl")
    save_to_jsonl(val, "../data/processed/val_davinci_classification.jsonl")

    # Main execution for GPT-3.5 format
    all_labels = read_excel_file('../data/processed/all_labels.xlsx', ['Process', 'Text', 'label'])
    train, val = create_finetuning_data(all_labels, process_to_file, create_gpt3_5_prompt)
    save_to_jsonl(train, "../data/processed/train_gpt_finetuning.jsonl")
    save_to_jsonl(val, "../data/processed/val_gpt_finetuning.jsonl")




if __name__ == "__main__":
    main()