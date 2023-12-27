import pandas as pd


# This class turns the raw data from Use Case Data(2).xlsx into the dataset used in the project.
class DataProcessor:
    def __init__(self, excel_filepath, process_to_file):
        self.excel_file = pd.ExcelFile(excel_filepath)
        self.process_to_file = process_to_file

    def process_matching_data(self, process_name: str, sheet_name: str):
        matching_data = self.excel_file.parse(sheet_name)
        print(matching_data.columns)

        if process_name in [
            "Hiring Employee",
            "Know Your Customer",
            "Travel Insurance Claim",
        ]:
            label_column = "0 = not relevant; 1 = business compliance relevance; 2 = (customer) informative relevance"
        else:
            label_column = "0 = not relevant; 1 = relevant"

        df_labels = matching_data[["requirement_text", label_column]].copy()
        df_labels = df_labels.rename(
            columns={label_column: "Label", "requirement_text": "Text"}
        )
        df_labels["Label"] = df_labels["Label"].replace(2, 1)
        df_labels["Label"] = (
            pd.to_numeric(df_labels["Label"], errors="coerce").fillna(0).astype(int)
        )
        df_labels.insert(0, "Process", process_name)

        process_description = self.read_process_description(process_name)
        df_labels["Process_description"] = process_description
        return df_labels

    def read_process_description(self, process_name):
        with open(self.process_to_file[process_name], "r") as file:
            return file.read().strip()

    def create_gold_standard_subset(self, df, sample_sizes):
        subsets = []

        for process_name, (n_positive, n_negative) in sample_sizes.items():
            process_df = df[df["Process"] == process_name]
            print(process_name)
            positive_samples = process_df[process_df["Label"] == 1].sample(n=n_positive)
            negative_samples = process_df[process_df["Label"] == 0].sample(n=n_negative)

            subsets.append(positive_samples)
            subsets.append(negative_samples)

        gold_standard_subset = pd.concat(subsets, ignore_index=True)
        return gold_standard_subset
