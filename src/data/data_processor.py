import nltk
import pandas as pd
import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))


class DataProcessor:
    """
    This class is responsible for processing raw data from an Excel file into a dataset used in the project.
    It uses the Spacy library for natural language processing tasks.
    Turns the raw data from Use Case Data(2).xlsx into the dataset used in the project.
    """

    def __init__(self, excel_filepath, process_to_file, nlp_model="en_core_web_sm"):
        """
        Initializes the DataProcessor object.

        Parameters:
        excel_filepath (str): The path to the Excel file to be processed.
        process_to_file (dict): A dictionary mapping process names to file paths.
        nlp_model (str): The name of the Spacy model to be used for natural language processing tasks.
        """
        self.excel_file = pd.ExcelFile(excel_filepath)
        self.process_to_file = process_to_file
        self.nlp = spacy.load(nlp_model)

    def process_matching_data(self, process_name: str, sheet_name: str):
        """
        Processes the matching data from the Excel file.

        Parameters:
        process_name (str): The name of the process to be processed.
        sheet_name (str): The name of the sheet in the Excel file where the data is located.

        Returns:
        DataFrame: A DataFrame containing the processed data.
        """
        matching_data = self.excel_file.parse(sheet_name)
        # Define the label column based on the process name
        if process_name in [
            "Hiring Employee",
            "Know Your Customer",
            "Travel Insurance Claim",
        ]:
            label_column = "0 = not relevant; 1 = business compliance relevance; 2 = (customer) informative relevance"
        else:
            label_column = "0 = not relevant; 1 = relevant"

        # Create a DataFrame with the relevant columns and rename them
        df_labels = matching_data[["requirement_text", label_column]].copy()
        df_labels = df_labels.rename(
            columns={label_column: "Label", "requirement_text": "Text"}
        )
        # Replace 2 with 1 in the Label column and convert it to integer
        df_labels["Label"] = df_labels["Label"].replace(2, 1)
        df_labels["Label"] = (
            pd.to_numeric(df_labels["Label"], errors="coerce").fillna(0).astype(int)
        )
        # Insert a new column with the process name
        df_labels.insert(0, "Process", process_name)

        # Add a column with the process description
        process_description = self.read_process_description(process_name)
        df_labels["Process_description"] = process_description
        return df_labels

    def read_process_description(self, process_name):
        """
        Reads the process description from a file.

        Parameters:
        process_name (str): The name of the process.

        Returns:
        str: The process description.
        """
        with open(self.process_to_file[process_name], "r") as file:
            return file.read().strip()

    @staticmethod
    def create_gold_standard_subset(df, sample_sizes):
        """
        Creates a gold standard subset from the DataFrame.

        Parameters:
        df (DataFrame): The DataFrame to create the subset from.
        sample_sizes (dict): A dictionary mapping process names to sample sizes.

        Returns:
        DataFrame, list: The gold standard subset and the indices of the sampled rows.
        """
        subsets = []
        gold_standard_indices = []
        for process_name, (n_positive, n_negative) in sample_sizes.items():
            process_df = df[df["Process"] == process_name]
            positive_samples = process_df[process_df["Label"] == 1].sample(n=n_positive)
            negative_samples = process_df[process_df["Label"] == 0].sample(n=n_negative)

            # Store the indices of the sampled rows
            gold_standard_indices.extend(positive_samples.index.tolist())
            gold_standard_indices.extend(negative_samples.index.tolist())

            subsets.append(positive_samples)
            subsets.append(negative_samples)

        gold_standard_subset = pd.concat(subsets, ignore_index=True)
        return gold_standard_subset, gold_standard_indices

    def preprocess_lemma(self, statements):
        """
        Preprocesses the statements by lemmatizing the words.

        Parameters:
        statements (Series): The statements to be preprocessed.

        Returns:
        Series: The preprocessed statements.
        """
        return statements.fillna("").apply(
            lambda x: " ".join([token.lemma_.lower() for token in self.nlp(x)])
        )

    @staticmethod
    def preprocess_text_nltk(text):
        """
        Preprocesses the text by tokenizing the words and removing punctuation.

        Parameters:
        text (str): The text to be preprocessed.

        Returns:
        str: The preprocessed text.
        """
        tokens = word_tokenize(text)
        punctuation_to_remove = {".", ","}
        tokens = [word for word in tokens if word not in punctuation_to_remove]

        return " ".join(tokens)

    def preprocess_statements_nltk(self, statements):
        """
        Preprocesses the statements by applying the preprocess_text_nltk method.

        Parameters:
        statements (Series): The statements to be preprocessed.

        Returns:
        Series: The preprocessed statements.
        """
        return statements.fillna("").apply(self.preprocess_text_nltk)
