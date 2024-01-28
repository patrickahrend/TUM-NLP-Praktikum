import pickle
from pathlib import Path
import os
import pandas as pd
import requests
from typing import List
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
)


def load_pickle(file_path: Path) -> pd.DataFrame:
    """
    Load a pickle file.

    Parameters:
    file_path (str): The path to the pickle file.

    Returns:
    object: The content of the pickle file.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


class UserInterface:
    """
    A class used to create a user interface for a legal text classifier.

    Attributes:
    api_url (str): The URL of the API.
    base_path (str): The base path of the application.
    text_input (str): The text input from the user.
    selected_model (str): The selected model by the user.
    selected_embedding (str): The selected embedding by the user.
    process_description (str): The process description.
    result_path (str): The path to the result.
    test_data (DataFrame): The test data.
    processes (list): The list of processes.
    dataset_type (str): The type of the dataset.
    is_tuned (bool): A flag indicating whether the model is tuned.
    model_options (list): The list of model options.

    Methods:
    initialize_app(): Initialize the Streamlit app.
    load_data(embedding_type: str, dataset_type: str): Load the data for the selected embedding and dataset type.
    load_model_results(): Load the model results.
    display_all_test_data_points(): Display all test data points.
    classify_text_with_embeddings(embeddings: list): Classify text with embeddings.
    classify_new_text(text_input: str, process_description: str, selected_model: str, selected_embedding: str, dataset_type: str, is_tuned: bool): Classify new text.
    display_results(): Display the results.
    """

    def __init__(self, api_url: str, base_path: Path):
        """
        Initialize the UserInterface class.

        Parameters:
        api_url (str): The URL of the API.
        base_path (str): The base path of the application.
        """
        self.api_url = api_url
        self.base_path = base_path
        self.initialize_app()
        self.text_input = None
        self.selected_model = None
        self.selected_embedding = None
        self.process_description = None
        self.result_path = None
        self.test_data = None
        self.processes = None
        self.dataset_type = None
        self.is_tuned = None
        self.model_options: List[str] = []

    def initialize_app(self):
        """
        Initialize the Streamlit app.
        """
        st.set_page_config(
            layout="wide", page_icon="🚀", page_title="Legal Text Classifier"
        )
        st.title("Legal Text Classifier")

        (
            tab1,
            tab2,
        ) = st.tabs(["Classify", "Model Evaluation"])

        with tab1:
            self.model_options = [
                "Logistic_Regression",
                "RandomForestClassifier",
                "SVC",
                "DecisionTreeClassifier",
                "BernoulliNB",
                "GaussianNB",
                "Perceptron",
                "SGDClassifier",
                "KNeighborsClassifier",
                "BertForClassification-Base",
                "BertForClassification-Large",
                "GPT3.5",
                "Recurrent Neural Network",
                "Rule-Based Mean Centroid",
                "Rule-Based Cosine Similarity",
            ]
            self.selected_model = st.selectbox("Select a model", self.model_options)

            embedding_options = [
                "tfidf",
                "word2vec",
                "fasttext",
                "bert",
                "gpt",
                "glove",
            ]

            # only allow bert for bert models
            if self.selected_model in [
                "BertForClassification-Base",
                "BertForClassification-Large",
            ]:
                embedding_options = ["bert"]
            # only allow gpt for gpt models
            elif self.selected_model == "GPT3.5":
                embedding_options = ["gpt"]
            self.selected_embedding = st.selectbox(
                "Select an embedding", embedding_options
            )

            test_set = st.toggle("Use Test Set", value=False)

            if test_set:
                self.dataset_type = st.radio(
                    "Select dataset type", ("separate", "combined")
                )
                self.is_tuned = st.toggle("Hyperparameter Tuned", value=False)
                if self.selected_model in [
                    "Recurrent Neural Network",
                    "BertForClassification-Base",
                    "BertForClassification-Large",
                    "GPT3.5",
                    "Rule-Based Mean Centroid",
                    "Rule-Based Cosine Similarity",
                ]:
                    self.load_model_results()
                else:
                    self.load_data(self.selected_embedding, self.dataset_type)

                self.display_all_test_data_points()

            else:
                ## Mocked of how user input could look like
                self.text_input = st.text_area("Enter the text passage here:")

                self.process_description = st.text_area(
                    "Enter the process description here:"
                )
                st.button("Classify", key="classify_mock")

        ## here show model evaluation results
        with tab2:
            self.display_results()

    def load_data(self, embedding_type: str, dataset_type: str) -> None:
        """
        Load the data for the selected embedding and dataset type.

        Parameters:
        embedding_type (str): The type of the embedding.
        dataset_type (str): The type of the dataset.
        """
        embedding_path = self.base_path / "data/processed/embeddings"
        all_embeddings = list(embedding_path.glob("*.pkl"))
        filtered_files = [
            f
            for f in all_embeddings
            if f"test_{dataset_type}" in str(f)
            and str(f).startswith(str(embedding_path / embedding_type))
        ]
        if not filtered_files:
            raise FileNotFoundError(
                f"No embeddings found for type {embedding_type} and dataset {dataset_type}"
            )
        embedding = load_pickle(filtered_files[0])
        self.processes = embedding["Process"].unique().tolist()
        self.test_data = embedding

    def load_model_results(self) -> None:
        """
        Load the model results.
        """
        model_results_path = (
            self.base_path
            / "references"
            / "model results"
            / "test_data_with_advanced_predictions.csv"
        )
        self.test_data = pd.read_csv(model_results_path)
        self.processes = self.test_data["Process"].unique().tolist()

    def display_all_test_data_points(self) -> None:
        """
        Display all test data points.
        """
        process_options = ["All Processes"] + self.processes
        selected_process = st.selectbox("Select a Process", process_options)

        # filter data based on the selected process
        filtered_data = (
            self.test_data
            if selected_process == "All Processes"
            else self.test_data[self.test_data["Process"] == selected_process]
        )

        if not filtered_data.empty:
            self.process_description = (
                filtered_data["Process_description"].iloc[0]
                if selected_process != "All Processes"
                else None
            )
            st.text_area(
                "Process Description",
                value=self.process_description,
                height=300,
                disabled=True,
            )

            # Concatenate all the texts related to the process into one document
            concatenated_texts = "\n\n".join(filtered_data["Text"])

            st.text_area(
                "All Texts", value=concatenated_texts, height=600, disabled=True
            )
            self.text_input = concatenated_texts

            if st.button("Classify"):
                results = []
                # on test set
                for index, row in filtered_data.iterrows():
                    if self.selected_model == "BertForClassification-Base":
                        results.append(
                            (
                                index,
                                row["Text"],
                                int(row["Bert_Base_Prediction"]),
                                row["Label"],
                            )
                        )
                    elif self.selected_model == "BertForClassification-Large":
                        results.append(
                            (
                                index,
                                row["Text"],
                                int(row["Bert_Large_Prediction"]),
                                row["Label"],
                            )
                        )
                    elif self.selected_model == "GPT3.5":
                        results.append(
                            (
                                index,
                                row["Text"],
                                int(row["GPT_Prediction"]),
                                row["Label"],
                            )
                        )
                    elif self.selected_model == "Recurrent Neural Network":
                        prediction_column = (
                            f"{self.selected_embedding}_RNN_Prediction_Tuned"
                            if self.is_tuned
                            else f"{self.selected_embedding}_RNN_Prediction"
                        )
                        results.append(
                            (
                                index,
                                row["Text"],
                                int(row[prediction_column]),
                                row["Label"],
                            )
                        )
                    elif self.selected_model == "Rule-Based Mean Centroid":
                        results.append(
                            (
                                index,
                                row["Text"],
                                int(
                                    row[
                                        f"{self.selected_embedding}_Mean_Centroid_Prediction"
                                    ]
                                ),
                                row["Label"],
                            )
                        )
                    elif self.selected_model == "Rule-Based Cosine Similarity":
                        results.append(
                            (
                                index,
                                row["Text"],
                                int(
                                    row[
                                        f"{self.selected_embedding}_Cosine_Similarity_Prediction"
                                    ]
                                ),
                                row["Label"],
                            )
                        )
                    else:
                        embeddings = row[5:].values.tolist()
                        predicted_label = self.classify_text_with_embeddings(
                            embeddings=embeddings,
                        )
                        # Store the results along with the correct label for display later
                        results.append(
                            (index, row["Text"], predicted_label, row["Label"])
                        )

                # Quantitative evaluation
                true_labels = [label for _, _, _, label in results]
                predicted_labels = [pred for _, _, pred, _ in results]

                accuracy = accuracy_score(true_labels, predicted_labels)
                precision = precision_score(
                    true_labels, predicted_labels, average="weighted"
                )
                recall = recall_score(true_labels, predicted_labels, average="weighted")
                f1 = f1_score(true_labels, predicted_labels, average="weighted")
                correct_predictions = sum(
                    t == p for t, p in zip(true_labels, predicted_labels)
                )
                incorrect_predictions = len(true_labels) - correct_predictions

                st.subheader("Classification Metrics")
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
                    with col1:
                        st.metric(label="Accuracy", value=round(accuracy, 2))
                    with col2:
                        st.metric(
                            label="Precision (Weighted)", value=round(precision, 2)
                        )
                    with col3:
                        st.metric(label="Recall (Weighted)", value=round(recall, 2))
                    with col4:
                        st.metric(label="F1 Score (Weighted)", value=round(f1, 2))
                    with col5:
                        st.metric(
                            label="Correct Predictions", value=correct_predictions
                        )
                    with col6:
                        st.metric(
                            label="Incorrect Predictions", value=incorrect_predictions
                        )

                # Sort the results to show the wrongly predicted results first
                sorted_results = sorted(results, key=lambda x: x[2] == x[3])

                # Qualitative evaluation
                st.subheader("Classification Results")
                for (
                    index,
                    original_text,
                    predicted_label,
                    correct_label,
                ) in sorted_results:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            st.text_area(
                                "Text",
                                value=original_text,
                                height=75,
                                key=f"text_{index}",
                            )
                        with col2:
                            # Remove the brackets from the label for display
                            st.metric(label="Predicted", value=predicted_label)
                        with col3:
                            st.metric(label="Actual", value=correct_label)
                        with col4:
                            st.caption("Status")
                            if predicted_label == correct_label:
                                st.success("Correctly Predicted")
                            else:
                                st.error("Wrongly Predicted")
                    st.divider()

        else:
            st.write("No data available for the selected process.")

    def classify_text_with_embeddings(self, embeddings: list) -> int:
        """
        Classify text with embeddings.

        Parameters:
        embeddings (list): The list of embeddings.

        Returns:
        int: The classification result.
        """
        request_data = {
            "model_name": self.selected_model,
            "embedding_name": self.selected_embedding,
            "dataset_type": self.dataset_type,
            "is_tuned": self.is_tuned,
            "embeddings": embeddings,
        }

        print("Sending the following data to /classify/ endpoint:", request_data)

        # Send request to FastAPI server
        response = requests.post(f"{self.api_url}/classify/", json=request_data)
        if response.status_code == 200:
            return response.json()["classification"]
        else:
            st.error(f"Error in classification: {response.status_code}")
            return -1

    def classify_new_text(
        self,
        text_input: str,
        process_description: str,
        selected_model: str,
        selected_embedding: str,
        dataset_type: str,
        is_tuned: bool,
    ) -> int:
        """
        Classify for new text to classify

        Parameters:
        text_input (str): The text input from the user.
        process_description (str): The process description.
        selected_model (str): The selected model by the user.
        selected_embedding (str): The selected embedding by the user.
        dataset_type (str): The type of the dataset.
        is_tuned (bool): A flag indicating whether the model is tuned.

        Returns:
        int: The classification result.
        """
        request_data = {
            "text_passage": text_input,
            "process_description": process_description,
            "model_name": selected_model,
            "embedding_name": selected_embedding,
            "dataset_type": dataset_type,
            "is_tuned": is_tuned,
        }
        print(request_data)

        # Send request to FastAPI server
        response = requests.post(f"{self.api_url}/classify/", json=request_data)
        if response.status_code == 200:
            return response.json()["classification"]
        else:
            st.error("Error in classification")
            return -1

    def display_results(self) -> None:
        """
        Display the results in the 2nd tab
        """
        model_results_dir = self.base_path / "references" / "model results"
        feature_importance_dir = self.base_path / "references" / "feature importance"

        model_results_files = list(model_results_dir.glob("*.csv"))
        feature_importance_files = list(feature_importance_dir.glob("*.csv"))

        selected_model_file = st.selectbox(
            "Select a model result file", model_results_files
        )
        if selected_model_file:
            df_model_results = pd.read_csv(selected_model_file)

            selected_columns = st.multiselect(
                "Select columns to display",
                df_model_results.columns.tolist(),
                default=df_model_results.columns.tolist(),
            )
            df_model_results = df_model_results[selected_columns]
            st.dataframe(
                df_model_results.sort_values(by=selected_columns[0], ascending=True)
            )

        selected_feature_file = st.selectbox(
            "Select a feature importance file", feature_importance_files
        )
        if selected_feature_file:
            df_feature_importance = pd.read_csv(selected_feature_file)
            selected_columns = st.multiselect(
                "Select columns to display",
                df_feature_importance.columns.tolist(),
                default=df_feature_importance.columns.tolist(),
            )
            df_feature_importance = df_feature_importance[selected_columns]
            st.dataframe(
                df_feature_importance.sort_values(
                    by=selected_columns[0], ascending=True
                )
            )


if __name__ == "__main__":
    backend_env_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    base_path_env = os.getenv("BASE_PATH")

    if base_path_env is not None:
        base_path = Path(base_path_env)
    else:
        base_path = Path(__file__).resolve().parents[1]

    app = UserInterface(backend_env_url, base_path)
