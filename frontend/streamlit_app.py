from pathlib import Path

import pandas as pd
import requests
import streamlit as st


class UserInterface:
    def __init__(self, api_url):
        self.api_url = api_url
        self.initialize_app()
        self.text_input = None
        self.selected_model = None
        self.model_options = None
        self.selected_embedding = None
        self.process_description = None
        self.result_path = "results.csv"

    def initialize_app(self):
        st.set_page_config(
            layout="wide", page_icon="🚀", page_title="Legal Text Classifier"
        )
        st.title("Legal Text Classifier")

        tab1, tab2, tab3 = st.tabs(
            ["Classify", "Hyperparameter Tuning", "Model Evaluation"]
        )

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

            self.selected_embedding = st.selectbox(
                "Select an embedding", embedding_options
            )

            self.text_input = st.text_area("Enter the text passage here:")

            self.process_description = st.text_area(
                "Enter the process description here:"
            )

            if st.button("Classify"):
                self.classify_text(
                    self.text_input,
                    self.process_description,
                    self.selected_model,
                    self.selected_embedding,
                )

        with tab2:
            ### here show hyperparamter tuning results
            st.write("Showing the results of the classification here")
            pass

        ## here show model evaluation results
        with tab3:
            self.display_results()

    def display_results(self):
        # Use Path for more robust path handling
        base_path = Path(__file__).parents[
            1
        ]  # Adjust the number based on your directory structure
        model_results_dir = base_path / "references" / "model results"
        feature_importance_dir = base_path / "references" / "feature importance"

        # List CSV files in each directory
        model_results_files = list(model_results_dir.glob("*.csv"))
        feature_importance_files = list(feature_importance_dir.glob("*.csv"))

        # Dropdown to select model result file
        selected_model_file = st.selectbox(
            "Select a model result file", model_results_files
        )
        if selected_model_file:
            df_model_results = pd.read_csv(selected_model_file)
            # Option to select columns
            selected_columns = st.multiselect(
                "Select columns to display",
                df_model_results.columns.tolist(),
                default=df_model_results.columns.tolist(),
            )
            df_model_results = df_model_results[selected_columns]
            st.dataframe(
                df_model_results.sort_values(by=selected_columns[0], ascending=True)
            )

        # Dropdown to select feature importance file
        selected_feature_file = st.selectbox(
            "Select a feature importance file", feature_importance_files
        )
        if selected_feature_file:
            df_feature_importance = pd.read_csv(selected_feature_file)
            # Option to select columns
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

    def classify_text(
        self, text_input, process_description, selected_model, selected_embedding
    ):
        request_data = {
            "text_passage": text_input,
            "process_description": process_description,
            "model_name": selected_model,
            "embedding_name": selected_embedding,
        }
        print(request_data)

        # Send request to FastAPI server
        response = requests.post(f"{self.api_url}/classify/", json=request_data)
        if response.status_code == 200:
            st.write(
                f'The classification result is: {response.json()["classification"]}'
            )
        else:
            st.error("Error in classification")

    def display_result(self):
        pass


if __name__ == "__main__":
    app = UserInterface("http://localhost:8000")
