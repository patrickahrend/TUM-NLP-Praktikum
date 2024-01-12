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
        self.result_path = None
        self.test_data = None
        self.processes = None

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

            test_set = st.toggle("Use Test Set", value=False)

            if test_set:
                self.load_test_data()
                self.display_all_test_data_points()

            else:
                self.text_input = st.text_area("Enter the text passage here:")

                self.process_description = st.text_area(
                    "Enter the process description here:"
                )

            # if st.button("Classify"):
            #     self.classify_text(
            #         self.text_input,
            #         self.process_description,
            #         self.selected_model,
            #         self.selected_embedding,
            #     )

        with tab2:
            ### here show hyperparamter tuning results
            st.write("Showing the results of the classification here")
            pass

        ## here show model evaluation results
        with tab3:
            self.display_results()

    def load_test_data(self):
        base_path = Path(__file__).resolve().parents[1]
        test_data_path = base_path / "data/evaluation/gold_standard.csv"
        self.test_data = pd.read_csv(test_data_path)
        self.processes = self.test_data["Process"].unique().tolist()

    def display_all_test_data_points(self):
        selected_process = st.selectbox("Select a Process", self.processes)
        filtered_data = self.test_data[self.test_data["Process"] == selected_process]

        if not filtered_data.empty:
            process_description = filtered_data["Process_description"].iloc[0]
            st.text_area(
                "Process Description",
                value=process_description,
                height=300,
                disabled=True,
            )

            # Concatenate all the texts related to the process into one document
            concatenated_texts = "\n\n".join(filtered_data["Text"])
            correct_labels = filtered_data["Label"].tolist()
            st.text_area(
                "All Texts", value=concatenated_texts, height=600, disabled=True
            )

            if st.button("Classify"):
                self.display_classification_result(concatenated_texts, correct_labels)
        else:
            st.write("No data available for the selected process.")

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
            return response.json()["classification"]
        else:
            st.error("Error in classification")
            return None

    def display_classification_result(self, concatenated_texts, correct_labels):
        classification_result = self.classify_text(
            concatenated_texts,
            self.process_description,
            self.selected_model,
            self.selected_embedding,
        )

        if classification_result:
            correct = classification_result == correct_labels
            color = "green" if correct else "red"
            # red if incorrect, green if correct
            st.markdown(
                f'<span style="color:{color};text-decoration:underline;">{concatenated_texts}</span>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    app = UserInterface("http://localhost:8000")
