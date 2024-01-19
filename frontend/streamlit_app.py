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
        self.dataset_type = None
        self.is_pca = None
        self.is_tuned = None

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
                self.dataset_type = st.radio(
                    "Select dataset type", ("separate", "combined")
                )
                self.is_tuned = st.toggle("Hyperparameter Tuned", value=False)
                self.is_pca = st.toggle("PCA Applied", value=False)
                self.load_test_data()
                self.display_all_test_data_points()

            else:
                self.text_input = st.text_area("Enter the text passage here:")

                self.process_description = st.text_area(
                    "Enter the process description here:"
                )

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
        process_options = ["All Processes"] + self.processes
        selected_process = st.selectbox("Select a Process", process_options)

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
            correct_labels = filtered_data["Label"].tolist()
            st.text_area(
                "All Texts", value=concatenated_texts, height=600, disabled=True
            )
            self.text_input = concatenated_texts

            if st.button("Classify"):
                classification_result = self.classify_text(
                    self.text_input,
                    self.process_description,
                    self.selected_model,
                    self.selected_embedding,
                    self.dataset_type,
                    self.is_tuned,
                    self.is_pca,
                )

                for original_text, predicted_label, correct_label in zip(
                    filtered_data["Text"], classification_result, correct_labels
                ):
                    is_correct = predicted_label == correct_label
                    color = "green" if is_correct else "red"

                    st.markdown(
                        f"""
                        <style>
                        .reportview-container .markdown-text-container {{
                            font-family: sans-serif;
                        }}
                        .text_area {{
                            border: 1px solid #ced4da;
                            padding: 10px 15px;
                            font-size: 16px;
                            line-height: 1.5;
                            color: #495057;
                            background-color: #fff;
                            border-radius: 4px;
                            resize: none;
                            box-shadow: inset 0 1px 2px rgba(0,0,0,.075);
                            margin-bottom: 10px;
                            overflow: auto;
                        }}
                        .text_area:focus {{
                            border-color: #80bdff;
                            outline: 0;
                            box-shadow: inset 0 1px 2px rgba(0,0,0,.075), 0 0 5px rgba(128,189,255,.5);
                        }}
                        </style>
                        <textarea readonly class="text_area" style="width: 100%; min-height: 100px; overflow: auto;">{original_text}</textarea>
                        <div style="color: {color}; font-size: 0.9em; margin-top: 5px;">
                            <strong>Model predicted:</strong> {predicted_label}, <strong>Ground truth:</strong> {correct_label}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.write("No data available for the selected process.")

    def display_results(self):
        base_path = Path(__file__).parents[1]
        model_results_dir = base_path / "references" / "model results"
        feature_importance_dir = base_path / "references" / "feature importance"

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

    def classify_text(
        self,
        text_input,
        process_description,
        selected_model,
        selected_embedding,
        dataset_type,
        is_tuned,
        is_pca,
    ):
        request_data = {
            "text_passage": text_input,
            "process_description": process_description,
            "model_name": selected_model,
            "embedding_name": selected_embedding,
            "dataset_type": dataset_type,
            "is_tuned": is_tuned,
            "is_pca": is_pca,
        }
        print(request_data)

        # Send request to FastAPI server
        response = requests.post(f"{self.api_url}/classify/", json=request_data)
        if response.status_code == 200:
            return response.json()["classification"]
        else:
            st.error("Error in classification")
            return None


if __name__ == "__main__":
    app = UserInterface("http://localhost:8000")
