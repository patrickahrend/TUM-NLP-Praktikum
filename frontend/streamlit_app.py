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

    def initialize_app(self):
        st.set_page_config(layout="wide")

        st.title("Legal Text Classifier")

        self.model_options = [
            "Logistic_Regression",
            "RandomForestClassifier",
            "SVC",
            "DecisionTreeClassifier",
            "BernoulliNB",
            "GaussianNB",
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

        self.selected_embedding = st.selectbox("Select an embedding", embedding_options)

        self.text_input = st.text_area("Enter the text passage here:")

        self.process_description = st.text_area("Enter the process description here:")

        if st.button("Classify"):
            self.classify_text()

    @st.cache
    def classify_text(self):
        request_data = {
            "text_passage": self.text_input,
            "process_description": self.process_description,
            "model_name": self.selected_model,
            "embedding_name": self.selected_embedding,
        }

        # Send request to FastAPI server
        response = requests.post(f"{self.api_url}/classify/", json=request_data)
        if response.status_code == 200:
            st.write(
                f'The classification result is: {response.json()["classification"]}'
            )
        else:
            st.error("Error in classification")

    def show_result(self):
        pass


if __name__ == "__main__":
    app = StreamlitApp("http://localhost:8000")
