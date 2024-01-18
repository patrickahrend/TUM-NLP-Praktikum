# Functional Imports
import logging
import pickle
from pathlib import Path

# Server import
import uvicorn

# API imports
from fastapi import FastAPI, HTTPException
from features.build_word_embedding import EmbeddingProcessor
from pydantic import BaseModel

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request Model
class ClassificationRequest(BaseModel):
    text_passage: str
    process_description: str
    model_name: str
    embedding_name: str
    dataset_type: str
    is_tuned: bool
    is_pca: bool


class ModelAPI:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.embedding_processor = EmbeddingProcessor()

    def load_model(
        self,
        model_name,
        embedding_name,
        is_tuned,
        is_pca,
        dataset_type,
    ):
        dataset_dir = "pca" if is_pca else "normal"
        tuned_dir = "tuned" if is_tuned else "no_tuning"
        embedding_mapping = {
            "word2vec": "w2v",
            "glove": "glove",
            "bert": "bert",
            "gpt": "gpt",
            "fasttext": "ft",
            "tfidf": "tfidf",
        }
        model_file = (
            self.model_path
            / dataset_dir
            / dataset_type
            / tuned_dir
            / embedding_mapping[embedding_name]
            / f"{model_name}.pkl"
        )
        if model_file.exists():
            with model_file.open("rb") as file:
                model = pickle.load(file)
                logger.info(f"Loaded model: {model_name}")
                return model
        else:
            logger.error(f"Model file {model_name}.pkl not found.")
            raise FileNotFoundError(f"Model file {model_name}.pkl not found.")

    def classify(
        self,
        model,
        process_description,
        text_passage,
        embedding_type,
        dataset_type,
    ):
        legal_passages = text_passage.split("\n\n")
        classification_results = []
        for passage in legal_passages:
            if dataset_type == "separate":
                combined_embedding_vector = self.embedding_processor.embed_new_text(
                    process_description, passage, embedding_type, dataset_type
                )
            else:  # combined
                combined_text = process_description + " " + passage
                combined_embedding_vector = self.embedding_processor.embed_new_text(
                    combined_text, "", embedding_type, dataset_type
                )

            prediction = model.predict(combined_embedding_vector)
            ## conversion to int is necessary for the frontend
            classification_results.append(int(prediction[0]))

        return classification_results


# Instantiate the FastAPI app
app = FastAPI()

project_dir = Path(__file__).resolve().parents[2]
model_path = project_dir / "models"
model_api = ModelAPI(model_path)
model_api.embedding_processor = EmbeddingProcessor()


@app.post("/classify/")
async def classify(request: ClassificationRequest):
    logger.info(f"Received request: {request}")
    try:
        model = model_api.load_model(
            request.model_name,
            request.embedding_name,
            request.is_tuned,
            request.is_pca,
            request.dataset_type,
        )
        logger.info(model.__repr__() + " loaded")
        classification_results = model_api.classify(
            model,
            request.process_description,
            request.text_passage,
            request.embedding_name,
            request.dataset_type,
        )

        logger.info("Classification completed.")
        classification_results = [int(result) for result in classification_results]

        return {"classification": classification_results}

    except Exception as e:
        logger.exception("Error during classification")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
