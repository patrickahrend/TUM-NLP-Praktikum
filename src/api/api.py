# Functional Imports
import logging
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np

# Server import
import uvicorn

# API imports
from fastapi import FastAPI, HTTPException

# Custom imports
from src.features.build_word_embeddings import EmbeddingProcessor
from pydantic import BaseModel, ValidationError

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationRequest(BaseModel):
    """
    Pydantic model for the classification request data.
    """

    text_passage: Optional[str] = None
    process_description: Optional[str] = None
    model_name: str
    embedding_name: str
    dataset_type: str
    is_tuned: bool
    embeddings: Optional[List[float]] = None


class ModelAPI:
    """
    API for loading models and classifying text.
    """

    def __init__(self, model_path: Path):
        """
        Initializes the ModelAPI object.

        Parameters:
        model_path (Path): The path to the directory containing the models.
        """
        self.model_path = model_path
        self.embedding_processor = EmbeddingProcessor()

    def load_model(
        self,
        model_name: str,
        embedding_name: str,
        is_tuned: bool,
        dataset_type: str,
    ) -> object:
        """
        Loads a model from a pickle file.

        Parameters:
        model_name (str): The name of the model.
        embedding_name (str): The name of the embedding.
        is_tuned (bool): Whether the model is tuned.
        dataset_type (str): The type of the dataset.

        Returns:
        The loaded model.
        """
        tuned_dir = "tuned" if is_tuned else "no_tuning"
        model_file = (
            self.model_path
            / "trained_models"
            / dataset_type
            / tuned_dir
            / embedding_name
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
        model: object,
        process_description: str,
        text_passage: str,
        embedding_type: str,
        dataset_type: str,
    ) -> list[int]:
        """
        Classifies a text passage using the given model and embedding.

        Parameters:
        model: The model to use for classification.
        process_description (str): The process description.
        text_passage (str): The text passage to classify.
        embedding_type (str): The type of the embedding.
        dataset_type (str): The type of the dataset.

        Returns:
        list: The classification results.
        """
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
            # conversion to int is necessary for the frontend
            classification_results.append(int(prediction[0]))

        return classification_results


# Instantiate the FastAPI app
app = FastAPI()

project_dir = Path(__file__).resolve().parents[2]
model_path = project_dir / "models"
model_api = ModelAPI(model_path)
# model_api.embedding_processor = EmbeddingProcessor()


@app.post("/classify/")
async def classify(request: ClassificationRequest) -> dict:
    """
    Endpoint for classifying text.

    Parameters:
    request (ClassificationRequest): The classification request data.

    Returns:
    dict: The classification results.
    """
    logger.info(f"Received request with body: {request}")
    try:
        request_object = ClassificationRequest.parse_obj(request)

        model = model_api.load_model(
            request.model_name,
            request.embedding_name,
            request.is_tuned,
            request.dataset_type,
        )
        logger.info(model.__repr__() + " loaded")
        if request.embeddings:
            logger.info("Classification started.")
            # reshape to 2d array
            embeddings_2d = np.array(request.embeddings).reshape(1, -1)
            prediction = model.predict(embeddings_2d)
            classification_results = int(prediction[0])

        # for new text
        else:
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
    except ValidationError as e:
        print(e.json())
    except Exception as e:
        logger.exception("Error during classification")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """
    The main function that starts the FastAPI server.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
