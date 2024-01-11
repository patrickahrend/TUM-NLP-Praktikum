# Functional Imports
import logging
import pickle
from pathlib import Path

import numpy as np

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


class ModelAPI:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.embedding_processor = EmbeddingProcessor()

    def load_model(self, model_name, embedding_name):
        model_file = self.model_path / embedding_name / f"{model_name}.pkl"
        if model_file.exists():
            with model_file.open("rb") as file:
                model = pickle.load(file)
                logger.info(f"Loaded model: {model_name}")
                return model
        else:
            logger.error(f"Model file {model_name}.pkl not found.")
            raise FileNotFoundError(f"Model file {model_name}.pkl not found.")

    def classify(self, model, process_description, text_passage, embedding_type):
        embedding_processor = EmbeddingProcessor()

        embeddings = embedding_processor.embed_new_text(
            process_description, text_passage, embedding_type
        )
        embedding_vector_1 = np.squeeze(embeddings[0])
        embedding_vector_2 = np.squeeze(embeddings[1])
        logger.info("Embeddings created", embeddings)
        logger.info(embeddings[0].shape)
        logger.info(embeddings[1].shape)
        logger.info(len(embeddings))
        combined_embedding_vector = np.concatenate(
            (embedding_vector_1, embedding_vector_2)
        ).reshape(1, -1)

        result = model.predict(combined_embedding_vector)

        return result


# Instantiate the FastAPI app
app = FastAPI()

project_dir = Path(__file__).resolve().parents[2]
model_path = project_dir / "models"
model_api = ModelAPI(model_path)


@app.post("/classify/")
async def classify(request: ClassificationRequest):
    logger.info(f"Received request: {request}")
    try:
        model = model_api.load_model(request.model_name, request.embedding_name)
        print(model.__repr__() + " loaded")
        result = model_api.classify(
            model,
            request.process_description,
            request.text_passage,
            request.embedding_name,
        )

        print(result + " classified")
        return {"classification": result}
    except Exception as e:
        logger.exception("Error during classification")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
