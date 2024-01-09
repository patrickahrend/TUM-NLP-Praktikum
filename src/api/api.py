import logging
import pickle
from pathlib import Path

import uvicorn
# API imports
from fastapi import FastAPI, HTTPException
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

    def classify(self, model, process_description, text_passage):
        return model.predict([process_description, text_passage])[0]


# Instantiate the FastAPI app
app = FastAPI()

project_dir = Path(__file__).resolve().parents[2]
model_path = project_dir / "models"
model_api = ModelAPI(model_path)


@app.post("/classify/")
async def classify(request: ClassificationRequest):
    try:
        model = model_api.load_model(request.model_name, request.embedding_name)
        result = model_api.classify(
            model, request.process_description, request.text_passage
        )
        return {"classification": result}
    except Exception as e:
        logger.exception("Error during classification")
        raise HTTPException(status_code=500, detail=str(e))


# Run with Uvicorn Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
