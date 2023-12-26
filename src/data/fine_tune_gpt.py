# fine_tune_gpt.py
import os
from pathlib import Path

from dotenv import load_dotenv


def fine_tune_model(training_file, validation_file, model_name="gpt-3.5-turbo-1106"):
    """
    Fine-tune the GPT model using the provided training and validation datasets.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("Set your OPENAI_API_KEY environment variable.")

    openai.api_key = api_key

    # Upload the training file
    training_response = openai.File.create(
        file=open(training_file, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response["id"]

    # Upload the validation file
    validation_response = openai.File.create(
        file=open(validation_file, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response["id"]

    # Start the fine-tuning job
    fine_tune_response = openai.FineTune.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=model_name,
    )
    fine_tune_job_id = fine_tune_response["id"]

    print(f"Started fine-tuning job {fine_tune_job_id}")


def get_complition():
    # max_tokens=1 and logprobs=2
    pass


if __name__ == "__main__":
    # Load environment variables, API keys, etc.
    # Configure logging if necessary

    project_dir = Path(__file__).resolve().parents[2]
    dotenv_path = project_dir / ".env"
    load_dotenv(dotenv_path)

    training_file = (
        project_dir / "data" / "processed" / "train_text_classification.jsonl"
    )
    validation_file = (
        project_dir / "data" / "processed" / "val_text_classification.jsonl"
    )

    # Call the fine-tuning function
    fine_tune_model(training_file, validation_file)
