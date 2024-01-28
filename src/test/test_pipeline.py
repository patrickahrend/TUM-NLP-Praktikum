import argparse
import unittest
from unittest.mock import patch
import os
from src.data.make_dataset import main as make_dataset_main
from src.features.make_embeddings import main as make_embeddings_main
from src.models import make_models
from pathlib import Path

import logging
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.base_path = Path(__file__).resolve().parents[2]

    def test_make_data(self):
        # Run the pipeline
        make_dataset_main(self.base_path / "data/raw", self.base_path / "data/processed")

        # Check that the gold standard subset CSV file was created
        gold_standard_path = self.base_path / "data/evaluation/gold_standard.csv"
        self.assertTrue(os.path.exists(gold_standard_path), f"File not found: {gold_standard_path}")

        # Check that the preprocessed training data CSV file was created
        training_data_path = self.base_path / "data/processed/training_data_preprocessed.csv"
        self.assertTrue(os.path.exists(training_data_path), f"File not found: {training_data_path}")

        # Check that the preprocessed gold standard subset CSV file was created
        gold_standard_preprocessed_path = self.base_path / "data/evaluation/gold_standard_preprocessed.csv"
        self.assertTrue(os.path.exists(gold_standard_preprocessed_path),
                        f"File not found: {gold_standard_preprocessed_path}")

    def test_make_embeddings(self):
        # Run the embeddings pipeline
        make_embeddings_main()

        # Define the expected embedding types
        embedding_types = ["word2vec", "glove", "fasttext", "tfidf", "bert", "gpt"]

        # Check for the presence of the embedding files for each type, as there should be 6 files for embedding type
        for emb_type in embedding_types:
            for data_type in ["train", "test"]:
                for context in ["combined", "separate", "legal_text"]:
                    file_path = self.base_path / f"data/processed/embeddings/{emb_type}_{data_type}_{context}.pkl"
                    self.assertTrue(os.path.exists(file_path), f"Embedding file not found: {file_path}")

    @staticmethod
    def call_main_with_args(self, dataset_variant, is_tuned):
        with patch('argparse._sys.argv', ['make_models.py',
                                          '--dataset_variant', dataset_variant,
                                          '--is_tuned' if is_tuned else '--no-tune']):
            make_models.main()

    def test_make_models(self):
        # Call the main function with the desired sets of arguments
        self.call_main_with_args('combined', False)
        self.call_main_with_args('combined', True)
        self.call_main_with_args('separate', False)
        self.call_main_with_args('separate', True)
        variants = ['combined', 'separate']
        tunings = ['tuned', 'no_tuning']
        embedding_types = ['word2vec', 'glove', 'fasttext', 'tfidf', 'bert', 'gpt']
        model_types = ["Logistic_Regression","RandomForestClassifier","SVC","DecisionTreeClassifier","BernoulliNB",
                       "GaussianNB", "Perceptron","SGDClassifier","GradientBoostingClassifier"]

        # Iterate over each combination of variant, tuning, embedding, and model type
        for variant in variants:
            for tuning in tunings:
                for emb_type in embedding_types:
                    for model_type in model_types:
                        model_file = self.base_path / f"models/trained_models/{variant}/{tuning}/{emb_type}/{model_type}.pkl"
                        self.assertTrue(os.path.exists(model_file), f"Model file not found: {model_file}")


if __name__ == '__main__':
    unittest.main()
