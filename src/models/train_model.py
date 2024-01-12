from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.data.pytorch_dataset import ProcessLegalTextDataset
from src.models.build_models import load_embeddings
from src.models.model_classes import BERTForClassification, RnnTextClassifier

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def bert_tokenize_and_embed(bert_model, process_descriptions, legal_texts):
    embeddings = []
    for description, text in zip(process_descriptions, legal_texts):
        encoded_input = bert_tokenizer.encode_plus(
            description,
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            model_output = bert_model(**encoded_input)
        embeddings.append(model_output.pooler_output.squeeze().numpy())
    return embeddings


def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_predictions)


def train_rnn(model, data_loader, criterion, optimizer):
    model.train()

    for epoch in range(100):
        for batch in data_loader:
            embedding, label = batch["embedding"], batch["label"]
            optimizer.zero_grad()
            y_pred = model(embedding)
            loss = criterion(y_pred, label)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{100}, Loss: {loss.item()}")
    return model


def main():
    project_dir = Path(__file__).resolve().parents[2]
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    trainings_data = pd.read_csv(
        project_dir / "data/processed/training_data_preprocessed.csv"
    )
    test_data = pd.read_csv(
        project_dir / "data/evaluation/gold_standard_preprocessed.csv"
    )

    ## BERT Classification

    model = BERTForClassification(num_classes=2)
    train_embeddings = bert_tokenize_and_embed(
        model, trainings_data["Process_description"], trainings_data["Text"]
    )
    train_dataset = ProcessLegalTextDataset(train_embeddings, trainings_data["Label"])

    test_embeddings = bert_tokenize_and_embed(
        model, test_data["Process_description"], test_data["Text"]
    )
    test_dataset = ProcessLegalTextDataset(test_embeddings, test_data["Label"])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    loss_function = CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy}")

    ## RNN Classification
    embedding_path = project_dir / "data/processed/embeddings"
    embeddings = load_embeddings(embedding_path)
    dataset = ProcessLegalTextDataset(embeddings, trainings_data["Label"])
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for embedding in embeddings.values():
        rnn_classifier = RnnTextClassifier(
            input_size=embedding.shape[1],
            output_size=1,
            hidden_size=256,
            num_layers=2,
            dropout=0.5,
        )

        # Define loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(rnn_classifier.parameters(), lr=0.001)

        rnn_classifier = train_rnn(
            rnn_classifier,
            data_loader,
            criterion,
            optimizer,
        )

        # TODO evaluate of rnn


if __name__ == "__main__":
    main()
