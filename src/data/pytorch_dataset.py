import torch
from torch.utils.data import Dataset

#
# class ProcessLegalTextDataset(Dataset):
#     def __init__(
#         self, process_descriptions, legal_texts, labels, tokenizer, max_length=512
#     ):
#         self.process_descriptions = process_descriptions
#         self.legal_texts = legal_texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __getitem__(self, idx):
#         encoding = self.tokenizer.encode_plus(
#             self.process_descriptions[idx],
#             self.legal_texts[idx],
#             add_special_tokens=True,
#             max_length=self.max_length,
#             return_token_type_ids=False,
#             padding="max_length",
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors="pt",
#         )
#
#         item = {
#             key: val.squeeze(0) for key, val in encoding.items()
#         }  # remove batch dimension
#         item["labels"] = torch.tensor(self.labels[idx])
#         return item
#
#     def __len__(self):
#         return len(self.labels)


class ProcessLegalTextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        # embeddings[idx] should be a single embedding vector representing your text
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return {
            "embedding": torch.tensor(embedding, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float),
        }

    def __len__(self):
        return len(self.labels)
