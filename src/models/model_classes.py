import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from transformers import BertModel

from model_base import ModelBase

## class weights to not overfit on the majority class, as the data is imbalanced (9:1), we do the inverse here
class_weights = {0: 1, 1: 9}


class LogisticRegressionModel(ModelBase):
    def __init__(self):
        super().__init__(
            "Logistic_Regression",
            LogisticRegression(max_iter=1000, class_weight=class_weights),
        )


class GaussianNBModel(ModelBase):
    def __init__(self):
        super().__init__("GaussianNB", GaussianNB())


class BernoulliNBModel(ModelBase):
    def __init__(self):
        super().__init__("BernoulliNB", BernoulliNB())


class RandomForestModel(ModelBase):
    def __init__(self):
        super().__init__(
            "RandomForestClassifier",
            RandomForestClassifier(class_weight=class_weights),
        )


class GradientBoostingModel(ModelBase):
    def __init__(self):
        super().__init__("GradientBoostingClassifier", GradientBoostingClassifier())


class DecisionTreeModel(ModelBase):
    def __init__(self):
        super().__init__("DecisionTreeClassifier", DecisionTreeClassifier())


class SVCModel(ModelBase):
    def __init__(self):
        super().__init__("SVC", SVC(class_weight=class_weights))


class PerceptronModel(ModelBase):
    def __init__(self):
        super().__init__("Perceptron", Perceptron())


class SGDClassifierModel(ModelBase):
    def __init__(self):
        super().__init__("SGDClassifier", SGDClassifier())


class KNNClassifierModel(ModelBase):
    def __init__(self):
        super().__init__("KNeighborsClassifier", KNeighborsClassifier())


class BERTForClassification(nn.Module):
    def __init__(self, num_classes):
        super(BERTForClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        return self.fc(pooled_output)


class RnnTextClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0):
        super(RnnTextClassifier, self).__init__()

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)

        out, _ = self.rnn(x, h0)

        out = self.dropout(out)
        out = self.fc(out[:, -1, :])

        return out
