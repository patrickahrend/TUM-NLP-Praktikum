import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from transformers import BertModel

from model_base import ModelBase


class LogisticRegressionModel(ModelBase):
    def __init__(self):
        super().__init__(
            "Logistic_Regression",
            LogisticRegression(),
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
            RandomForestClassifier(),
        )


class GradientBoostingModel(ModelBase):
    def __init__(self):
        super().__init__(
            "GradientBoostingClassifier",
            GradientBoostingClassifier(),
        )


class DecisionTreeModel(ModelBase):
    def __init__(self):
        super().__init__(
            "DecisionTreeClassifier",
            DecisionTreeClassifier(),
        )


class SVCModel(ModelBase):
    def __init__(self):
        super().__init__(
            "SVC",
            SVC(),
        )


class PerceptronModel(ModelBase):
    def __init__(self):
        super().__init__(
            "Perceptron",
            Perceptron(),
        )


class SGDClassifierModel(ModelBase):
    def __init__(self):
        super().__init__(
            "SGDClassifier",
            SGDClassifier(),
        )



#################################################################################
# Only used in the notebook, but wanted to include here as well for comparison. #
#################################################################################
class BERTForClassification(nn.Module):
    def __init__(self):
        super(BERTForClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        return self.fc(pooled_output)


class RnnTextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0,device= 'cpu'):
        super(RnnTextClassifier, self).__init__()

        # model params
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # reshape input
        x = x.unsqueeze(1).to(self.device)

        # initialize hidden state
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # get RNN output
        out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])

        return out
