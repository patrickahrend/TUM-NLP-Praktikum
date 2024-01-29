import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from transformers import BertModel

from src.models.model_base import ModelBase


class LogisticRegressionModel(ModelBase):
    """
    Logistic Regression model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__(
            "Logistic_Regression",
            LogisticRegression(),
        )


class GaussianNBModel(ModelBase):
    """
    Gaussian Naive Bayes model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__("GaussianNB", GaussianNB())


class BernoulliNBModel(ModelBase):
    """
    Bernoulli Naive Bayes model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__("BernoulliNB", BernoulliNB())


class RandomForestModel(ModelBase):
    """
    Random Forest model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__(
            "RandomForestClassifier",
            RandomForestClassifier(),
        )


class GradientBoostingModel(ModelBase):
    """
    Gradient Boosting model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__(
            "GradientBoostingClassifier",
            GradientBoostingClassifier(),
        )


class DecisionTreeModel(ModelBase):
    """
    Decision Tree model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__(
            "DecisionTreeClassifier",
            DecisionTreeClassifier(),
        )


class SVCModel(ModelBase):
    """
    Support Vector Classifier model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__(
            "SVC",
            SVC(),
        )


class PerceptronModel(ModelBase):
    """
    Perceptron model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__(
            "Perceptron",
            Perceptron(),
        )


class SGDClassifierModel(ModelBase):
    """
    Stochastic Gradient Descent Classifier model class which inherits from the ModelBase class.
    """

    def __init__(self):
        super().__init__(
            "SGDClassifier",
            SGDClassifier(),
        )


#################################################################################
# Only used in the notebook, but wanted to include here as well for comparison. #
#################################################################################
class BERTForClassification(nn.Module):
    """
    BERT model for classification tasks. Inherits from the PyTorch nn.Module class.
    """

    def __init__(self):
        super(BERTForClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the BERT model.

        Parameters
        ----------
            input_ids : Tensor
                The input data (token ids).
            attention_mask : Tensor
                The attention mask for the input data.

        Returns
        -------
            Tensor
                The output from the linear layer.
        """
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output
        return self.fc(pooled_output)


class RnnTextClassifier(nn.Module):
    """
    RNN model for text classification tasks. Inherits from the PyTorch nn.Module class.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout=0,
        device="cpu",
    ):
        super(RnnTextClassifier, self).__init__()

        # model params
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # layers
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RNN model.

        Parameters
        ----------
            x : Tensor
                The input data.

        Returns
        -------
            Tensor
                The output from the linear layer.
        """
        # reshape input
        x = x.unsqueeze(1).to(self.device)

        # initialize hidden state
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            self.device
        )

        # get RNN output
        out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])

        return out
