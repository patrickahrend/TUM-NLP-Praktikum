from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_base import ModelBase


class LogisticRegressionModel(ModelBase):
    def __init__(self):
        super().__init__("Logistic_Regression", LogisticRegression())


class MultinomialNBModel(ModelBase):
    def __init__(self):
        super().__init__("MultinomialNB", MultinomialNB())


class GaussianNBModel(ModelBase):
    def __init__(self):
        super().__init__("GaussianNB", GaussianNB())


class BernoulliNBModel(ModelBase):
    def __init__(self):
        super().__init__("BernoulliNB", BernoulliNB())


class RandomForestModel(ModelBase):
    def __init__(self):
        super().__init__("RandomForestClassifier", RandomForestClassifier())


class GradientBoostingModel(ModelBase):
    def __init__(self):
        super().__init__("GradientBoostingClassifier", GradientBoostingClassifier())


class DecisionTreeModel(ModelBase):
    def __init__(self):
        super().__init__("DecisionTreeClassifier", DecisionTreeClassifier())


class SVCModel(ModelBase):
    def __init__(self):
        super().__init__("SVC", SVC())


class PerceptronModel(ModelBase):
    def __init__(self):
        super().__init__("Perceptron", Perceptron())


class SGDClassifierModel(ModelBase):
    def __init__(self):
        super().__init__("SGDClassifier", SGDClassifier())
