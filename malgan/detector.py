# -*- coding: utf-8 -*-

from enum import Enum
from typing import Union

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.qda import QDA
# from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

import torch

TorchOrNumpy = Union[np.ndarray, torch.Tensor]


# noinspection PyPep8Naming
class BlackBoxDetector:
    r"""
    Black box detector that intends to mimic an antivirus/anti-Malware program that detects whether
    a specific program is either malware or benign.
    """
    class Type(Enum):
        r""" Learner algorithm to be used by the black-box detector """
        DecisionTree = DecisionTreeClassifier()
        LogisticRegression = LogisticRegression(solver='lbfgs', max_iter=int(1e6))
        MultiLayerPerceptron = MLPClassifier()
        RandomForest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        KNeighbors = KNeighborsClassifier(n_jobs=-1)
        SVM = SVC(gamma="auto")
        RadiusNeighbors = RadiusNeighborsClassifier(n_jobs=-1)
        ExtraTree = ExtraTreeClassifier()
        NaiveBayes = GaussianNB()
        AdaBoostLR = AdaBoostClassifier(base_estimator=LogisticRegression)
        Bernoulli = BernoulliNB()
        GradientBoosting = GradientBoostingClassifier()
        # estimators = [('dc', DecisionTree),
        #     ('lr', LogisticRegression),
        #     ('knn', KNeighbors)]#,
        #     #('nbc', NaiveBayes),
        #     #('bnc', Bernoulli)]
        estimators = [
            ('adblr', AdaBoostLR),
            ('gbc', GradientBoosting),
            ('rf', RandomForest),
            
        ]
        Stackingx3Logistic = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression, n_jobs=4, cv=10)
        StackingExtraTree = StackingClassifier(estimators=estimators, final_estimator=ExtraTree, n_jobs=-1, cv = 10)
        StackingRandomForest = StackingClassifier(estimators=estimators, final_estimator=RandomForest, n_jobs=-1, cv = 10)
        Stackingx3DecisionTree = StackingClassifier(estimators=estimators, final_estimator=DecisionTree, n_jobs=4, cv = 10)
        # StackingAdaBoost = StackingClassifier(estimators=estimators, final_estimator=AdaBoost, n_jobs=-1, cv = 10)
        StackingNaiveBayes = StackingClassifier(estimators=estimators, final_estimator=NaiveBayes, n_jobs=-1, cv = 10)
        
        Stackingx3KNeighbors = StackingClassifier(estimators=estimators, final_estimator=KNeighbors, n_jobs=4, cv = 10)
        StackingMultiLayer = StackingClassifier(estimators=estimators, final_estimator=MultiLayerPerceptron, n_jobs=-1, cv = 10)
        StackingSGDC = StackingClassifier(estimators=estimators, final_estimator=SGDClassifier(n_jobs=-1), n_jobs=-1, cv = 10)
        StackingLDA = StackingClassifier(estimators=estimators, final_estimator=LinearDiscriminantAnalysis(), n_jobs=-1, cv = 10)
        StackingBernoulli = StackingClassifier(estimators=estimators, final_estimator=Bernoulli, n_jobs=-1, cv = 10)
        StackingGradient = StackingClassifier(estimators=estimators, final_estimator=GradientBoosting, n_jobs=-1, cv = 10)
        Votingx5 = VotingClassifier(estimators=estimators, n_jobs=-1, voting='soft')
        BaggingDecision = BaggingClassifier(n_jobs=-1)

        @staticmethod
        def names():
            r""" Builds the list of all enum names """
            return [c.name for c in BlackBoxDetector.Type]

        @staticmethod
        def get_from_name(name):
            r"""
            Gets the enum item from the specified name

            :param name: Name of the enum object
            :return: Enum item associated with the specified name
            """
            for c in BlackBoxDetector.Type:
                if c.name == name:
                    return c
            raise ValueError("Unknown enum \"%s\" for class \"%s\"", name, __class__.name)

    def __init__(self, learner_type: 'BlackBoxDetector.Type'):
        self.type = learner_type
        # noinspection PyCallingNonCallable
        self._model = sklearn.clone(self.type.value)
        self.training = True

    def fit(self, X: TorchOrNumpy, y: TorchOrNumpy):
        r"""
        Fits the learner.  Supports NumPy and PyTorch arrays as input.  Returns a torch tensor
        as output.

        :param X: Examples upon which to train
        :param y: Labels for the examples
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        self._model.fit(X, y)
        self.training = False

    def predict(self, X: TorchOrNumpy) -> torch.tensor:
        r"""
        Predict the labels for \p X

        :param X: Set of examples for which label probabilities should be predicted
        :return: Predicted value for \p X
        """
        if self.training:
            raise ValueError("Detector does not appear to be trained but trying to predict")
        if torch.cuda.is_available():
            X = X.cpu()
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        y = torch.from_numpy(self._model.predict(X)).float()
        return y.cuda() if torch.cuda.is_available() else y
