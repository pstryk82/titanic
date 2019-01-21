from sklearn.linear_model import LogisticRegression
from ClassifierLibrary.AbstractClassifier import AbstractClassifier

class LogisticRegressionClassifier(AbstractClassifier):
    def __init__(self):
        self.engine = LogisticRegression()


    def printSumary(self, accuracues, std):
        print('Logistic Regression mean accuracy: ', accuracues)
        print('Logistic Regression mean standard deviation: ', std)