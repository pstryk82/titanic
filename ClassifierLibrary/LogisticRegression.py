from sklearn.linear_model import LogisticRegression
from ClassifierLibrary.AbstractClassifier import AbstractClassifier

class LogisticRegressionClassifier(AbstractClassifier):
    def __init__(self):
        super().__init__()
        self.engine = LogisticRegression(solver='lbfgs')

    def printSummary(self, score, std, scoringFunction):
        print('Logistic Regression mean ', scoringFunction, ': ', score)
        print('Logistic Regression standard deviation: ', std)