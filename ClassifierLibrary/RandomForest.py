from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from ClassifierLibrary.AbstractClassifier import AbstractClassifier

class RandomForestClassifier(AbstractClassifier):
    def __init__(self, criterion='entropy'):
        self.engine = SklearnRandomForestClassifier(criterion=criterion, n_estimators=10)


    def printSummary(self, accuracies, std):
        print('Random Forest mean accuracy: ', accuracies)
        print('Random Forest standard deviation: ', std)
