from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from ClassifierLibrary.AbstractClassifier import AbstractClassifier

class RandomForestClassifier(AbstractClassifier):
    def __init__(self, criterion='entropy'):
        super().__init__()
        self.engine = SklearnRandomForestClassifier(criterion=criterion, n_estimators=10)


    def printSummary(self, score, std, scoringFunction):
        print('Random Forest mean ', scoringFunction, ': ', score)
        print('Random Forest standard deviation: ', std)
