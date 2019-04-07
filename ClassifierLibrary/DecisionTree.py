from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from ClassifierLibrary.AbstractClassifier import AbstractClassifier

class DecisionTreeClassifier(AbstractClassifier):
    def __init__(self, criterion='entropy'):
        super().__init__()
        self.engine = SklearnDecisionTreeClassifier(criterion=criterion)


    def printSummary(self, score, std, scoringFunction):
        print('Decision Tree mean ', scoringFunction, ': ', score)
        print('Decision Tree standard deviation: ', std)
