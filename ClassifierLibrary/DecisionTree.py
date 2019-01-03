from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from ClassifierLibrary.AbstractClassifier import AbstractClassifier

class DecisionTreeClassifier(AbstractClassifier):
    def __init__(self, criterion='entropy'):
        self.engine = SklearnDecisionTreeClassifier(criterion=criterion)


    def printSummary(self, accuracies, std):
        print('Decision Tree mean accuracy: ', accuracies)
        print('Decision Tree standard deviation: ', std)
