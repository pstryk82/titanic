from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from ClassifierLibrary.AbstractClassifier import AbstractClassifier

class DecisionTreeClassifier(AbstractClassifier):
    def __init__(self, criterion='entropy'):
        self.engine = SklearnDecisionTreeClassifier(criterion=criterion)

    def fit(self, X, y):
        self.engine.fit(X, y)

    def predict(self, X):
        y_pred = self.engine.predict(X)
        return y_pred

    def classify(self, X, y, cv=10, verbose=False):
        self.fit(X, y)
        y_pred = self.predict(X)
        accuracies, std = self.crossValidate(X, y_pred, cv)
        if verbose:
            print('Decision Tree mean accuracy: ', accuracies)
            print('Decision Tree standard deviation: ', std)

        return accuracies, std

    def crossValidate(self, X, y, cv=10):
        accuracies = cross_val_score(estimator=self.engine, X=X, y=y, cv=cv)
        return accuracies.mean(), accuracies.std()
