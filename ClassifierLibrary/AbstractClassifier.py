from sklearn.model_selection import cross_val_score

class AbstractClassifier:

    def fit(self, X, y):
        self.engine.fit(X, y)

    def predict(self, X):
        y_pred = self.engine.predict(X)
        return y_pred

    def crossValidate(self, X, y, cv=10):
        accuracies = cross_val_score(estimator=self.engine, X=X, y=y, cv=cv)
        return accuracies.mean(), accuracies.std()

    def classify(self, X, y, cv=10, verbose=False):
        self.fit(X, y)
        y_pred = self.predict(X)
        accuracies, std = self.crossValidate(X, y_pred, cv)
        if verbose:
            self.printSummary(accuracies, std)

        return accuracies, std

    def printSummary(self, accuracies, std):
        raise Exception('This method should be implemented in child class')