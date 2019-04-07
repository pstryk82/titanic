from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class AbstractModel:
    def __init__(self):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

    def fit(self, X, y):
        self.engine.fit(X, y)

    def predict(self, X):
        y_pred = self.engine.predict(X)
        return y_pred

    def scaleFeatures(self, X, y):
        X = self.scaler_X.fit_transform(X)
        # y.reshape(-1, 1)
        y = self.scaler_Y.fit_transform(y)
        return X, y

    def crossValidate(self, X, y, cv=10, scoring='accuracy'):
        accuracies = cross_val_score(estimator=self.engine, X=X, y=y, cv=cv, scoring=scoring)
        return accuracies.mean(), accuracies.std()

    def estimate(self, X, y, scoring, cv=10, verbose=False):
        self.fit(X, y)
        score, std = self.crossValidate(X=X, y=y, cv=cv, scoring=scoring)
        if verbose:
            self.printSummary(score, std, scoring)

        return score, std