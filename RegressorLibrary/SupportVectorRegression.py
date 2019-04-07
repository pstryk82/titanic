from sklearn.svm import SVR
from RegressorLibrary.AbstractRegressor import AbstractRegressor


class SupportVectorRegressor(AbstractRegressor):
    def __init__(self, kernel='rbf'):
        super().__init__()
        self.engine = SVR(kernel=kernel)

    def predict(self, X):
        y_pred = super().predict(X)
        y_pred = self.scaler_Y.inverse_transform(y_pred)
        return y_pred

    def printSummary(self, score, std, scoringFunction):
        print('Support Vector Regression mean ', scoringFunction, ': ', score)
        print('Support Vector Regression deviation: ', scoringFunction, ': ', std)
