from sklearn.linear_model import LinearRegression

from RegressorLibrary.AbstractRegressor import AbstractRegressor


class LinearRegressor(AbstractRegressor):
    def __init__(self,):
        super().__init__()
        self.engine = LinearRegression()

    def printSummary(self, score, std, scoringFunction):
        print('Linear Regression ', scoringFunction, ': ', score)
        print('Linear Regression deviation: ', scoringFunction, ': ', std)
