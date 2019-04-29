from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

from RegressorLibrary.AbstractRegressor import AbstractRegressor


class DecisionTreeRegressor(AbstractRegressor):
    def __init__(self):
        super().__init__()
        self.engine = SklearnDecisionTreeRegressor()

    def printSummary(self, score, std, scoringFunction):
        print('Decision Tree Regression ', scoringFunction, ': ', score)
        print('Decision Tree Regression deviation: ', scoringFunction, ': ', std)
