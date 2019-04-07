from app.AbstractModel import AbstractModel


class AbstractRegressor(AbstractModel):

    def printSummary(self, score, std, scoringFunction):
        raise Exception('This method should be implemented in child class')

    def estimate(self, X, y, scoring='neg_mean_absolute_error', cv=10, verbose=False):
        return super().estimate(X, y, scoring, cv, verbose)