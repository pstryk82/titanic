from app.AbstractModel import AbstractModel

class AbstractClassifier(AbstractModel):

    def printSummary(self, score, std, scoringFunction):
        raise Exception('This method should be implemented in child class')

    def estimate(self, X, y, scoring='accuracy', cv=10, verbose=False):
        return super().estimate(X, y, scoring, cv, verbose)