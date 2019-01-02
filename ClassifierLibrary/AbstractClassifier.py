class AbstractClassifier:

    def classify(self, X, y, cv=10, verbose=False):
        raise Exception('This method should be implemented in child class')
