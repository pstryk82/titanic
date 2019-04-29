# A separate model for assessing age of passengers where this information is not given.

import numpy
import pandas as pd
import statsmodels.tools as smtools
import matplotlib.pyplot as plt

from RegressorLibrary.DecisionTree import DecisionTreeRegressor
from RegressorLibrary.Linear import LinearRegressor
from RegressorLibrary.SupportVectorRegression import SupportVectorRegressor
from app import toolkit
from app.DataPreprocessor import DataPreprocessor

# Merge train and test datasets into one
original_train_dataframe = pd.read_csv('./dataset/train.csv')
original_test_dataframe = pd.read_csv('./dataset/test.csv')
dataframe = original_train_dataframe.append(original_test_dataframe, ignore_index=True)

# Auxiliary preprocessing concerning all records
preprocessor = DataPreprocessor(dataframe)
preprocessor.resolve_surname()
dataframe = preprocessor.dataframe
dataframe.drop(labels=['PassengerId', 'Name', 'Cabin', 'Embarked', 'Fare', 'Pclass', 'Ticket'], axis=1, inplace=True)
dataframe.drop(labels=['Survived'], axis=1, inplace=True)

# Split them again based on if passenger's Age is known or not
known_dataframe = dataframe[~numpy.isnan(dataframe['Age'])]
prod_dataframe = dataframe[numpy.isnan(dataframe['Age'])]
# note: this line achieves the same, but I don't fully understand how it works ;)
# known_dataframe, prod_dataframe = [x for _, x in dataframe.groupby(numpy.isnan(dataframe['Age'])) ]

# Let's draw some stuff to see things better
def draw_plots():
    plt.scatter(known_dataframe.iloc[:, 1], known_dataframe.iloc[:, 0], marker='o')  # Parch
    plt.scatter(known_dataframe.iloc[:, 4], known_dataframe.iloc[:, 0], marker='+')  # SibSp
    plt.scatter(known_dataframe.iloc[:, 7], known_dataframe.iloc[:, 0], marker='x')  # FamilySize
    plt.xlabel('Parch');
    plt.ylabel('Age')
    plt.legend()
    # plt.colorbar()
    plt.show()

# draw_plots()


# Data preprocessing
known_dataframe = pd.get_dummies(known_dataframe, columns=['Sex', 'Surname'], drop_first=True)

# Extract independent and dependent variable matrices
X = known_dataframe.drop(labels=['Age'], axis=1)
# add a column of 1s
X = smtools.add_constant(X)

y = known_dataframe._getitem_column('Age')
y = y.values.reshape(-1, 1)

# Backward Elimination
significance_level = 0.05
X = toolkit.backward_elimination_using_adjR2(X, y)
# X = toolkit.backward_elimination_using_pvalues(X, y, significance_level)


# Fit the model using few regressors, cross-validate each one, pick the one with lowest MSE
def fit_and_estimate(regressor, X, y, scale_features=False):
    if scale_features:
        X, y = regressor.scaleFeatures(X, y)

    y = numpy.ravel(y)
    score, std = regressor.estimate(X, y, scoring='neg_mean_squared_error', verbose=True)
    score, std = regressor.estimate(X, y, scoring='neg_mean_absolute_error', verbose=True)
    score, std = regressor.estimate(X, y, scoring='explained_variance', verbose=True)
    score, std = regressor.estimate(X, y, scoring='r2', verbose=True)

svr_regressor = SupportVectorRegressor(kernel='rbf')
fit_and_estimate(svr_regressor, X, y, scale_features=True)

decision_tree_regressor = DecisionTreeRegressor()
fit_and_estimate(decision_tree_regressor, X, y)

linear_regressor = LinearRegressor()
fit_and_estimate(linear_regressor, X, y)


# @TODO add more regressors and pick the most accurate one
#       DONE: stick to SVR
# @TODO call this script from classify.py and make it pass the dataframes back there
# @TODO load data in classify.py as 1 dataset and pass it here
