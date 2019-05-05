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
original_dataframe = original_train_dataframe.append(original_test_dataframe, ignore_index=True)

# Auxiliary preprocessing concerning all records
preprocessor = DataPreprocessor(original_dataframe)
preprocessor.resolve_surname()
original_dataframe = preprocessor.dataframe

age_dataframe = original_dataframe.drop(labels=['Name', 'Cabin', 'Embarked', 'Fare', 'Pclass', 'Ticket', 'Survived'], axis=1)
age_dataframe = pd.get_dummies(age_dataframe, columns=['Sex', 'Surname'], drop_first=True)

# age_dataframe.drop(labels=['Surname'], axis=1, inplace=True)
# age_dataframe = pd.get_dummies(age_dataframe, columns=['Sex'], drop_first=True)



# Split them again based on if passenger's Age is known or not
age_train_dataframe = age_dataframe[~numpy.isnan(age_dataframe['Age'])]
age_test_dataframe = age_dataframe[numpy.isnan(age_dataframe['Age'])]

# Extract independent and dependent variable matrices
X = age_train_dataframe.drop(labels=['Age', 'PassengerId'], axis=1)
passenger_ids = original_dataframe._getitem_column('PassengerId')


y = age_train_dataframe._getitem_column('Age')
y = y.values.reshape(-1, 1)

# Backward Elimination
significance_level = 0.05
X = toolkit.backward_elimination_using_adjR2(X, y)
# X = toolkit.backward_elimination_using_pvalues(X, y, significance_level)

# reflect the same columns setup in age_test_dataframe
age_test_dataframe = age_test_dataframe[X.columns.values]

# add a column of 1s
X = smtools.add_constant(X)
age_test_dataframe = smtools.add_constant(age_test_dataframe)


# Fit the model using few regressors, cross-validate each one, pick the one with lowest MSE
def fit_and_estimate(regressor, X, y, scale_features=False):
    if scale_features:
        X, y = regressor.scaleFeatures(X, y)

    y = numpy.ravel(y)
    score, std = regressor.estimate(X, y, scoring='neg_mean_squared_error', verbose=True)
    score, std = regressor.estimate(X, y, scoring='neg_mean_absolute_error', verbose=True)
    score, std = regressor.estimate(X, y, scoring='explained_variance', verbose=True)
    score, std = regressor.estimate(X, y, scoring='r2', verbose=True)

regressor = SupportVectorRegressor(kernel='rbf')
fit_and_estimate(regressor, X, y, scale_features=True)

age_predicted = regressor.predict(age_test_dataframe)

age_test_dataframe['Age'] = age_predicted

original_dataframe['Age'].replace(numpy.nan, age_test_dataframe['Age'], inplace=True)


# @TODO merge predicted age back to original datasets; be careful not to confuse rows
    # DONE

# @TODO call this script from classify.py and make it pass the dataframes back there
# @TODO load data in classify.py as 1 dataset and pass it here
