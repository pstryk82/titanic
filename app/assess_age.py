# A separate model for assessing age of passengers where this information is not given.
import numpy
import pandas as pd
import statsmodels.tools as smtools
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
dataframe.drop(labels=['Name', 'Cabin', 'Embarked', 'Fare', 'Pclass', 'Survived', 'Ticket'], axis=1, inplace=True)

# Split them again based on if passenger's Age is known or not
train_dataframe = dataframe[~numpy.isnan(dataframe['Age'])]
test_dataframe = dataframe[numpy.isnan(dataframe['Age'])]
# note: this line achieves the same, but I don't fully understand how it works ;)
# train_dataframe, test_dataframe = [x for _, x in dataframe.groupby(numpy.isnan(dataframe['Age'])) ]

# Data preprocessing
train_dataframe = pd.get_dummies(train_dataframe, columns=['Sex', 'Surname'], drop_first=True)

# Extract independent and dependent variable matrices
X = train_dataframe.drop(labels=['PassengerId', 'Age'], axis=1)
y = train_dataframe._getitem_column('Age')

# Backward Elimination

# add a column of 1s
X = smtools.add_constant(X)

significance_level = 0.05
X = toolkit.backward_elimination_using_pvalues(X, y, significance_level)

# Fit the model using few regressors, cross-validate each one, pick the one with lowest MSE
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)
